"""
GPU加速HDBSCAN聚类实现

提供基于RAPIDS cuML的GPU HDBSCAN聚类：
- 支持余弦距离
- 无需预设参数
- 自适应确定簇数
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch

try:
    from cuml.cluster import HDBSCAN

    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False

from sklearn.cluster import HDBSCAN as SklearnHDBSCAN

from .gpu_metrics import gpu_silhouette_score_cosine


class GPUHDBSCANClustering:
    """GPU加速的HDBSCAN聚类实现"""

    def __init__(self, device: torch.device, random_state: int = 42, debug_print: bool = False):
        """
        初始化GPU HDBSCAN聚类器

        Args:
            device: GPU设备
            random_state: 随机种子
            debug_print: 是否启用调试输出
        """
        self.device = device
        self.random_state = random_state
        self.debug_print = debug_print
        self.logger = logging.getLogger(__name__)

        if not CUML_AVAILABLE:
            self.logger.warning("RAPIDS cuML不可用，将回退到CPU版本的sklearn HDBSCAN")

    def fit_predict(
        self,
        data: torch.Tensor,
        min_cluster_size: Optional[int] = None,
        min_samples: Optional[int] = None,
        metric: str = "cosine",
        use_gpu: bool = True,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        执行HDBSCAN聚类

        Args:
            data: 输入数据 [N, D]
            min_cluster_size: 最小簇大小（自动估计如果为None）
            min_samples: 最小样本数（自动设置如果为None）
            metric: 距离度量方法
            use_gpu: 是否使用GPU加速

        Returns:
            labels: 聚类标签 [N]，-1表示噪声点
            info: 聚类信息字典
        """
        n_samples = data.shape[0]

        # 自动设置参数
        if min_cluster_size is None:
            # 启发式：样本数的平方根的一半，但不少于10
            min_cluster_size = max(10, int(np.sqrt(n_samples) / 2))

        if min_samples is None:
            # 通常设置为min_cluster_size
            min_samples = min_cluster_size

        self.logger.info(f"HDBSCAN参数: min_cluster_size={min_cluster_size}, min_samples={min_samples}")

        # 尝试使用GPU版本
        if use_gpu and CUML_AVAILABLE and self.device.type == "cuda":
            try:
                if self.debug_print:
                    self.logger.info("使用GPU HDBSCAN聚类...")
                labels, info = self._fit_predict_gpu(data, min_cluster_size, min_samples, metric)
                self.logger.info("使用GPU HDBSCAN聚类成功")
                return labels, info
            except Exception as e:
                self.logger.warning(f"GPU HDBSCAN失败: {e}，回退到CPU版本")

        # 回退到CPU版本
        if self.debug_print:
            self.logger.info("使用CPU HDBSCAN聚类...")
        labels, info = self._fit_predict_cpu(data, min_cluster_size, min_samples, metric)
        self.logger.info("使用CPU HDBSCAN聚类")
        return labels, info

    def _fit_predict_gpu(
        self,
        data: torch.Tensor,
        min_cluster_size: int,
        min_samples: int,
        metric: str,
    ) -> Tuple[torch.Tensor, Dict]:
        """GPU版本的HDBSCAN聚类（基于概率分布）"""
        # 转换数据到cuDF格式（不进行手动归一化）
        if metric == "cosine":
            # 直接使用概率分布数据，不进行L2标准化
            data_cupy = data.detach().cpu().numpy()
            # 保持余弦距离度量
            metric = "cosine"
        else:
            data_cupy = data.detach().cpu().numpy()

        # 初始化cuML HDBSCAN
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            cluster_selection_epsilon=0.0,
        )

        # 执行聚类
        labels_np = clusterer.fit_predict(data_cupy)
        labels = torch.tensor(labels_np, device=self.device)

        # 计算聚类统计
        unique_labels = torch.unique(labels)
        n_clusters = (unique_labels >= 0).sum().item()  # 排除噪声标签-1
        n_noise = (labels == -1).sum().item()

        # 计算轮廓系数（只针对非噪声点）
        silhouette_avg = 0.0
        if n_clusters > 1:
            # 过滤噪声点
            non_noise_mask = labels >= 0
            if non_noise_mask.sum() > 1:
                # 使用GPU加速的余弦距离轮廓系数计算
                data_filtered = data[non_noise_mask]
                labels_filtered = labels[non_noise_mask]

                try:
                    silhouette_avg = gpu_silhouette_score_cosine(data_filtered, labels_filtered)
                except Exception as e:
                    self.logger.warning(f"轮廓系数计算失败: {e}")

        info = {
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "noise_ratio": n_noise / len(labels),
            "silhouette_score": silhouette_avg,
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "metric": metric,
            "cluster_sizes": self._compute_cluster_sizes(labels, n_clusters),
        }

        return labels, info

    def _fit_predict_cpu(
        self,
        data: torch.Tensor,
        min_cluster_size: int,
        min_samples: int,
        metric: str,
    ) -> Tuple[torch.Tensor, Dict]:
        """CPU版本的HDBSCAN聚类"""
        # 转换到CPU numpy
        data_np = data.cpu().numpy()

        # 初始化sklearn HDBSCAN
        clusterer = SklearnHDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            cluster_selection_epsilon=0.0,
            n_jobs=-1,  # 使用所有CPU核心
        )

        # 执行聚类
        labels_np = clusterer.fit_predict(data_np)
        labels = torch.tensor(labels_np, device=self.device)

        # 计算聚类统计
        unique_labels = torch.unique(labels)
        n_clusters = (unique_labels >= 0).sum().item()
        n_noise = (labels == -1).sum().item()

        # 计算轮廓系数（只针对非噪声点）
        silhouette_avg = 0.0
        if n_clusters > 1:
            # 过滤噪声点
            non_noise_mask = labels >= 0
            # 使用GPU加速的余弦距离轮廓系数计算
            if non_noise_mask.sum() > 1:
                data_filtered = data[non_noise_mask]
                labels_filtered = labels[non_noise_mask]

                try:
                    silhouette_avg = gpu_silhouette_score_cosine(data_filtered, labels_filtered)
                except Exception as e:
                    self.logger.warning(f"轮廓系数计算失败: {e}")

        info = {
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "noise_ratio": n_noise / len(labels),
            "silhouette_score": silhouette_avg,
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "metric": metric,
            "cluster_sizes": self._compute_cluster_sizes(labels, n_clusters),
        }

        return labels, info

    def _compute_cluster_sizes(self, labels: torch.Tensor, n_clusters: int) -> Dict[int, int]:
        """计算各簇的大小"""
        cluster_sizes = {}
        unique_labels = torch.unique(labels)

        for label in unique_labels:
            label_int = label.item()
            if label_int >= 0:  # 排除噪声标签-1
                cluster_sizes[label_int] = (labels == label).sum().item()

        return cluster_sizes

    def auto_tune_parameters(
        self,
        data: torch.Tensor,
        min_cluster_size_range: Optional[Tuple[int, int]] = None,
        use_gpu: bool = True,
    ) -> Dict:
        """
        自动调参寻找最佳HDBSCAN参数

        Args:
            data: 输入数据 [N, D]
            min_cluster_size_range: min_cluster_size搜索范围
            use_gpu: 是否使用GPU

        Returns:
            最佳参数和对应结果的字典
        """
        n_samples = data.shape[0]

        # 设置搜索范围
        if min_cluster_size_range is None:
            min_size = max(5, int(np.sqrt(n_samples) / 4))
            max_size = min(int(np.sqrt(n_samples)), n_samples // 10)
            min_cluster_size_range = (min_size, max_size)

        min_size, max_size = min_cluster_size_range
        candidates = np.logspace(np.log10(min_size), np.log10(max_size), num=5).astype(int)
        candidates = sorted(set(candidates))  # 去重并排序

        best_score = -1
        best_params = None
        best_result = None

        self.logger.info(f"调参候选值: min_cluster_size = {candidates}")

        for min_cluster_size in candidates:
            if self.debug_print:
                self.logger.info(f"测试 min_cluster_size={min_cluster_size}...")

            labels, info = self.fit_predict(
                data,
                min_cluster_size=min_cluster_size,
                min_samples=min_cluster_size,
                use_gpu=use_gpu,
            )

            # 评估指标：轮廓系数 - 噪声点惩罚
            score = info["silhouette_score"] - 0.1 * info["noise_ratio"]

            if self.debug_print:
                self.logger.info(
                    f"min_cluster_size={min_cluster_size}: "
                    f"n_clusters={info['n_clusters']}, "
                    f"silhouette={info['silhouette_score']:.3f}, "
                    f"noise_ratio={info['noise_ratio']:.3f}, "
                    f"score={score:.3f}"
                )
            else:
                self.logger.debug(
                    f"min_cluster_size={min_cluster_size}: "
                    f"n_clusters={info['n_clusters']}, "
                    f"silhouette={info['silhouette_score']:.3f}, "
                    f"noise_ratio={info['noise_ratio']:.3f}, "
                    f"score={score:.3f}"
                )

            if score > best_score and info["n_clusters"] > 0:
                best_score = score
                best_params = {
                    "min_cluster_size": min_cluster_size,
                    "min_samples": min_cluster_size,
                }
                best_result = (labels, info)

        if best_params is None:
            # 如果所有参数都失败，使用默认参数
            self.logger.warning("自动调参失败，使用默认参数")
            labels, info = self.fit_predict(data, use_gpu=use_gpu)
            best_params = {
                "min_cluster_size": info["min_cluster_size"],
                "min_samples": info["min_samples"],
            }
            best_result = (labels, info)
            best_score = info["silhouette_score"]

        self.logger.info(f"最佳参数: {best_params}, 得分: {best_score:.3f}")

        return {
            "best_params": best_params,
            "best_score": best_score,
            "best_labels": best_result[0],
            "best_info": best_result[1],
        }
