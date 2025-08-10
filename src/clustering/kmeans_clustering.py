"""
GPU加速K-Means聚类实现

使用PyTorch实现GPU加速的K-Means聚类，支持：
- 余弦距离度量
- Elbow Method自动k值选择
- 轮廓系数验证
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
from sklearn.metrics import silhouette_score
from tqdm import tqdm


class GPUKMeansClustering:
    """GPU加速的K-Means聚类实现"""

    def __init__(self, device: torch.device, random_state: int = 42):
        """
        初始化GPU K-Means聚类器

        Args:
            device: GPU设备
            random_state: 随机种子
        """
        self.device = device
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)

    def kmeans_cosine(
        self,
        data: torch.Tensor,
        k: int,
        max_iters: int = 300,
        tol: float = 1e-4,
        init_method: str = "k-means++",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用余弦距离的K-Means聚类

        Args:
            data: 输入数据 [N, D]
            k: 聚类数量
            max_iters: 最大迭代次数
            tol: 收敛容忍度
            init_method: 初始化方法 ('k-means++' 或 'random')

        Returns:
            centers: 聚类中心 [K, D]
            labels: 聚类标签 [N]
        """
        n_samples, n_features = data.shape

        # L2标准化数据以支持余弦距离
        data_normalized = torch.nn.functional.normalize(data, p=2, dim=1)

        # 初始化聚类中心
        if init_method == "k-means++":
            centers = self._kmeans_plus_plus_init(data_normalized, k)
        else:
            # 随机初始化
            torch.manual_seed(self.random_state)
            indices = torch.randperm(n_samples, device=self.device)[:k]
            centers = data_normalized[indices].clone()

        centers = torch.nn.functional.normalize(centers, p=2, dim=1)

        for iteration in range(max_iters):
            # 计算余弦相似度（使用点积，因为数据已标准化）
            similarities = torch.mm(data_normalized, centers.t())  # [N, K]

            # 分配到最相似的中心
            labels = similarities.argmax(dim=1)  # [N]

            # 更新中心
            new_centers = torch.zeros_like(centers)
            for i in range(k):
                mask = labels == i
                if mask.sum() > 0:
                    cluster_points = data_normalized[mask]
                    new_center = cluster_points.mean(dim=0)
                    new_centers[i] = torch.nn.functional.normalize(new_center, p=2, dim=0)
                else:
                    # 如果某个簇为空，保持原中心
                    new_centers[i] = centers[i]

            # 检查收敛
            center_shift = torch.norm(new_centers - centers, dim=1).max()
            centers = new_centers

            if center_shift < tol:
                self.logger.debug(f"K-Means收敛于第{iteration + 1}次迭代")
                break

        return centers, labels

    def _kmeans_plus_plus_init(self, data: torch.Tensor, k: int) -> torch.Tensor:
        """K-Means++初始化方法"""
        n_samples, n_features = data.shape
        centers = torch.zeros(k, n_features, device=self.device)

        # 设置随机种子
        torch.manual_seed(self.random_state)

        # 第一个中心随机选择
        first_idx = torch.randint(0, n_samples, (1,), device=self.device)
        centers[0] = data[first_idx]

        for i in range(1, k):
            # 计算到已有中心的最小距离（使用余弦距离）
            similarities = torch.mm(data, centers[:i].t())  # [N, i]
            max_similarities = similarities.max(dim=1)[0]  # [N]
            distances = 1 - max_similarities  # 余弦距离
            distances = torch.clamp(distances, min=0.0)  # 确保非负

            # 按概率选择下一个中心
            probabilities = distances / distances.sum()
            cumulative_probs = torch.cumsum(probabilities, dim=0)

            r = torch.rand(1, device=self.device)
            next_idx = (cumulative_probs >= r).nonzero(as_tuple=True)[0][0]
            centers[i] = data[next_idx]

        return centers

    def compute_inertia_cosine(self, data: torch.Tensor, centers: torch.Tensor, labels: torch.Tensor) -> float:
        """计算基于余弦距离的惯性值"""
        data_normalized = torch.nn.functional.normalize(data, p=2, dim=1)
        centers_normalized = torch.nn.functional.normalize(centers, p=2, dim=1)

        inertia = 0.0
        for i in range(centers.shape[0]):
            mask = labels == i
            if mask.sum() > 0:
                cluster_data = data_normalized[mask]  # [cluster_size, D]
                cluster_center = centers_normalized[i : i + 1]  # [1, D]

                # 余弦相似度
                similarities = torch.mm(cluster_data, cluster_center.t()).squeeze()
                # 余弦距离
                distances = 1 - similarities
                inertia += distances.sum().item()

        return inertia

    def find_optimal_k_elbow(
        self,
        data: torch.Tensor,
        k_range: Optional[Tuple[int, int]] = None,
        max_iters: int = 300,
        n_runs: int = 3,
    ) -> Dict:
        """
        使用Elbow Method自动选择最优k值

        Args:
            data: 输入数据 [N, D]
            k_range: k值搜索范围 (min_k, max_k)
            max_iters: K-Means最大迭代次数
            n_runs: 每个k值运行次数（选择最佳结果）

        Returns:
            包含最优k值和相关指标的字典
        """
        n_samples = data.shape[0]

        # 自动确定k值范围
        if k_range is None:
            min_k = max(2, int(n_samples**0.5 / 10))  # 启发式下界
            max_k = min(int(n_samples**0.5), 200)  # 启发式上界
            k_range = (min_k, max_k)

        min_k, max_k = k_range
        self.logger.info(f"在范围[{min_k}, {max_k}]中搜索最优k值")

        k_values = list(range(min_k, max_k + 1, max(1, (max_k - min_k) // 20)))  # 最多20个候选k值
        inertias = []
        silhouette_scores = []

        for k in tqdm(k_values, desc="Elbow Method搜索"):
            best_inertia = float("inf")
            best_labels = None

            # 多次运行选择最佳结果
            for run in range(n_runs):
                centers, labels = self.kmeans_cosine(data, k, max_iters=max_iters)
                inertia = self.compute_inertia_cosine(data, centers, labels)

                if inertia < best_inertia:
                    best_inertia = inertia
                    best_labels = labels

            inertias.append(best_inertia)

            # 计算轮廓系数（转换为CPU进行计算）
            if k > 1:
                data_cpu = data.cpu().numpy()
                labels_cpu = best_labels.cpu().numpy()
                # 使用余弦距离计算轮廓系数
                sil_score = silhouette_score(data_cpu, labels_cpu, metric="cosine")
                silhouette_scores.append(sil_score)
            else:
                silhouette_scores.append(0.0)

        # 寻找Elbow点
        optimal_k = self._find_elbow_point(k_values, inertias)

        # 找到最佳轮廓系数对应的k
        best_silhouette_k = k_values[torch.tensor(silhouette_scores).argmax().item()]

        self.logger.info(f"Elbow方法推荐k={optimal_k}")
        self.logger.info(f"轮廓系数最高k={best_silhouette_k}")

        return {
            "optimal_k_elbow": optimal_k,
            "optimal_k_silhouette": best_silhouette_k,
            "k_values": k_values,
            "inertias": inertias,
            "silhouette_scores": silhouette_scores,
            "recommended_k": optimal_k,  # 优先使用Elbow方法
        }

    def _find_elbow_point(self, k_values: List[int], inertias: List[float]) -> int:
        """寻找Elbow点"""
        if len(inertias) < 3:
            return k_values[0]

        # 计算二阶差分
        inertias_tensor = torch.tensor(inertias, dtype=torch.float32)

        # 标准化惯性值到[0,1]范围
        inertias_norm = (inertias_tensor - inertias_tensor.min()) / (inertias_tensor.max() - inertias_tensor.min() + 1e-8)

        # 计算一阶和二阶差分
        first_diff = inertias_norm[1:] - inertias_norm[:-1]
        second_diff = first_diff[1:] - first_diff[:-1]

        # 寻找二阶差分最大的点（最大曲率）
        if len(second_diff) > 0:
            elbow_idx = second_diff.argmax().item() + 1  # +1因为二阶差分索引偏移
            if elbow_idx < len(k_values):
                return k_values[elbow_idx]

        # 如果找不到明显的肘部，返回中间值
        return k_values[len(k_values) // 2]

    def fit_predict(
        self,
        data: torch.Tensor,
        k: Optional[int] = None,
        auto_k: bool = True,
        k_range: Optional[Tuple[int, int]] = None,
        max_iters: int = 300,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        执行聚类并返回标签

        Args:
            data: 输入数据 [N, D]
            k: 指定聚类数量（如果auto_k=False）
            auto_k: 是否自动选择k值
            k_range: k值搜索范围
            max_iters: 最大迭代次数

        Returns:
            labels: 聚类标签 [N]
            info: 聚类信息字典
        """
        if auto_k:
            # 自动选择k值
            k_info = self.find_optimal_k_elbow(data, k_range, max_iters)
            optimal_k = k_info["recommended_k"]
            self.logger.info(f"自动选择的k值: {optimal_k}")
        else:
            if k is None:
                raise ValueError("当auto_k=False时必须指定k值")
            optimal_k = k
            k_info = {}

        # 执行最终聚类
        centers, labels = self.kmeans_cosine(data, optimal_k, max_iters=max_iters)
        final_inertia = self.compute_inertia_cosine(data, centers, labels)

        # 计算最终轮廓系数
        if optimal_k > 1:
            data_cpu = data.cpu().numpy()
            labels_cpu = labels.cpu().numpy()
            final_silhouette = silhouette_score(data_cpu, labels_cpu, metric="cosine")
        else:
            final_silhouette = 0.0

        info = {
            "n_clusters": optimal_k,
            "inertia": final_inertia,
            "silhouette_score": final_silhouette,
            "centers": centers,
            **k_info,  # 包含k值搜索的详细信息
        }

        return labels, info
