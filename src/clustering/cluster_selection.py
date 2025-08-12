"""
聚类-轮选数据选择核心逻辑

实现基于聚类的数据选择策略：
1. 使用GPU聚类算法对数据进行聚类
2. 从每个簇中轮流选择质量最高的数据
3. 确保选择的数据量符合目标要求
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from .hdbscan_clustering import GPUHDBSCANClustering
from .kmeans_clustering import GPUKMeansClustering


class ClusterBasedSelection:
    """基于聚类的数据选择器"""

    def __init__(self, device: torch.device, random_state: int = 42, debug_print: bool = False):
        """
        初始化聚类选择器

        Args:
            device: GPU设备
            random_state: 随机种子
            debug_print: 是否启用调试输出
        """
        self.device = device
        self.random_state = random_state
        self.debug_print = debug_print
        self.logger = logging.getLogger(__name__)

        # 初始化聚类器
        self.kmeans_clusterer = GPUKMeansClustering(device, random_state, debug_print)
        self.hdbscan_clusterer = GPUHDBSCANClustering(device, random_state, debug_print)

    def select_data_by_clustering(
        self,
        scored_data: List[Dict],
        all_logits_by_dataset: Dict[str, List[torch.Tensor]],
        target_count: int,
        clustering_method: str = "kmeans",
        clustering_params: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        使用聚类-轮选策略选择数据

        Args:
            scored_data: 评分后的数据列表
            all_logits_by_dataset: 按数据集分组的logits张量
            target_count: 目标选择数量
            clustering_method: 聚类方法 ('kmeans' 或 'hdbscan')
            clustering_params: 聚类参数

        Returns:
            选择后的数据列表
        """
        self.logger.info(f"开始聚类-轮选数据选择，目标数量: {target_count}")
        self.logger.info(f"聚类方法: {clustering_method}")

        if clustering_params is None:
            clustering_params = {}

        # 1. 准备特征数据
        features, sample_mapping = self._prepare_clustering_features(scored_data, all_logits_by_dataset)

        if len(features) == 0:
            self.logger.error("没有可用的特征数据进行聚类")
            return scored_data[:target_count]

        # 2. 执行聚类
        cluster_labels, cluster_info = self._perform_clustering(features, clustering_method, clustering_params)

        # 3. 聚类-轮选策略选择数据
        selected_data = self._round_robin_selection_from_clusters(sample_mapping, cluster_labels, target_count)

        self.logger.info(f"聚类-轮选完成，选择了 {len(selected_data)} 个样本")
        self.logger.info(f"聚类信息: {cluster_info}")

        return selected_data

    def _prepare_clustering_features(
        self,
        scored_data: List[Dict],
        all_logits_by_dataset: Dict[str, List[torch.Tensor]],
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """准备聚类特征数据"""
        self.logger.info("准备聚类特征数据...")

        # 收集所有logits张量作为聚类特征
        all_features = []
        sample_mapping = []

        # 按scored_data的顺序重新组织logits
        dataset_logits_index = {name: 0 for name in all_logits_by_dataset.keys()}

        for data_item in scored_data:
            dataset_name = data_item["dataset"]
            current_idx = dataset_logits_index.get(dataset_name, 0)

            if dataset_name in all_logits_by_dataset and current_idx < len(all_logits_by_dataset[dataset_name]):
                logits_tensor = all_logits_by_dataset[dataset_name][current_idx]

                # 注意：logits_tensor已经是概率分布（在selection.py中通过softmax转换得到）
                # 直接使用概率分布，不进行重复的softmax转换
                probs_tensor = logits_tensor  # [L, E] 概率分布

                # 将 [L, E] 概率分布展平为一维特征向量 [L*E]
                flattened_features = probs_tensor.flatten()  # [L*E]
                all_features.append(flattened_features)
                sample_mapping.append(data_item)

                dataset_logits_index[dataset_name] += 1
            else:
                self.logger.warning(f"数据集 {dataset_name} 的logits不足或不存在")

        if len(all_features) == 0:
            self.logger.error("没有收集到任何特征数据")
            return torch.empty(0, 0, device=self.device), []

        # 堆叠为特征矩阵 [N, L*E]，确保使用float32类型
        features_matrix = torch.stack(all_features).to(device=self.device, dtype=torch.float32)

        self.logger.info(f"聚类特征矩阵形状: {features_matrix.shape}")
        self.logger.info(f"有效样本数: {len(sample_mapping)}")

        if self.debug_print:
            self.logger.info(f"特征向量维度: {features_matrix.shape[1]}")
            if features_matrix.shape[0] > 0:
                self.logger.info(f"特征数据类型: {features_matrix.dtype}")

        return features_matrix, sample_mapping

    def _perform_clustering(
        self,
        features: torch.Tensor,
        method: str,
        params: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        """执行聚类"""
        self.logger.info(f"开始 {method} 聚类...")

        if method.lower() == "kmeans":
            return self._kmeans_clustering(features, params)
        elif method.lower() == "hdbscan":
            return self._hdbscan_clustering(features, params)
        else:
            raise ValueError(f"不支持的聚类方法: {method}")

    def _kmeans_clustering(self, features: torch.Tensor, params: Dict) -> Tuple[torch.Tensor, Dict]:
        """执行K-Means聚类"""
        auto_k = params.get("auto_k", True)
        k = params.get("k", None)
        k_range = params.get("k_range", None)
        max_iters = params.get("max_iters", 300)

        # 并行相关参数
        enable_parallel = params.get("enable_parallel_kmeans", False)
        parallel_processes = params.get("parallel_processes", 4)
        gpu_allocation_strategy = params.get("gpu_allocation_strategy", "round_robin")

        if self.debug_print and enable_parallel and auto_k:
            self.logger.info(f"启用K-means并行模式: 进程数={parallel_processes}, 分配策略={gpu_allocation_strategy}")

        labels, info = self.kmeans_clusterer.fit_predict(
            features,
            k=k,
            auto_k=auto_k,
            k_range=k_range,
            max_iters=max_iters,
            enable_parallel=enable_parallel,
            parallel_processes=parallel_processes,
            gpu_allocation_strategy=gpu_allocation_strategy,
        )

        return labels, info

    def _hdbscan_clustering(self, features: torch.Tensor, params: Dict) -> Tuple[torch.Tensor, Dict]:
        """执行HDBSCAN聚类"""
        min_cluster_size = params.get("min_cluster_size", None)
        min_samples = params.get("min_samples", None)
        metric = params.get("metric", "cosine")
        use_gpu = params.get("use_gpu", True)
        auto_tune = params.get("auto_tune", False)

        if auto_tune:
            # 自动调参
            tuning_result = self.hdbscan_clusterer.auto_tune_parameters(features, use_gpu=use_gpu)
            return tuning_result["best_labels"], tuning_result["best_info"]
        else:
            # 使用指定参数
            labels, info = self.hdbscan_clusterer.fit_predict(
                features,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric=metric,
                use_gpu=use_gpu,
            )
            return labels, info

    def _round_robin_selection_from_clusters(
        self,
        sample_mapping: List[Dict],
        cluster_labels: torch.Tensor,
        target_count: int,
    ) -> List[Dict]:
        """从各簇中轮流选择高质量数据"""
        self.logger.info("开始聚类-轮选数据选择...")

        # 1. 按簇组织数据
        clusters = self._organize_data_by_clusters(sample_mapping, cluster_labels)

        # 过滤掉噪声簇（标签为-1）
        valid_clusters = {k: v for k, v in clusters.items() if k >= 0}

        if not valid_clusters:
            self.logger.warning("没有有效的聚类，回退到质量分数排序")
            sample_mapping.sort(key=lambda x: x["scores"], reverse=True)
            return sample_mapping[:target_count]

        self.logger.info(f"有效簇数: {len(valid_clusters)}")
        for cluster_id, cluster_data in valid_clusters.items():
            self.logger.info(f"簇 {cluster_id}: {len(cluster_data)} 个样本")

        # 2. 在每个簇内按质量分数排序
        for cluster_data in valid_clusters.values():
            cluster_data.sort(key=lambda x: x["scores"], reverse=True)

        # 3. 轮流从各簇中选择数据
        selected_data = []
        cluster_indices = {cluster_id: 0 for cluster_id in valid_clusters.keys()}

        with tqdm(total=target_count, desc="聚类-轮选") as pbar:
            while len(selected_data) < target_count:
                # 一轮选择
                round_selections = 0

                for cluster_id in sorted(valid_clusters.keys()):
                    if len(selected_data) >= target_count:
                        break

                    current_idx = cluster_indices[cluster_id]
                    cluster_data = valid_clusters[cluster_id]

                    # 从当前簇选择下一个最高质量的样本
                    if current_idx < len(cluster_data):
                        selected_sample = cluster_data[current_idx]
                        selected_data.append(selected_sample)
                        cluster_indices[cluster_id] += 1
                        round_selections += 1
                        pbar.update(1)

                # 如果这轮没有选择任何样本，说明所有簇都用完了
                if round_selections == 0:
                    self.logger.warning("所有簇的数据都已用完，但未达到目标数量")
                    break

        # 4. 如果选择的数据超过目标数量，删除质量最低的样本
        if len(selected_data) > target_count:
            self.logger.info(f"选择了 {len(selected_data)} 个样本，需要删除 {len(selected_data) - target_count} 个")
            # 按质量分数排序，保留前target_count个
            selected_data.sort(key=lambda x: x["scores"], reverse=True)
            selected_data = selected_data[:target_count]

        self.logger.info(f"最终选择了 {len(selected_data)} 个样本")
        if selected_data:
            scores = [d["scores"] for d in selected_data]
            self.logger.info(f"质量分数范围: {min(scores):.6f} ~ {max(scores):.6f}")

        return selected_data

    def _organize_data_by_clusters(
        self,
        sample_mapping: List[Dict],
        cluster_labels: torch.Tensor,
    ) -> Dict[int, List[Dict]]:
        """按聚类标签组织数据"""
        clusters = {}

        for i, sample in enumerate(sample_mapping):
            cluster_id = cluster_labels[i].item()

            if cluster_id not in clusters:
                clusters[cluster_id] = []

            clusters[cluster_id].append(sample)

        return clusters

    def analyze_cluster_quality(
        self,
        sample_mapping: List[Dict],
        cluster_labels: torch.Tensor,
    ) -> Dict:
        """分析各簇的质量分布"""
        clusters = self._organize_data_by_clusters(sample_mapping, cluster_labels)

        cluster_analysis = {}

        for cluster_id, cluster_data in clusters.items():
            if cluster_id >= 0:  # 排除噪声簇
                scores = [d["scores"] for d in cluster_data]
                cluster_analysis[cluster_id] = {
                    "size": len(cluster_data),
                    "mean_score": sum(scores) / len(scores),
                    "max_score": max(scores),
                    "min_score": min(scores),
                    "score_std": torch.tensor(scores).std().item(),
                }

        return cluster_analysis
