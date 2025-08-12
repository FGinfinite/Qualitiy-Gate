# src/clustering/gpu_metrics.py
"""
GPU加速的聚类评估指标实现
"""

import logging
from typing import Optional

import torch
from tqdm import tqdm


def gpu_silhouette_score_cosine(
    data: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: Optional[int] = None,
) -> float:
    """
    GPU上精确计算余弦距离的轮廓系数（基于概率分布特征）

    Args:
        data: [N, D] 概率分布特征矩阵，每行为一个样本的概率分布特征向量
        labels: [N] 聚类标签
        chunk_size: 分块计算的大小，用于节省GPU内存。如果为None则自动选择

    Returns:
        精确的轮廓系数值，使用标准余弦相似度计算

    Note:
        输入的data应该是概率分布特征（如softmax后的logits），
        函数内部会自动使用标准余弦相似度计算，无需预先归一化
    """
    logger = logging.getLogger(__name__)

    if data.shape[0] != labels.shape[0]:
        raise ValueError(f"数据样本数 {data.shape[0]} 与标签数 {labels.shape[0]} 不匹配")

    n_samples = data.shape[0]
    unique_labels = torch.unique(labels)
    n_clusters = len(unique_labels)

    # 边界情况处理
    if n_clusters <= 1:
        logger.info("聚类数量 <= 1，轮廓系数为0")
        return 0.0

    if n_samples <= 1:
        logger.info("样本数量 <= 1，轮廓系数为0")
        return 0.0

    # 自动选择分块大小以控制内存使用
    if chunk_size is None:
        # 估算距离矩阵大小 (bytes)
        matrix_size_bytes = n_samples * n_samples * 4  # float32
        max_memory_bytes = 20 * 1024**3  # 20GB 限制

        if matrix_size_bytes <= max_memory_bytes:
            # 可以使用全矩阵计算
            logger.info("可以使用全距离矩阵计算")
            return _gpu_silhouette_full_matrix(data, labels, unique_labels)
        else:
            # 使用分块计算
            chunk_size = max(100, int((max_memory_bytes / (n_samples * 4)) ** 0.5)) + 1000
            logger.info(f"使用分块计算，chunk_size={chunk_size}")
            return _gpu_silhouette_chunked(data, labels, unique_labels, chunk_size)
    else:
        logger.info(f"使用分块计算，chunk_size={chunk_size}")
        return _gpu_silhouette_chunked(data, labels, unique_labels, chunk_size)


def _gpu_silhouette_full_matrix(
    data: torch.Tensor,
    labels: torch.Tensor,
    unique_labels: torch.Tensor,
) -> float:
    """使用全距离矩阵计算轮廓系数（基于概率分布）"""
    n_samples = data.shape[0]

    # 使用标准余弦相似度计算所有样本对的相似度矩阵 [N, N]
    # 扩展维度以便批量计算：data [N, 1, D], data.t() [1, N, D] -> [N, N, D]
    data_expanded = data.unsqueeze(1)  # [N, 1, D]
    data_transposed = data.unsqueeze(0)  # [1, N, D]

    # 使用标准余弦相似度函数
    similarity_matrix = torch.nn.functional.cosine_similarity(data_expanded, data_transposed, dim=2)  # [N, N]
    distance_matrix = 1.0 - similarity_matrix  # 余弦距离

    silhouette_scores = torch.zeros(n_samples, device=data.device, dtype=data.dtype)

    for i in tqdm(range(n_samples), "全距离矩阵计算轮廓系数进度"):
        current_label = labels[i]

        # 同簇样本掩码 (排除自己)
        same_cluster_mask = (labels == current_label) & (torch.arange(n_samples, device=data.device) != i)

        if same_cluster_mask.sum() == 0:
            # 如果该样本是簇内唯一样本，轮廓系数为0
            silhouette_scores[i] = 0.0
            continue

        # a(i): 与同簇内其他样本的平均距离
        a_i = distance_matrix[i][same_cluster_mask].mean()

        # b(i): 与最近邻簇中样本的平均距离
        min_avg_dist = float("inf")

        for cluster_label in unique_labels:
            if cluster_label == current_label:
                continue

            other_cluster_mask = labels == cluster_label
            if other_cluster_mask.sum() > 0:
                avg_dist_to_cluster = distance_matrix[i][other_cluster_mask].mean()
                min_avg_dist = min(min_avg_dist, avg_dist_to_cluster)

        b_i = min_avg_dist

        # s(i) = (b(i) - a(i)) / max(a(i), b(i))
        if b_i == float("inf"):
            silhouette_scores[i] = 0.0
        else:
            denominator = max(a_i, b_i)
            if denominator > 0:
                silhouette_scores[i] = (b_i - a_i) / denominator
            else:
                silhouette_scores[i] = 0.0

    return silhouette_scores.mean().item()


def _gpu_silhouette_chunked(
    data: torch.Tensor,
    labels: torch.Tensor,
    unique_labels: torch.Tensor,
    chunk_size: int,
) -> float:
    """使用分块计算轮廓系数，节省GPU内存（基于概率分布）"""
    n_samples = data.shape[0]
    silhouette_scores = torch.zeros(n_samples, device=data.device, dtype=data.dtype)

    # 分块处理每个样本
    for start_idx in tqdm(range(0, n_samples, chunk_size), "分块计算轮廓系数进度"):
        end_idx = min(start_idx + chunk_size, n_samples)
        chunk_indices = torch.arange(start_idx, end_idx, device=data.device)

        # 计算当前块与所有样本的距离
        chunk_data = data[start_idx:end_idx]  # [chunk_size, D]
        chunk_labels = labels[start_idx:end_idx]  # [chunk_size]

        # 使用标准余弦相似度计算相似度矩阵 [chunk_size, N]
        chunk_expanded = chunk_data.unsqueeze(1)  # [chunk_size, 1, D]
        data_expanded = data.unsqueeze(0)  # [1, N, D]
        similarity_matrix = torch.nn.functional.cosine_similarity(chunk_expanded, data_expanded, dim=2)
        distance_matrix = 1.0 - similarity_matrix

        for local_i, global_i in enumerate(chunk_indices):
            current_label = chunk_labels[local_i]

            # 同簇样本掩码 (排除自己)
            same_cluster_mask = (labels == current_label) & (torch.arange(n_samples, device=data.device) != global_i)

            if same_cluster_mask.sum() == 0:
                silhouette_scores[global_i] = 0.0
                continue

            # a(i): 与同簇内其他样本的平均距离
            a_i = distance_matrix[local_i][same_cluster_mask].mean()

            # b(i): 与最近邻簇中样本的平均距离
            min_avg_dist = float("inf")

            for cluster_label in unique_labels:
                if cluster_label == current_label:
                    continue

                other_cluster_mask = labels == cluster_label
                if other_cluster_mask.sum() > 0:
                    avg_dist_to_cluster = distance_matrix[local_i][other_cluster_mask].mean()
                    min_avg_dist = min(min_avg_dist, avg_dist_to_cluster)

            b_i = min_avg_dist

            # s(i) = (b(i) - a(i)) / max(a(i), b(i))
            if b_i == float("inf"):
                silhouette_scores[global_i] = 0.0
            else:
                denominator = max(a_i, b_i)
                if denominator > 0:
                    silhouette_scores[global_i] = (b_i - a_i) / denominator
                else:
                    silhouette_scores[global_i] = 0.0

    return silhouette_scores.mean().item()


def gpu_silhouette_score_batch(
    data: torch.Tensor,
    labels_list: list[torch.Tensor],
    chunk_size: Optional[int] = None,
) -> list[float]:
    """
    批量计算多个标签配置的轮廓系数，适用于Elbow Method等需要测试多个k值的场景

    Args:
        data: [N, D] 概率分布特征矩阵
        labels_list: 多个标签配置的列表，每个元素为 [N] 的标签张量
        chunk_size: 分块大小

    Returns:
        对应每个标签配置的轮廓系数列表
    """
    return [gpu_silhouette_score_cosine(data, labels, chunk_size) for labels in labels_list]
