# src/selection/quality_scoring.py
"""
基于质量门控和困惑度的数据筛选逻辑
实现 quality_gate.md 中描述的方法
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch


def compute_token_weights(perplexities: torch.Tensor, alpha: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    """
    计算样本内token权重（高熵更重要）

    Args:
        perplexities: token困惑度，形状为 [T] （单个样本）
        alpha: 强调高熵token的权重（默认1；>1 更偏重高熵；<1 更均衡）
        eps: 数值下界，防分母为0

    Returns:
        token权重，形状为 [T]，满足 sum(w) = 1
    """
    # w_{i,t} = ppl_{i,t}^α / (Σ_u ppl_{i,u}^α + ε)
    ppl_powered = torch.pow(perplexities, alpha)
    weights = ppl_powered / (ppl_powered.sum() + eps)
    return weights


def compute_layer_scores(
    quality_gates: torch.Tensor,  # [L, T]
    perplexities: torch.Tensor,  # [T]
    alpha: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    按token加权得到逐层分数

    Args:
        quality_gates: 质量门控logits (sigmoid前)，形状为 [L, T]
        perplexities: token困惑度，形状为 [T]
        alpha: token权重参数
        eps: 数值下界

    Returns:
        逐层分数，形状为 [L]
    """
    # 计算token权重
    weights = compute_token_weights(perplexities, alpha, eps)  # [T]

    # 对质量门控应用sigmoid
    g = torch.sigmoid(quality_gates)  # [L, T]

    # 按token加权求和
    # s_{i,l} = Σ_t w_{i,t} * g_{i,l,t}
    layer_scores = (g * weights[None, :]).sum(dim=1)  # [L]

    return layer_scores


def one_step_mapping(
    S: np.ndarray,  # [N, L] 所有样本的逐层分数
    eps: float = 1e-8,
    tau: float = 0.0,  # 可选的稳定性参数
) -> np.ndarray:
    """
    逐层"一步映射"到统一尺度并加权

    把不同层的列拉到统一尺度，同时让均值小（更"会关门"）的层权重大

    Args:
        S: 逐层分数矩阵，形状为 [N, L]
        eps: 数值下界
        tau: 稳定性参数（可选，用于 μ_l + tau 替代 max(μ_l, ε)）

    Returns:
        归一化后的分数矩阵 R，形状为 [N, L]
    """
    # 计算每列统计量（跨样本）
    a = S.min(axis=0)  # [L]
    b = S.max(axis=0)  # [L]
    mu = S.mean(axis=0)  # [L]

    # 一步映射
    # R_{i,l} = (S_{i,l} - a_l) / (max(b_l - a_l, ε) * max(μ_l, ε))
    if tau > 0:
        # 使用 tau 作为稳定性参数
        den = np.maximum(b - a, eps) * (mu + tau)
    else:
        # 使用默认的 max(μ_l, ε)
        den = np.maximum(b - a, eps) * np.maximum(mu, eps)

    R = (S - a[None, :]) / den[None, :]  # [N, L]

    return R


def compute_quality_scores(
    all_quality_gates: List[torch.Tensor],  # List of [L, T_i]
    all_perplexities: List[torch.Tensor],  # List of [T_i]
    alpha: float = 1.0,
    eps: float = 1e-8,
    tau: float = 0.0,
) -> np.ndarray:
    """
    计算所有样本的质量分数

    Args:
        all_quality_gates: 所有样本的质量门控logits列表
        all_perplexities: 所有样本的困惑度列表
        alpha: token权重参数
        eps: 数值下界
        tau: 稳定性参数

    Returns:
        质量分数数组，形状为 [N]
    """
    log = logging.getLogger(__name__)

    N = len(all_quality_gates)
    if N == 0:
        return np.array([])

    # 阶段3: 统计收集
    log.info("=== 阶段3: 计算逐层分数 ===")
    S_rows = []
    for i, (quality_gates, perplexities) in enumerate(zip(all_quality_gates, all_perplexities)):
        layer_scores = compute_layer_scores(quality_gates, perplexities, alpha, eps)
        S_rows.append(layer_scores.cpu().numpy())

        if i < 3:
            log.debug(f"样本{i} 逐层分数: {layer_scores.cpu().numpy()}")

    S = np.stack(S_rows, axis=0)  # [N, L]
    log.info(f"逐层分数矩阵 S 形状: {S.shape}")
    log.info(f"S 统计: min={S.min():.6f}, max={S.max():.6f}, mean={S.mean():.6f}")

    # 阶段4: 一步映射到统一尺度
    log.info("=== 阶段4: 一步映射到统一尺度 ===")
    R = one_step_mapping(S, eps, tau)
    log.info(f"归一化后矩阵 R 形状: {R.shape}")
    log.info(f"R 统计: min={R.min():.6f}, max={R.max():.6f}, mean={R.mean():.6f}")

    # 阶段5: 样本质量分数
    log.info("=== 阶段5: 计算样本质量分数 ===")
    q = R.mean(axis=1)  # [N]
    log.info(f"质量分数 q 形状: {q.shape}")
    log.info(f"q 统计: min={q.min():.6f}, max={q.max():.6f}, mean={q.mean():.6f}")

    # 输出前10和后10的质量分数
    sorted_indices = np.argsort(-q)
    log.info(f"前10个质量分数: {q[sorted_indices[:10]]}")
    log.info(f"后10个质量分数: {q[sorted_indices[-10:]]}")

    return q


def select_top_k_percent(
    quality_scores: np.ndarray,
    sample_ids: List[str],
    messages_list: List[List[Dict]],
    dataset_names: List[str],
    top_ratio: float = 0.30,
) -> Tuple[List[Dict], np.ndarray]:
    """
    按质量分数降序选择 top-k%

    Args:
        quality_scores: 质量分数数组，形状为 [N]
        sample_ids: 样本ID列表
        messages_list: messages列表
        dataset_names: 数据集名称列表
        top_ratio: 选择比例（例如：0.30 表示选择前30%）

    Returns:
        选择的数据列表，每个元素包含 {dataset, id, scores, messages}
        选择的索引数组
    """
    log = logging.getLogger(__name__)

    N = len(quality_scores)
    k = int(np.ceil(top_ratio * N))

    log.info(f"=== 选择 top-{top_ratio * 100:.2f}% ===")
    log.info(f"总样本数: {N}")
    log.info(f"选择数量: {k}")

    # 降序排序
    sorted_indices = np.argsort(-quality_scores)
    selected_indices = sorted_indices[:k]

    # 构建选择的数据
    selected_data = []
    for idx in selected_indices:
        selected_data.append(
            {
                "dataset": dataset_names[idx],
                "id": sample_ids[idx],
                "scores": float(quality_scores[idx]),
                "messages": messages_list[idx],
            }
        )

    log.info(f"选择完成，共 {len(selected_data)} 个样本")
    log.info(f"选择的质量分数范围: [{quality_scores[selected_indices].min():.6f}, {quality_scores[selected_indices].max():.6f}]")

    return selected_data, selected_indices


def quality_based_selection(
    all_quality_gates: List[torch.Tensor],
    all_perplexities: List[torch.Tensor],
    sample_ids: List[str],
    messages_list: List[List[Dict]],
    dataset_names: List[str],
    selection_percentage: float = 0.30,
    alpha: float = 1.0,
    eps: float = 1e-8,
    tau: float = 0.0,
) -> List[Dict]:
    """
    基于质量分数的数据选择（完整流程）

    Args:
        all_quality_gates: 所有样本的质量门控logits列表
        all_perplexities: 所有样本的困惑度列表
        sample_ids: 样本ID列表
        messages_list: messages列表
        dataset_names: 数据集名称列表
        selection_percentage: 选择比例
        alpha: token权重参数
        eps: 数值下界
        tau: 稳定性参数

    Returns:
        选择的数据列表
    """
    log = logging.getLogger(__name__)
    log.info("=== 开始基于质量分数的数据选择 ===")
    log.info(f"参数: alpha={alpha}, eps={eps}, tau={tau}")
    log.info(f"选择比例: {selection_percentage * 100:.2f}%")

    # 计算质量分数
    quality_scores = compute_quality_scores(
        all_quality_gates,
        all_perplexities,
        alpha=alpha,
        eps=eps,
        tau=tau,
    )

    # 选择 top-k%
    selected_data, _ = select_top_k_percent(
        quality_scores,
        sample_ids,
        messages_list,
        dataset_names,
        top_ratio=selection_percentage,
    )

    log.info("=== 质量选择完成 ===")

    return selected_data
