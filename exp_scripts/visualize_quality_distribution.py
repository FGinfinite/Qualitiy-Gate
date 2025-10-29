#!/usr/bin/env python3
"""
质量分数分布可视化脚本

该脚本加载 router_data 文件，使用不同的质量计算策略，
并绘制质量分数的分布直方图，用于验证困惑度加权和列归一化的有效性。

使用方法:
    python exp_scripts/visualize_quality_distribution.py \
        --router-data-dir outputs/stage_2_selection/.../router_data \
        --output quality_distribution.png
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

# 设置中文字体支持
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False


def load_router_data(router_data_path: str) -> dict:
    """加载路由数据"""
    data = torch.load(router_data_path, map_location="cpu")
    return data


def compute_token_weights(perplexities: torch.Tensor, alpha: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    """计算样本内token权重（高熵更重要）"""
    ppl_powered = torch.pow(perplexities, alpha)
    weights = ppl_powered / (ppl_powered.sum() + eps)
    return weights


def compute_layer_scores(
    quality_gates: torch.Tensor,
    perplexities: torch.Tensor,
    alpha: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """按token加权得到逐层分数"""
    weights = compute_token_weights(perplexities, alpha, eps)
    layer_scores = (quality_gates * weights[None, :]).sum(dim=1)
    return layer_scores


def one_step_mapping(S: np.ndarray, eps: float = 1e-8, tau: float = 0.0) -> np.ndarray:
    """逐层归一化映射"""
    a = S.min(axis=0)
    b = S.max(axis=0)
    mu = S.mean(axis=0)

    if tau > 0:
        den = np.maximum(b - a, eps) * (mu + tau)
    else:
        den = np.maximum(b - a, eps) * np.maximum(mu, eps)

    R = (S - a[None, :]) / den[None, :]
    return R


def method_1_full(
    all_quality_gates: List[torch.Tensor], all_perplexities: List[torch.Tensor], alpha: float = 1.0, eps: float = 1e-8, tau: float = 0.0
) -> np.ndarray:
    """方法1: 完整方法（困惑度加权 + 列归一化）"""
    # 计算逐层分数
    S_rows = []
    for quality_gates, perplexities in zip(all_quality_gates, all_perplexities):
        layer_scores = compute_layer_scores(quality_gates, perplexities, alpha, eps)
        S_rows.append(layer_scores.cpu().numpy())

    S = np.stack(S_rows, axis=0)

    # 列归一化
    R = one_step_mapping(S, eps, tau)

    # 计算质量分数
    q = R.mean(axis=1)

    # Min-Max归一化到[0, 1]
    q_normalized = (q - q.min()) / (q.max() - q.min() + eps)

    return q_normalized


def method_2_weighted_only(all_quality_gates: List[torch.Tensor], all_perplexities: List[torch.Tensor], alpha: float = 1.0, eps: float = 1e-8) -> np.ndarray:
    """方法2: 只用困惑度加权，不用列归一化"""
    # 计算逐层分数
    S_rows = []
    for quality_gates, perplexities in zip(all_quality_gates, all_perplexities):
        layer_scores = compute_layer_scores(quality_gates, perplexities, alpha, eps)
        S_rows.append(layer_scores.cpu().numpy())

    S = np.stack(S_rows, axis=0)

    # 直接求均值（不做列归一化）
    q = S.mean(axis=1)

    # Min-Max归一化到[0, 1]
    q_normalized = (q - q.min()) / (q.max() - q.min() + eps)

    return q_normalized


def method_3_normalized_only(all_quality_gates: List[torch.Tensor], all_perplexities: List[torch.Tensor], eps: float = 1e-8, tau: float = 0.0) -> np.ndarray:
    """方法3: 只用列归一化，不用困惑度加权（直接平均）"""
    # 直接对每层求平均（不用困惑度加权）
    S_rows = []
    for quality_gates, perplexities in zip(all_quality_gates, all_perplexities):
        # 找到有效token（困惑度 > 0）
        valid_mask = perplexities > 0
        valid_length = valid_mask.sum().item()

        if valid_length > 0:
            # 对每层的有效token求平均
            layer_scores = quality_gates[:, :valid_length].mean(dim=1)
        else:
            layer_scores = quality_gates[:, :1].mean(dim=1)

        S_rows.append(layer_scores.cpu().numpy())

    S = np.stack(S_rows, axis=0)

    # 列归一化
    R = one_step_mapping(S, eps, tau)

    # 计算质量分数
    q = R.mean(axis=1)

    # Min-Max归一化到[0, 1]
    q_normalized = (q - q.min()) / (q.max() - q.min() + eps)

    return q_normalized


def method_4_raw_mean(all_quality_gates: List[torch.Tensor], all_perplexities: List[torch.Tensor], eps: float = 1e-8) -> np.ndarray:
    """方法4: 什么都不用，直接对quality_gates求均值"""
    scores = []
    for quality_gates, perplexities in zip(all_quality_gates, all_perplexities):
        # 找到有效token
        valid_mask = perplexities > 0
        valid_length = valid_mask.sum().item()

        if valid_length > 0:
            # 对所有层和所有有效token求均值
            score = quality_gates[:, :valid_length].mean().item()
        else:
            score = quality_gates[:, :1].mean().item()

        scores.append(score)

    q = np.array(scores)

    # Min-Max归一化到[0, 1]
    q_normalized = (q - q.min()) / (q.max() - q.min() + eps)

    return q_normalized


def prepare_data_from_router_data(router_data_dir: str) -> Tuple[Dict[str, np.ndarray], Dict[str, List]]:
    """从router_data目录加载所有数据集的数据"""
    router_data_dir = Path(router_data_dir)

    all_quality_gates = []
    all_perplexities = []
    all_dataset_names = []

    # 查找所有router_data文件
    for file_path in sorted(router_data_dir.glob("*_router_data.pt")):
        dataset_name = file_path.stem.replace("_router_data", "")

        print(f"加载数据集: {dataset_name} ({file_path.name})")
        router_data = load_router_data(str(file_path))

        quality_gates = router_data["quality_gates"]  # [N, L, max_seq_len]
        perplexities = router_data["perplexities"]  # [N, max_seq_len]
        num_samples = router_data["num_samples"]

        print(f"  样本数: {num_samples}")
        print(f"  质量门控形状: {quality_gates.shape}")
        print(f"  困惑度形状: {perplexities.shape}")

        # 提取每个样本的有效数据
        for i in range(num_samples):
            qg = quality_gates[i]  # [L, max_seq_len]
            ppl = perplexities[i]  # [max_seq_len]

            # 找到有效长度
            valid_mask = ppl > 0
            valid_length = valid_mask.sum().item()

            if valid_length > 0:
                qg_valid = qg[:, :valid_length]
                ppl_valid = ppl[:valid_length]
            else:
                qg_valid = qg[:, :1]
                ppl_valid = ppl[:1]

            all_quality_gates.append(qg_valid)
            all_perplexities.append(ppl_valid)
            all_dataset_names.append(dataset_name)

    print(f"\n总样本数: {len(all_quality_gates)}")

    # 计算4种方法的质量分数
    print("\n计算质量分数...")
    print("  方法1: 完整方法（困惑度加权 + 列归一化）")
    scores_method1 = method_1_full(all_quality_gates, all_perplexities)

    print("  方法2: 只用困惑度加权")
    scores_method2 = method_2_weighted_only(all_quality_gates, all_perplexities)

    print("  方法3: 只用列归一化")
    scores_method3 = method_3_normalized_only(all_quality_gates, all_perplexities)

    print("  方法4: 原始均值")
    scores_method4 = method_4_raw_mean(all_quality_gates, all_perplexities)

    scores_dict = {
        "method1": scores_method1,
        "method2": scores_method2,
        "method3": scores_method3,
        "method4": scores_method4,
    }

    data_dict = {
        "dataset_names": all_dataset_names,
    }

    return scores_dict, data_dict


def plot_distributions(scores_dict: Dict[str, np.ndarray], dataset_names: List[str], output_path: str, bin_width: float = 0.01):
    """绘制质量分数分布图

    Args:
        scores_dict: 各方法的质量分数字典
        dataset_names: 数据集名称列表
        output_path: 输出图片路径
        bin_width: 直方图bin宽度（默认0.01即1%）
    """
    # 设置图表样式
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()

    # 获取所有唯一的数据集名称
    unique_datasets = sorted(list(set(dataset_names)))
    print(f"\n数据集列表: {unique_datasets}")

    # 为每个数据集分配颜色（使用更鲜明的颜色）
    color_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    dataset_colors = {dataset: color_palette[i % len(color_palette)] for i, dataset in enumerate(unique_datasets)}

    method_names = [
        "Method 1: Full Method (Perplexity Weighted + Column Normalization)",
        "Method 2: Perplexity Weighted Only (No Column Normalization)",
        "Method 3: Column Normalization Only (No Perplexity Weighting)",
        "Method 4: Raw Mean (No Special Processing)",
    ]

    method_keys = ["method1", "method2", "method3", "method4"]

    for idx, (ax, method_name, method_key) in enumerate(zip(axes, method_names, method_keys)):
        scores = scores_dict[method_key]

        # 🔧 为当前方法自适应计算bins和x轴范围
        score_min = scores.min()
        score_max = scores.max()
        score_range = score_max - score_min

        # 留一些边距（5%）
        padding = score_range * 0.05
        x_min = max(0, score_min - padding)
        x_max = min(1, score_max + padding)

        # 动态计算bins：使用用户指定的bin宽度
        num_bins = int(np.ceil((x_max - x_min) / bin_width))
        bins = np.linspace(x_min, x_max, num_bins + 1)

        print(f"\n{method_name}:")
        print(f"  数据范围: [{score_min:.4f}, {score_max:.4f}]")
        print(f"  X轴范围: [{x_min:.4f}, {x_max:.4f}]")
        print(f"  Bin宽度: {bin_width:.4f} ({bin_width * 100:.2f}%)")
        print(f"  Bins数量: {num_bins}")

        # 为每个数据集绘制柱状图
        for dataset in unique_datasets:
            # 筛选当前数据集的分数
            mask = np.array([name == dataset for name in dataset_names])
            dataset_scores = scores[mask]

            # 绘制直方图
            ax.hist(
                dataset_scores,
                bins=bins,
                alpha=0.65,
                color=dataset_colors[dataset],
                label=f"{dataset} (n={len(dataset_scores)})",
                edgecolor="black",
                linewidth=0.8,
            )

        # 设置标题和标签
        ax.set_title(method_name, fontsize=13, fontweight="bold", pad=15)
        ax.set_xlabel("Quality Score", fontsize=11, fontweight="bold")
        ax.set_ylabel("Number of Samples", fontsize=11, fontweight="bold")

        # 🔧 设置自适应的x轴范围
        ax.set_xlim(x_min, x_max)

        # 🔧 设置自适应的x轴刻度（每隔1%显示一个标签，避免过于密集）
        tick_step = 0.01  # 每隔1%显示一个刻度
        tick_start = np.ceil(x_min / tick_step) * tick_step
        tick_end = np.floor(x_max / tick_step) * tick_step
        tick_positions = np.arange(tick_start, tick_end + tick_step / 2, tick_step)

        # 如果刻度太少（<5个），就增加密度
        if len(tick_positions) < 5:
            tick_step = 0.005  # 改为每隔0.5%
            tick_start = np.ceil(x_min / tick_step) * tick_step
            tick_end = np.floor(x_max / tick_step) * tick_step
            tick_positions = np.arange(tick_start, tick_end + tick_step / 2, tick_step)

        ax.set_xticks(tick_positions)
        ax.set_xticklabels([f"{x:.3f}" for x in tick_positions], rotation=45, ha="right", fontsize=9)

        # 设置y轴刻度
        ax.tick_params(axis="y", labelsize=9)

        # 添加网格
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

        # 添加图例
        ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

        # 添加垂直线标记中位数
        for dataset in unique_datasets:
            mask = np.array([name == dataset for name in dataset_names])
            dataset_scores = scores[mask]
            median = np.median(dataset_scores)
            ax.axvline(median, color=dataset_colors[dataset], linestyle="--", linewidth=1.5, alpha=0.8)

        # 打印统计信息
        print(f"\n{method_name}:")
        for dataset in unique_datasets:
            mask = np.array([name == dataset for name in dataset_names])
            dataset_scores = scores[mask]
            print(
                f"  {dataset}: "
                f"均值={dataset_scores.mean():.4f}, "
                f"标准差={dataset_scores.std():.4f}, "
                f"中位数={np.median(dataset_scores):.4f}, "
                f"[25%, 75%]=({np.percentile(dataset_scores, 25):.4f}, {np.percentile(dataset_scores, 75):.4f})"
            )

    # 添加总标题
    fig.suptitle("Quality Score Distribution Comparison: Effects of Different Calculation Methods", fontsize=16, fontweight="bold", y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n✅ 图表已保存到: {output_path}")
    plt.close()


def print_separation_analysis(scores_dict: Dict[str, np.ndarray], dataset_names: List[str]):
    """打印数据集分离度分析"""
    print("\n" + "=" * 80)
    print("数据集分离度分析")
    print("=" * 80)

    unique_datasets = sorted(list(set(dataset_names)))
    method_keys = ["method1", "method2", "method3", "method4"]
    method_names = ["方法1（完整）", "方法2（加权）", "方法3（归一化）", "方法4（原始）"]

    for method_key, method_name in zip(method_keys, method_names):
        scores = scores_dict[method_key]

        print(f"\n{method_name}:")

        if len(unique_datasets) == 2:
            # 如果只有两个数据集，计算它们之间的分离度
            dataset1, dataset2 = unique_datasets
            mask1 = np.array([name == dataset1 for name in dataset_names])
            mask2 = np.array([name == dataset2 for name in dataset_names])

            scores1 = scores[mask1]
            scores2 = scores[mask2]

            mean1 = scores1.mean()
            mean2 = scores2.mean()
            std1 = scores1.std()
            std2 = scores2.std()

            # Cohen's d (效应量)
            pooled_std = np.sqrt((std1**2 + std2**2) / 2)
            cohens_d = abs(mean1 - mean2) / pooled_std if pooled_std > 0 else 0

            print(f"  {dataset1} vs {dataset2}:")
            print(f"    均值差异: {abs(mean1 - mean2):.4f}")
            print(f"    Cohen's d: {cohens_d:.4f} ", end="")

            if cohens_d < 0.2:
                print("(可忽略)")
            elif cohens_d < 0.5:
                print("(小)")
            elif cohens_d < 0.8:
                print("(中)")
            else:
                print("(大)")

            # 重叠度（通过分位数估计）
            q25_1 = np.percentile(scores1, 25)
            q75_1 = np.percentile(scores1, 75)
            q25_2 = np.percentile(scores2, 25)
            q75_2 = np.percentile(scores2, 75)

            if mean1 > mean2:
                overlap = max(0, min(q75_2, q75_1) - max(q25_1, q25_2))
            else:
                overlap = max(0, min(q75_1, q75_2) - max(q25_2, q25_1))

            print(f"    分位数重叠区间: {overlap:.4f}")


def main():
    parser = argparse.ArgumentParser(description="可视化质量分数分布")
    parser.add_argument("--router-data-dir", required=True, help="router_data目录路径")
    parser.add_argument("--output", default="quality_distribution.png", help="输出图片路径 (默认: quality_distribution.png)")
    parser.add_argument("--bin-width", type=float, default=0.01, help="直方图bin宽度 (默认: 0.01即1%%)")
    parser.add_argument("--analyze", action="store_true", help="打印详细的分离度分析")

    args = parser.parse_args()

    print("=" * 80)
    print("质量分数分布可视化")
    print("=" * 80)
    print(f"Router Data 目录: {args.router_data_dir}")
    print(f"输出路径: {args.output}")
    print(f"Bin 宽度: {args.bin_width} ({args.bin_width * 100:.2f}%)")

    # 加载数据并计算质量分数
    scores_dict, data_dict = prepare_data_from_router_data(args.router_data_dir)

    # 绘制分布图
    plot_distributions(scores_dict, data_dict["dataset_names"], args.output, bin_width=args.bin_width)

    # 打印分离度分析
    if args.analyze:
        print_separation_analysis(scores_dict, data_dict["dataset_names"])

    print("\n" + "=" * 80)
    print("完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
