#!/usr/bin/env python3
"""
自定义质量门损失函数可视化脚本 (改进版)

本脚本旨在以更直观的方式可视化和比较不同的自定义损失函数，
用于 Select-MoE 模型中的质量门（Quality Gate）。

新的可视化包含两部分：
1. 一个核心的条形图，直接对比三种损失函数在不同分布下的值。
2. 一组详细的直方图，展示每种模拟分布的具体形态。
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Maple Mono NF CN"]


# --- 损失函数实现 ---


def beta_moment_matching_loss(good_ratio: torch.Tensor, target_mean: float = 0.5, target_var: float = 0.05, w_mean: float = 1.0, w_var: float = 1.0) -> float:
    """计算矩匹配损失的标量值"""
    if good_ratio.numel() == 0:
        return 0.0
    batch_mean = good_ratio.mean()
    batch_var = good_ratio.var()
    loss_mean = (batch_mean - target_mean) ** 2
    loss_var = (batch_var - target_var) ** 2
    return (w_mean * loss_mean + w_var * loss_var).item()


def mean_variance_regularization_loss(good_ratio: torch.Tensor, lambda_var: float = 0.1) -> float:
    """计算均值-方差正则化损失的标量值"""
    if good_ratio.numel() == 0:
        return 0.0
    batch_mean = good_ratio.mean()
    batch_var = good_ratio.var()
    loss_mean = (batch_mean - 0.5) ** 2
    loss_var = -lambda_var * batch_var
    return (loss_mean + loss_var).item()


def entropy_regularization_loss(good_ratio: torch.Tensor, lambda_entropy: float = 0.05, n_bins: int = 20) -> float:
    """使用熵来鼓励分布多样性的损失函数"""
    if good_ratio.numel() == 0:
        return 0.0
    # 1. 计算熵 (通过分箱)
    hist = torch.histc(good_ratio, bins=n_bins, min=0, max=1)
    probs = hist / hist.sum()
    entropy = -torch.sum(probs * torch.log(probs + 1e-8))

    # 2. 计算均值损失
    batch_mean = good_ratio.mean()
    loss_mean = (batch_mean - 0.5) ** 2

    # 3. 组合损失
    batch_loss = loss_mean - lambda_entropy * entropy
    return batch_loss.item()


# --- 模拟数据生成 ---


def generate_distributions(n_samples=1000):
    """生成一系列模拟的 good_ratio 分布"""
    distributions = {
        "理想分布\n(Beta(2,2))": torch.tensor(np.random.beta(2, 2, n_samples), dtype=torch.float32),
        "U形分布\n(高方差)": torch.tensor(np.random.beta(0.5, 0.5, n_samples), dtype=torch.float32),
        "两极分化": torch.cat([torch.normal(0.2, 0.05, size=(n_samples // 2,)), torch.normal(0.8, 0.05, size=(n_samples // 2,))]).clamp(0, 1),
        "均匀分布": torch.rand(n_samples),
        "正态分布\n(中心化)": torch.normal(0.5, 0.1, size=(n_samples,)).clamp(0, 1),
        "正态分布\n(偏向0)": torch.normal(0.2, 0.1, size=(n_samples,)).clamp(0, 1),
        "正态分布\n(偏向1)": torch.normal(0.8, 0.1, size=(n_samples,)).clamp(0, 1),
        "坍缩到0\n(方差=0)": torch.zeros(n_samples),
        "坍缩到1\n(方差=0)": torch.ones(n_samples),
        "坍缩到0.5\n(方差=0)": torch.full((n_samples,), 0.5),
    }
    return distributions


# --- 可视化 ---


def visualize_loss_functions_improved(distributions):
    """
    创建更直观的“总-分”结构可视化图表
    """
    # 1. 首先计算所有损失值和统计数据
    results = []
    for title, data in distributions.items():
        results.append(
            {
                "title": title,
                "data": data,
                "mean": data.mean().item(),
                "variance": data.var().item(),
                "矩匹配损失": beta_moment_matching_loss(data),
                "均值-方差损失": mean_variance_regularization_loss(data),
                "熵最大化损失": entropy_regularization_loss(data),
            }
        )

    df = pd.DataFrame(results)

    # 2. 创建图表布局
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 3], hspace=0.45)
    fig.suptitle("自定义质量门损失函数分析", fontsize=24, y=0.98)

    # --- Part 1: 核心对比条形图 ---
    ax_summary = fig.add_subplot(gs[0, 0])

    df_melted = df.melt(id_vars="title", value_vars=["矩匹配损失", "均值-方差损失", "熵最大化损失"], var_name="损失类型", value_name="损失值")

    sns.barplot(x="title", y="损失值", hue="损失类型", data=df_melted, ax=ax_summary)

    ax_summary.set_title("核心对比：三种损失函数在不同分布下的表现", fontsize=16)
    ax_summary.set_xlabel("模拟 good_ratio 分布类型", fontsize=12)
    ax_summary.set_ylabel("损失值", fontsize=12)
    ax_summary.tick_params(axis="x", rotation=45, labelsize=11)
    ax_summary.grid(axis="y", linestyle="--", alpha=0.7)
    ax_summary.axhline(0, color="black", linewidth=1.2)  # 突出0刻度线
    ax_summary.legend(title="损失函数类型")

    # --- Part 2: 详细分布直方图 ---
    gs_details = gs[1, 0].subgridspec(4, 3, hspace=0.6, wspace=0.2)
    axes_details = gs_details.subplots().flatten()

    sub_title = fig.add_subplot(gs[1, 0], frameon=False)
    sub_title.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    sub_title.set_title("分布细节：各种模拟分布的形状和统计数据", fontsize=16, pad=40)

    for i, res in enumerate(results):
        if i >= len(axes_details):
            break
        ax = axes_details[i]

        sns.histplot(res["data"].numpy(), bins=50, kde=True, ax=ax, color="skyblue")

        stats_text = f"均值={res['mean']:.2f}, 方差={res['variance']:.3f}"
        ax.set_title(f"{res['title'].replace('n', ' ')}\n({stats_text})", fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(True, linestyle="--", alpha=0.3)

    for j in range(i + 1, len(axes_details)):
        fig.delaxes(axes_details[j])

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = "custom_loss_visualization_improved.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"改进版的可视化图表已保存到: {save_path}")

    plt.show()


if __name__ == "__main__":
    print("生成模拟 good_ratio 分布...")
    simulated_distributions = generate_distributions()

    print("计算并可视化损失函数 (改进版)...")
    visualize_loss_functions_improved(simulated_distributions)
