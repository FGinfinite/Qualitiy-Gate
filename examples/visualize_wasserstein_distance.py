#!/usr/bin/env python3
"""
Wasserstein距离计算可视化脚本

本脚本演示了如何计算和可视化两个样本之间的Wasserstein距离。
Wasserstein距离也被称为Earth Mover's Distance（EMD），用于衡量两个概率分布的差异。

在Select-MoE项目中，我们使用Wasserstein距离来衡量不同样本的MoE路由分布之间的相似性。
距离越大，样本的路由模式差异越大，从而在多样性选择中被认为是更加多样化的样本。
支持读取整个router_data文件夹内的所有.pt文件，并对每个数据集进行Wasserstein距离分析。
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import glob

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy.stats import wasserstein_distance

from src.stages.selection import (
    compute_batch_wasserstein_distance_gpu,
    load_router_data,
)

plt.rcParams["font.sans-serif"] = ["Maple Mono NF CN"]


def load_all_router_data_files(router_data_path):
    """加载router_data文件或目录中的所有router_data文件"""
    if os.path.isfile(router_data_path) and router_data_path.endswith('.pt'):
        # 单个文件
        return {os.path.basename(router_data_path).replace('_router_data.pt', ''): load_router_data(router_data_path)}
    elif os.path.isdir(router_data_path):
        # 目录，查找所有_router_data.pt文件
        router_data_files = glob.glob(os.path.join(router_data_path, '*_router_data.pt'))
        if not router_data_files:
            raise ValueError(f"在目录 {router_data_path} 中未找到任何_router_data.pt文件")
        
        all_router_data = {}
        for file_path in sorted(router_data_files):
            dataset_name = os.path.basename(file_path).replace('_router_data.pt', '')
            print(f"加载数据集: {dataset_name} - {file_path}")
            all_router_data[dataset_name] = load_router_data(file_path)
        
        return all_router_data
    else:
        raise ValueError(f"路径不是有效的.pt文件或目录: {router_data_path}")


def visualize_probability_distributions(probs1, probs2, layer_idx=0, title_suffix=""):
    """可视化两个样本在某一层的概率分布"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # 提取某一层的概率分布
    prob_layer1 = probs1[layer_idx].numpy()  # [num_experts]
    prob_layer2 = probs2[layer_idx].numpy()  # [num_experts]

    expert_indices = np.arange(len(prob_layer1))

    # 子图1：条形图对比
    width = 0.35
    ax1.bar(
        expert_indices - width / 2,
        prob_layer1,
        width,
        label="样本 1",
        alpha=0.7,
        color="skyblue",
    )
    ax1.bar(
        expert_indices + width / 2,
        prob_layer2,
        width,
        label="样本 2",
        alpha=0.7,
        color="lightcoral",
    )
    ax1.set_xlabel("专家索引")
    ax1.set_ylabel("概率")
    ax1.set_title(f"第{layer_idx}层专家概率分布对比{title_suffix}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 子图2：累积分布函数（CDF）
    cdf1 = np.cumsum(prob_layer1)
    cdf2 = np.cumsum(prob_layer2)

    ax2.plot(expert_indices, cdf1, "o-", label="样本 1 CDF", color="blue", linewidth=2)
    ax2.plot(expert_indices, cdf2, "s-", label="样本 2 CDF", color="red", linewidth=2)
    ax2.fill_between(expert_indices, cdf1, cdf2, alpha=0.3, color="gray", label="CDF差异区域")
    ax2.set_xlabel("专家索引")
    ax2.set_ylabel("累积概率")
    ax2.set_title(f"累积分布函数（CDF）{title_suffix}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 子图3：Wasserstein距离可视化
    # 计算每个点的CDF差异
    cdf_diff = np.abs(cdf1 - cdf2)
    bars = ax3.bar(expert_indices, cdf_diff, alpha=0.7, color="green")

    # 计算该层的Wasserstein距离（CDF差异的总和）
    layer_wasserstein = np.sum(cdf_diff)
    ax3.set_xlabel("专家索引")
    ax3.set_ylabel("|CDF1 - CDF2|")
    ax3.set_title(f"第{layer_idx}层Wasserstein距离 = {layer_wasserstein:.4f}{title_suffix}")
    ax3.grid(True, alpha=0.3)

    # 在柱状图上标注贡献值
    for i, (bar, diff) in enumerate(zip(bars, cdf_diff)):
        if diff > 0.01:  # 只标注较大的值
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{diff:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    return fig, layer_wasserstein


def calculate_total_wasserstein_step_by_step(probs1, probs2, show_first_n_layers=3):
    """逐步计算总Wasserstein距离，展示每层的贡献"""
    num_layers = probs1.shape[0]
    layer_distances = []

    print(f"计算两个样本之间的Wasserstein距离（共{num_layers}层）:")
    print("=" * 70)

    for layer_idx in range(num_layers):
        # 提取该层的概率分布
        prob1 = probs1[layer_idx].numpy()
        prob2 = probs2[layer_idx].numpy()

        # 使用scipy计算该层的Wasserstein距离
        expert_indices = np.arange(len(prob1))
        layer_distance = wasserstein_distance(expert_indices, expert_indices, prob1, prob2)
        layer_distances.append(layer_distance)

        if layer_idx < show_first_n_layers:
            print(f"第{layer_idx:2d}层: {layer_distance:.6f}")

            # 展示前几层的详细计算过程
            cdf1 = np.cumsum(prob1)
            cdf2 = np.cumsum(prob2)
            cdf_diff = np.abs(cdf1 - cdf2)
            manual_distance = np.sum(cdf_diff)

            print(f"        手动计算: sum(|CDF1-CDF2|) = {manual_distance:.6f}")
            print(f"        概率分布1前5个值: {prob1[:5]}")
            print(f"        概率分布2前5个值: {prob2[:5]}")
            print(f"        CDF差异前5个值: {cdf_diff[:5]}")
            print()

    if num_layers > show_first_n_layers:
        print(f"...")
        print(f"第{num_layers - 1:2d}层: {layer_distances[-1]:.6f}")

    total_distance = sum(layer_distances)
    print("=" * 70)
    print(f"总Wasserstein距离: {total_distance:.6f} (所有层距离之和)")
    print(f"平均层距离: {total_distance / num_layers:.6f}")
    print()

    return total_distance, layer_distances


def demonstrate_gpu_computation(probs1, probs2):
    """演示GPU加速计算与CPU计算的对比"""
    print("GPU vs CPU计算对比:")
    print("=" * 50)

    # 准备数据
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_probs1 = probs1.unsqueeze(0).to(device)  # [1, L, E]
    batch_probs2 = probs2.unsqueeze(0).to(device)  # [1, L, E]

    if device.type == "cuda":
        # GPU计算
        gpu_distance_matrix = compute_batch_wasserstein_distance_gpu(batch_probs1, batch_probs2)
        gpu_result = gpu_distance_matrix[0, 0].item()
        print(f"GPU计算结果: {gpu_result:.6f}")
    else:
        print("CUDA不可用，无法演示GPU计算")
        gpu_result = None

    # CPU计算（逐层求和）
    layer_distances = []
    for layer_idx in range(probs1.shape[0]):
        prob1 = probs1[layer_idx].numpy()
        prob2 = probs2[layer_idx].numpy()
        cdf1 = np.cumsum(prob1)
        cdf2 = np.cumsum(prob2)
        layer_dist = np.sum(np.abs(cdf1 - cdf2))
        layer_distances.append(layer_dist)

    cpu_result = sum(layer_distances)
    print(f"CPU计算结果: {cpu_result:.6f}")

    if gpu_result is not None:
        print(f"差异: {abs(gpu_result - cpu_result):.8f}")
        if abs(gpu_result - cpu_result) < 1e-5:
            print("✓ GPU和CPU计算结果一致")
        else:
            print("⚠ GPU和CPU计算结果有差异")

    print()


def create_distance_heatmap(router_data, max_samples=20):
    """创建样本间距离的热力图"""
    print("=" * 70)
    print("创建样本间Wasserstein距离热力图")
    print("=" * 70)

    moe_logits = router_data["moe_logits"]  # [N, L, E]
    sample_ids = router_data["sample_ids"]

    # 限制样本数量以提高可视化效果
    n_samples = min(len(sample_ids), max_samples)
    selected_indices = np.linspace(0, len(sample_ids) - 1, n_samples, dtype=int)

    moe_subset = moe_logits[selected_indices]  # [n_samples, L, E]
    ids_subset = [sample_ids[i] for i in selected_indices]

    print(f"计算{n_samples}个样本之间的距离矩阵...")

    # 计算距离矩阵
    distance_matrix = np.zeros((n_samples, n_samples))

    # 尝试使用GPU加速计算
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device.type == "cuda":
        print("使用GPU加速计算...")
        # 转换为概率并移至GPU
        moe_probs = torch.softmax(moe_subset.float(), dim=-1).to(device)
        
        # 批量计算距离矩阵
        gpu_distance_matrix = compute_batch_wasserstein_distance_gpu(moe_probs, moe_probs)
        distance_matrix = gpu_distance_matrix.cpu().numpy()
        
        print(f"✓ GPU计算完成")
    else:
        print("使用CPU计算...")
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                # 转换为概率分布
                probs_i = torch.softmax(moe_subset[i].float(), dim=-1)
                probs_j = torch.softmax(moe_subset[j].float(), dim=-1)

                # 计算总距离
                total_dist = 0
                for layer_idx in range(probs_i.shape[0]):
                    prob1 = probs_i[layer_idx].numpy()
                    prob2 = probs_j[layer_idx].numpy()
                    expert_indices = np.arange(len(prob1))
                    layer_dist = wasserstein_distance(expert_indices, expert_indices, prob1, prob2)
                    total_dist += layer_dist

                distance_matrix[i, j] = total_dist
                distance_matrix[j, i] = total_dist

    # 创建热力图
    plt.figure(figsize=(12, 10))

    # 简化样本ID显示
    short_ids = [sample_id.split("_")[1] if "_" in sample_id else sample_id for sample_id in ids_subset]

    mask = np.triu(np.ones_like(distance_matrix, dtype=bool), k=1)
    sns.heatmap(
        distance_matrix,
        mask=mask,
        xticklabels=short_ids,
        yticklabels=short_ids,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        square=True,
    )

    plt.title(
        f"样本间Wasserstein距离热力图 - {router_data['dataset_name']}\n({n_samples}个样本)",
        fontsize=14,
        pad=20,
    )
    plt.xlabel("样本ID", fontsize=12)
    plt.ylabel("样本ID", fontsize=12)

    # 添加说明
    plt.figtext(
        0.02,
        0.02,
        "说明：颜色越深表示距离越大，样本间差异越大\n对角线为0（样本与自身距离）\n矩阵对称（距离具有对称性）",
        fontsize=10,
        ha="left",
    )

    plt.tight_layout()
    return plt.gcf(), distance_matrix


def analyze_single_dataset_wasserstein(dataset_name, router_data, args):
    """分析单个数据集的Wasserstein距离"""
    print(f"\n{'=' * 80}")
    print(f"Wasserstein距离分析 - 数据集: {dataset_name}")
    print(f"{'=' * 80}")
    
    moe_logits = router_data["moe_logits"]  # [N, L, E]
    sample_ids = router_data["sample_ids"]

    print(f"数据集: {dataset_name}")
    print(f"样本数: {len(sample_ids)}")
    print(f"模型层数: {moe_logits.shape[1]}")
    print(f"专家数: {moe_logits.shape[2]}")
    print()

    # 2. 选择两个样本进行对比
    sample1_idx = min(args.sample1_idx, len(sample_ids) - 1)
    sample2_idx = min(args.sample2_idx, len(sample_ids) - 1)

    if sample1_idx == sample2_idx:
        sample2_idx = (sample1_idx + 1) % len(sample_ids)

    print(f"对比样本:")
    print(f"  样本1: 索引{sample1_idx}, ID={sample_ids[sample1_idx]}")
    print(f"  样本2: 索引{sample2_idx}, ID={sample_ids[sample2_idx]}")
    print()

    # 提取样本的MoE logits并转换为概率
    logits1 = moe_logits[sample1_idx]  # [L, E]
    logits2 = moe_logits[sample2_idx]  # [L, E]

    probs1 = torch.softmax(logits1.float(), dim=-1)  # [L, E]
    probs2 = torch.softmax(logits2.float(), dim=-1)  # [L, E]

    # 3. 逐步计算总距离
    total_distance, layer_distances = calculate_total_wasserstein_step_by_step(probs1, probs2)

    # 4. 可视化特定层的分布
    layer_idx = min(args.layer_idx, probs1.shape[0] - 1)
    title_suffix = f" ({dataset_name} - 样本{sample1_idx} vs 样本{sample2_idx})"

    fig1, layer_wasserstein = visualize_probability_distributions(probs1, probs2, layer_idx, title_suffix)

    if args.save_plots:
        safe_dataset_name = dataset_name.replace('/', '_').replace('\\', '_')
        save_path = os.path.join(args.output_dir, f"layer_{layer_idx}_comparison_{safe_dataset_name}.png")
        fig1.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ 已保存层分析图: {save_path}")

    plt.show()

    # 5. 演示GPU计算
    demonstrate_gpu_computation(probs1, probs2)

    # 6. 创建距离热力图
    fig2, dist_matrix = create_distance_heatmap(router_data, max_samples=15)

    if args.save_plots:
        safe_dataset_name = dataset_name.replace('/', '_').replace('\\', '_')
        save_path = os.path.join(args.output_dir, f"distance_heatmap_{safe_dataset_name}.png")
        fig2.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ 已保存距离热力图: {save_path}")

    plt.show()

    # 7. 层级距离分析
    plt.figure(figsize=(12, 6))
    layers = range(len(layer_distances))
    plt.plot(layers, layer_distances, "o-", linewidth=2, markersize=6)
    plt.xlabel("层索引")
    plt.ylabel("Wasserstein距离")
    plt.title(f"{dataset_name} - 各层Wasserstein距离分布\n总距离: {total_distance:.4f}")
    plt.grid(True, alpha=0.3)

    # 标注最大和最小的几个点
    layer_distances_array = np.array(layer_distances)
    max_idx = np.argmax(layer_distances_array)
    min_idx = np.argmin(layer_distances_array)

    plt.annotate(
        f"最大: 第{max_idx}层\n{layer_distances[max_idx]:.4f}",
        xy=(max_idx, layer_distances[max_idx]),
        xytext=(max_idx + 1, layer_distances[max_idx] + 0.1),
        arrowprops=dict(arrowstyle="->", color="red"),
        fontsize=10,
        color="red",
    )

    plt.annotate(
        f"最小: 第{min_idx}层\n{layer_distances[min_idx]:.4f}",
        xy=(min_idx, layer_distances[min_idx]),
        xytext=(min_idx + 1, layer_distances[min_idx] - 0.1),
        arrowprops=dict(arrowstyle="->", color="blue"),
        fontsize=10,
        color="blue",
    )

    plt.tight_layout()

    if args.save_plots:
        safe_dataset_name = dataset_name.replace('/', '_').replace('\\', '_')
        save_path = os.path.join(args.output_dir, f"layer_distances_{safe_dataset_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ 已保存层级距离图: {save_path}")

    plt.show()
    
    return {
        'dataset_name': dataset_name,
        'total_distance': total_distance,
        'layer_distances': layer_distances,
        'average_distance': dist_matrix[dist_matrix > 0].mean() if dist_matrix[dist_matrix > 0].size > 0 else 0
    }


def main():
    parser = argparse.ArgumentParser(description="Wasserstein距离计算和可视化")
    parser.add_argument("router_data_path", help="路由数据文件路径(.pt格式)或包含多个router_data文件的目录")
    parser.add_argument("--sample1-idx", type=int, default=0, help="第一个样本的索引 (默认: 0)")
    parser.add_argument("--sample2-idx", type=int, default=1, help="第二个样本的索引 (默认: 1)")
    parser.add_argument("--layer-idx", type=int, default=0, help="要详细分析的层索引 (默认: 0)")
    parser.add_argument("--save-plots", action="store_true", help="保存图片到文件")
    parser.add_argument("--output-dir", default="./outputs/visual_figs/wasserstein_plots", help="图片保存目录")
    parser.add_argument("--dataset-filter", help="只分析匹配此模式的数据集 (支持通配符)")
    parser.add_argument("--disable-gpu", action="store_true", help="禁用GPU加速，强制使用CPU计算")

    args = parser.parse_args()

    # 创建输出目录
    if args.save_plots:
        os.makedirs(args.output_dir, exist_ok=True)

    print("Wasserstein距离计算可视化演示")
    print("=" * 70)

    # 1. 加载数据
    print(f"加载路由数据: {args.router_data_path}")
    all_router_data = load_all_router_data_files(args.router_data_path)
    
    # 过滤数据集
    if args.dataset_filter:
        import fnmatch
        filtered_data = {}
        for dataset_name in all_router_data:
            if fnmatch.fnmatch(dataset_name, args.dataset_filter):
                filtered_data[dataset_name] = all_router_data[dataset_name]
        all_router_data = filtered_data
        print(f"应用过滤器 '{args.dataset_filter}', 匹配到 {len(all_router_data)} 个数据集")
    
    print(f"将分析 {len(all_router_data)} 个数据集: {list(all_router_data.keys())}")
    
    # 分析每个数据集
    all_results = []
    for dataset_name, router_data in all_router_data.items():
        result = analyze_single_dataset_wasserstein(dataset_name, router_data, args)
        all_results.append(result)
    
    # 生成总体分析报告
    if len(all_results) > 1:
        print(f"\n{'=' * 80}")
        print("总体Wasserstein距离分析报告")
        print(f"{'=' * 80}")
        
        avg_total_distances = [r['total_distance'] for r in all_results]
        avg_distances = [r['average_distance'] for r in all_results]
        
        print(f"分析了 {len(all_results)} 个数据集")
        print(f"总距离范围: {min(avg_total_distances):.4f} - {max(avg_total_distances):.4f}")
        print(f"平均样本间距离范围: {min(avg_distances):.4f} - {max(avg_distances):.4f}")
        print()
        
        # 按数据集展示统计信息
        print("各数据集Wasserstein距离统计:")
        print(f"{'数据集':<15} {'样本总距离':<12} {'平均层距离':<12} {'最大层距离':<12} {'样本平均距离':<15}")
        print("-" * 75)
        for result in all_results:
            avg_layer_distance = np.mean(result['layer_distances'])
            max_layer_distance = np.max(result['layer_distances'])
            print(f"{result['dataset_name']:<15} {result['total_distance']:<12.4f} "
                  f"{avg_layer_distance:<12.4f} {max_layer_distance:<12.4f} {result['average_distance']:<15.4f}")

    print("\n" + "=" * 70)
    print("关键概念解释：")
    print("1. Wasserstein距离衡量两个概率分布之间的差异")
    print("2. 在MoE模型中，每层每个token都会产生专家选择概率")
    print("3. 我们对每个样本在每层的专家概率进行平均，得到该样本该层的代表性分布")
    print("4. 通过计算所有层的Wasserstein距离总和，得到两个样本的总体差异")
    print("5. 距离越大的样本在多样性选择中优先级越高")
    print("6. GPU加速支持：使用批量计算法显著提升大规模数据的计算效率")
    
    print("=" * 70)
    print("可视化完成！")


if __name__ == "__main__":
    main()
