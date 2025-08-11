#!/usr/bin/env python3
"""
Select-MoE数据选择综合可视化分析脚本

本脚本提供了对Select-MoE数据选择过程的全面分析和可视化，包括：
1. 质量门分析：展示样本的质量分数分布
2. MoE路由分析：分析专家选择模式
3. 逐层二级路由余弦相似度计算：样本间相似性度量
4. FPS算法应用：多样性选择过程
5. 选择结果对比：质量选择 vs 多样性选择

这个脚本帮助理解整个Select-MoE数据选择管道的工作原理。
支持读取整个router_data文件夹内的所有.pt文件，并对每个数据集进行分析。
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
from sklearn.manifold import MDS

from src.stages.selection import (
    compute_batch_cosine_distance_gpu,
    farthest_point_sampling_gpu,
    load_router_data,
)

plt.rcParams["font.sans-serif"] = ["Maple Mono NF CN"]


def analyze_quality_gates(router_data, top_k=10):
    """分析质量门的输出分布"""
    quality_logits = router_data["quality_logits"]  # [N, L, 2]
    sample_ids = router_data["sample_ids"]

    print("=" * 70)
    print("质量门分析")
    print("=" * 70)

    # 计算每个样本的质量分数（good概率的平均值）
    quality_probs = torch.softmax(quality_logits.float(), dim=-1)  # [N, L, 2]
    good_probs = quality_probs[:, :, 0]  # [N, L] - good类别的概率

    # 每个样本在所有层的平均good概率作为质量分数
    sample_quality_scores = good_probs.mean(dim=1).numpy()  # [N]

    print("质量分数统计:")
    print(f"  平均值: {sample_quality_scores.mean():.4f}")
    print(f"  标准差: {sample_quality_scores.std():.4f}")
    print(f"  最小值: {sample_quality_scores.min():.4f}")
    print(f"  最大值: {sample_quality_scores.max():.4f}")
    print()

    # 找出质量最高和最低的样本
    sorted_indices = np.argsort(sample_quality_scores)

    print(f"质量最低的{top_k}个样本:")
    for i in range(min(top_k, len(sorted_indices))):
        idx = sorted_indices[i]
        print(f"  {sample_ids[idx]}: {sample_quality_scores[idx]:.4f}")
    print()

    print(f"质量最高的{top_k}个样本:")
    for i in range(min(top_k, len(sorted_indices))):
        idx = sorted_indices[-(i + 1)]
        print(f"  {sample_ids[idx]}: {sample_quality_scores[idx]:.4f}")
    print()

    return sample_quality_scores, sorted_indices


def analyze_moe_routing(router_data):
    """分析MoE路由模式"""
    moe_logits = router_data["moe_logits"]  # [N, L, E]

    print("=" * 70)
    print("MoE路由分析")
    print("=" * 70)

    # 转换为概率分布
    moe_probs = torch.softmax(moe_logits.float(), dim=-1)  # [N, L, E]

    # 计算专家使用统计
    expert_usage = moe_probs.mean(dim=(0, 1)).numpy()  # [E] - 每个专家的平均使用率

    print("专家使用统计:")
    print(f"  专家数量: {len(expert_usage)}")
    print(f"  平均使用率: {expert_usage.mean():.4f}")
    print(f"  使用率标准差: {expert_usage.std():.4f}")
    print(f"  负载平衡度: {1 - expert_usage.std() * len(expert_usage):.4f}")
    print()

    # 分析路由多样性
    # 计算每个样本的路由熵（跨专家维度）
    entropy_per_layer = -torch.sum(moe_probs * torch.log(moe_probs + 1e-8), dim=-1)  # [N, L]
    sample_avg_entropy = entropy_per_layer.mean(dim=1).numpy()  # [N]

    print("路由多样性统计（熵）:")
    print(f"  平均熵: {sample_avg_entropy.mean():.4f}")
    print(f"  熵标准差: {sample_avg_entropy.std():.4f}")
    print(f"  最大可能熵: {np.log(len(expert_usage)):.4f}")
    print()

    return expert_usage, sample_avg_entropy


def compute_comprehensive_distances(router_data, max_samples=50):
    """计算样本间的综合距离矩阵"""
    moe_logits = router_data["moe_logits"]  # [N, L, E]

    # 限制样本数量
    n_samples = min(len(moe_logits), max_samples)
    selected_indices = np.linspace(0, len(moe_logits) - 1, n_samples, dtype=int)

    moe_subset = moe_logits[selected_indices]  # [n_samples, L, E]
    distance_matrix = np.zeros((n_samples, n_samples))

    print("=" * 70)
    print("逐层二级路由余弦相似度计算")
    print("=" * 70)
    print(f"计算{n_samples}个样本的距离矩阵...")

    # 尝试使用GPU加速计算
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        print("使用GPU加速计算...")
        # 转换为概率并移至GPU
        moe_probs = torch.softmax(moe_subset.float(), dim=-1).to(device)

        # compute_batch_cosine_distance_gpu需要批次间比较，不能直接计算N×N矩阵
        # 改用循环方式逐对计算距离
        distance_matrix = torch.zeros(n_samples, n_samples, device=device)
        for i in range(n_samples):
            for j in range(i, n_samples):  # 只计算上三角矩阵
                sample_i = moe_probs[i : i + 1]  # [1, L, E]
                sample_j = moe_probs[j : j + 1]  # [1, L, E]
                dist = compute_batch_cosine_distance_gpu(sample_i, sample_j)
                distance_matrix[i, j] = dist[0, 0]  # 提取标量值
                distance_matrix[j, i] = dist[0, 0]  # 对称填充

        distance_matrix = distance_matrix.cpu().numpy()
    else:
        print("使用CPU计算...")
        # CPU计算
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                # 转换为概率分布
                probs_i = torch.softmax(moe_subset[i].float(), dim=-1)
                probs_j = torch.softmax(moe_subset[j].float(), dim=-1)

                # 计算逐层余弦相似度距离
                total_similarity = 0
                for layer_idx in range(probs_i.shape[0]):
                    prob1 = probs_i[layer_idx].numpy()
                    prob2 = probs_j[layer_idx].numpy()

                    # 计算余弦相似度
                    dot_product = np.dot(prob1, prob2)
                    norm1 = np.linalg.norm(prob1)
                    norm2 = np.linalg.norm(prob2)
                    cosine_sim = dot_product / (norm1 * norm2 + 1e-8)
                    total_similarity += cosine_sim

                # 转换为距离
                total_dist = 1.0 - total_similarity

                distance_matrix[i, j] = total_dist
                distance_matrix[j, i] = total_dist

    print("✓ 距离矩阵计算完成")
    # 过滤掉对角线上的0值和无效值
    non_zero_distances = distance_matrix[distance_matrix > 0]
    if len(non_zero_distances) > 0:
        print(f"  平均距离: {non_zero_distances.mean():.4f}")
        print(f"  距离标准差: {non_zero_distances.std():.4f}")
    else:
        print("  警告: 所有距离都为0或无效，可能计算出现问题")
    print()

    return distance_matrix, selected_indices


def perform_selection_comparison(router_data, distance_matrix, demo_indices, quality_scores, selection_ratio=0.2):
    """对比不同的选择策略"""
    n_total = len(demo_indices)
    n_select = max(1, int(n_total * selection_ratio))

    print("=" * 70)
    print("选择策略对比")
    print("=" * 70)
    print(f"从{n_total}个样本中选择{n_select}个样本（选择率: {selection_ratio:.1%}）")
    print()

    # 1. 质量选择：选择质量分数最高的样本
    quality_subset = quality_scores[demo_indices]
    quality_selected_local = np.argsort(quality_subset)[-n_select:]  # 本地索引
    quality_selected_global = demo_indices[quality_selected_local]  # 全局索引

    print("1. 质量选择（选择质量分数最高的样本）:")
    for i, global_idx in enumerate(quality_selected_global):
        local_idx = quality_selected_local[i]
        print(f"   样本{global_idx} (本地{local_idx}): 质量分数 {quality_subset[local_idx]:.4f}")
    print()

    # 2. 多样性选择：使用FPS算法
    distance_tensor = torch.from_numpy(distance_matrix).float()
    if torch.cuda.is_available():
        distance_tensor = distance_tensor.cuda()
    fps_selected_local = farthest_point_sampling_gpu(distance_tensor, n_select, seed=42)
    fps_selected_global = demo_indices[fps_selected_local]

    print("2. 多样性选择（FPS算法）:")
    for i, local_idx in enumerate(fps_selected_local):
        global_idx = fps_selected_global[i]
        print(f"   样本{global_idx} (本地{local_idx}): 质量分数 {quality_subset[local_idx]:.4f}")
    print()

    # 3. 随机选择作为基准
    np.random.seed(42)
    random_selected_local = np.random.choice(n_total, n_select, replace=False)
    random_selected_global = demo_indices[random_selected_local]

    print("3. 随机选择（基准）:")
    for i, local_idx in enumerate(random_selected_local):
        global_idx = random_selected_global[i]
        print(f"   样本{global_idx} (本地{local_idx}): 质量分数 {quality_subset[local_idx]:.4f}")
    print()

    # 计算选择结果的统计指标
    strategies = {
        "Quality": (quality_selected_local, quality_selected_global),
        "Diversity": (fps_selected_local, fps_selected_global),
        "Random": (random_selected_local, random_selected_global),
    }

    print("选择结果统计:")
    print("-" * 50)

    for strategy_name, (local_indices, _global_indices) in strategies.items():
        # 质量分数统计
        selected_quality = quality_subset[local_indices]
        avg_quality = selected_quality.mean()

        # 多样性统计（选中样本间的平均距离）
        selected_distances = []
        for i in range(len(local_indices)):
            for j in range(i + 1, len(local_indices)):
                selected_distances.append(distance_matrix[local_indices[i], local_indices[j]])

        avg_diversity = np.mean(selected_distances) if selected_distances else 0

        print(f"{strategy_name:>10}: 平均质量={avg_quality:.4f}, 平均距离={avg_diversity:.4f}")

    print()

    return strategies


def create_comprehensive_visualization(router_data, distance_matrix, demo_indices, quality_scores, expert_usage, sample_entropy, strategies, save_path=None):
    """创建综合可视化图表"""
    sample_ids = router_data["sample_ids"]
    moe_logits = router_data["moe_logits"]
    quality_logits = router_data["quality_logits"]

    # 使用GridSpec创建2x3的布局
    fig = plt.figure(figsize=(24, 14))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.25)

    # --- 创建6个子图 ---
    ax1 = fig.add_subplot(gs[0, 0])  # 质量分数分布
    ax2 = fig.add_subplot(gs[0, 1])  # 各层质量分数
    ax3 = fig.add_subplot(gs[0, 2])  # 专家使用率
    ax4 = fig.add_subplot(gs[1, 0])  # 路由多样性
    ax5 = fig.add_subplot(gs[1, 1])  # 距离矩阵热力图
    ax6 = fig.add_subplot(gs[1, 2])  # 样本2D投影分布

    # 1. 质量分数分布
    ax1.hist(quality_scores, bins=30, alpha=0.7, color="skyblue", edgecolor="black")
    ax1.set_xlabel("质量分数")
    ax1.set_ylabel("样本数量")
    ax1.set_title(f"样本质量分数分布 ({len(quality_scores)}条)")
    ax1.grid(True, alpha=0.3)

    # 2. 各层质量分数
    quality_by_layer = torch.softmax(quality_logits.float(), dim=-1)[:, :, 0]  # [N, L]
    layer_avg_quality = quality_by_layer.mean(dim=0).numpy()  # [L]
    ax2.plot(range(len(layer_avg_quality)), layer_avg_quality, "o-", linewidth=2)
    ax2.set_xlabel("层索引")
    ax2.set_ylabel("平均质量分数")
    ax2.set_title("各层质量分数分布")
    ax2.grid(True, alpha=0.3)

    # 3. MoE专家使用分析
    ax3.bar(range(len(expert_usage)), expert_usage, alpha=0.7, color="lightgreen")
    ax3.set_xlabel("专家索引")
    ax3.set_ylabel("平均使用率")
    ax3.set_title("专家使用率分布")
    ax3.grid(True, alpha=0.3)

    # 4. 样本路由多样性
    ax4.hist(sample_entropy, bins=30, alpha=0.7, color="orange", edgecolor="black")
    ax4.set_xlabel("路由熵")
    ax4.set_ylabel("样本数量")
    ax4.set_title("样本路由多样性分布")
    ax4.grid(True, alpha=0.3)

    # 5. 距离矩阵热力图
    mask = np.triu(np.ones_like(distance_matrix, dtype=bool), k=1)
    sns.heatmap(distance_matrix, mask=~mask, cmap="viridis", square=True, cbar_kws={"label": "余弦相似度距离"}, ax=ax5)
    ax5.set_title(f"样本间余弦相似度距离矩阵\n({len(demo_indices)}个样本)")

    # 6. 样本2D投影分布
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, max_iter=3000, eps=1e-9)
    coords_2d = mds.fit_transform(distance_matrix)

    # 根据质量分数给样本着色
    quality_subset = quality_scores[demo_indices]
    scatter = ax6.scatter(
        coords_2d[:, 0],
        coords_2d[:, 1],
        c=quality_subset,
        cmap="viridis",
        s=60,
        alpha=0.7,
        edgecolors="black",
        linewidth=0.5,
    )

    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax6)
    cbar.set_label("质量分数", rotation=270, labelpad=15)

    # 突出显示多样性选择的结果
    diversity_local_indices, _ = strategies["Diversity"]
    selected_coords = coords_2d[diversity_local_indices]
    ax6.scatter(
        selected_coords[:, 0],
        selected_coords[:, 1],
        marker="*",
        s=200,
        c="red",
        alpha=0.9,
        edgecolors="white",
        linewidth=2,
        label="多样性选择 (FPS)",
    )

    ax6.set_xlabel("第一主成分")
    ax6.set_ylabel("第二主成分")
    ax6.set_title("样本2D投影分布与多样性选择结果")
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # 构建标题信息
    title_info = f"Select-MoE数据选择综合分析 - {router_data['dataset_name']}"
    subtitle_info = f"总样本: {len(sample_ids)}, 分析样本: {len(demo_indices)}"

    # 如果是聚合数据集，添加数据集构成信息
    if "source_datasets" in router_data:
        datasets_info = ", ".join(router_data["source_datasets"])
        if len(datasets_info) > 80:  # 如果太长则截断
            datasets_info = datasets_info[:77] + "..."
        subtitle_info += f"\n包含数据集: {datasets_info}"

    plt.suptitle(
        title_info + "\n" + subtitle_info,
        fontsize=16,
        y=0.98,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ 已保存综合分析图: {save_path}")

    return fig


def load_all_router_data_files(router_data_path):
    """加载router_data文件或目录中的所有router_data文件"""
    if os.path.isfile(router_data_path) and router_data_path.endswith(".pt"):
        # 单个文件
        return {os.path.basename(router_data_path).replace("_router_data.pt", ""): load_router_data(router_data_path)}
    elif os.path.isdir(router_data_path):
        # 目录，查找所有_router_data.pt文件
        router_data_files = glob.glob(os.path.join(router_data_path, "*_router_data.pt"))
        if not router_data_files:
            raise ValueError(f"在目录 {router_data_path} 中未找到任何_router_data.pt文件")

        all_router_data = {}
        for file_path in sorted(router_data_files):
            dataset_name = os.path.basename(file_path).replace("_router_data.pt", "")
            print(f"加载数据集: {dataset_name} - {file_path}")
            all_router_data[dataset_name] = load_router_data(file_path)

        return all_router_data
    else:
        raise ValueError(f"路径不是有效的.pt文件或目录: {router_data_path}")


def aggregate_router_data(all_router_data):
    """将多个数据集的router_data合并为一个整体数据集"""
    if len(all_router_data) == 1:
        # 只有一个数据集，直接返回
        dataset_name, router_data = next(iter(all_router_data.items()))
        router_data["dataset_name"] = f"{dataset_name} (单一数据集)"
        return router_data

    print("=" * 70)
    print("聚合多个数据集的router数据...")
    print("=" * 70)

    # 收集所有数据
    all_quality_logits = []
    all_moe_logits = []
    all_sample_ids = []

    dataset_names = list(all_router_data.keys())
    sample_counts = []

    # 验证张量形状兼容性
    reference_quality_shape = None
    reference_moe_shape = None

    for dataset_name, router_data in all_router_data.items():
        quality_logits = router_data["quality_logits"]  # [N, L, 2]
        moe_logits = router_data["moe_logits"]  # [N, L, E]

        # 检查形状兼容性
        if reference_quality_shape is None:
            reference_quality_shape = quality_logits.shape[1:]  # [L, 2]
            reference_moe_shape = moe_logits.shape[1:]  # [L, E]
        else:
            if quality_logits.shape[1:] != reference_quality_shape:
                raise ValueError(f"数据集 {dataset_name} 的quality_logits形状 {quality_logits.shape[1:]} 与参考形状 {reference_quality_shape} 不兼容")
            if moe_logits.shape[1:] != reference_moe_shape:
                raise ValueError(f"数据集 {dataset_name} 的moe_logits形状 {moe_logits.shape[1:]} 与参考形状 {reference_moe_shape} 不兼容")

        sample_count = len(router_data["sample_ids"])
        sample_counts.append(sample_count)

        # 添加数据集前缀到sample_ids以避免冲突
        prefixed_sample_ids = [f"{dataset_name}_{sid}" for sid in router_data["sample_ids"]]

        all_quality_logits.append(quality_logits)
        all_moe_logits.append(moe_logits)
        all_sample_ids.extend(prefixed_sample_ids)

        print(f"  {dataset_name}: {sample_count} 样本, 形状 {quality_logits.shape}")

    # 合并张量
    aggregated_quality_logits = torch.cat(all_quality_logits, dim=0)
    aggregated_moe_logits = torch.cat(all_moe_logits, dim=0)

    total_samples = sum(sample_counts)
    print("\n合并完成:")
    print(f"  总样本数: {total_samples}")
    print(f"  质量logits形状: {aggregated_quality_logits.shape}")
    print(f"  MoE logits形状: {aggregated_moe_logits.shape}")
    print(f"  包含数据集: {', '.join(dataset_names)}")
    print()

    # 构建聚合后的router_data
    aggregated_data = {
        "quality_logits": aggregated_quality_logits,
        "moe_logits": aggregated_moe_logits,
        "sample_ids": all_sample_ids,
        "dataset_name": f"所有数据集聚合 ({len(dataset_names)}个数据集)",
        "source_datasets": dataset_names,
        "dataset_sample_counts": dict(zip(dataset_names, sample_counts, strict=True)),
    }

    return aggregated_data


def analyze_single_dataset(dataset_name, router_data, args):
    """分析单个数据集"""
    print(f"\n{'=' * 80}")
    print(f"分析数据集: {dataset_name}")
    print(f"{'=' * 80}")

    print(f"总样本数: {len(router_data['sample_ids'])}")
    print(f"模型层数: {router_data['moe_logits'].shape[1]}")
    print(f"专家数量: {router_data['moe_logits'].shape[2]}")
    print()

    # 2. 分析质量门
    quality_scores, quality_ranking = analyze_quality_gates(router_data)

    # 3. 分析MoE路由
    expert_usage, sample_entropy = analyze_moe_routing(router_data)

    # 4. 计算距离矩阵
    distance_matrix, demo_indices = compute_comprehensive_distances(router_data, args.max_samples)

    # 5. 对比选择策略
    strategies = perform_selection_comparison(router_data, distance_matrix, demo_indices, quality_scores, args.selection_ratio)

    # 6. 创建综合可视化
    print("=" * 70)
    print(f"创建{dataset_name}数据集综合可视化...")
    print("=" * 70)

    save_path = None
    if args.save_plots:
        if dataset_name == "聚合数据集" and "source_datasets" in router_data:
            # 聚合数据集使用特殊命名
            safe_dataset_name = f"aggregated_{'_'.join(router_data['source_datasets'])}"
            safe_dataset_name = safe_dataset_name.replace("/", "_").replace("\\", "_")
        else:
            safe_dataset_name = dataset_name.replace("/", "_").replace("\\", "_")
        save_path = os.path.join(args.output_dir, f"comprehensive_analysis_{safe_dataset_name}.png")

    fig = create_comprehensive_visualization(router_data, distance_matrix, demo_indices, quality_scores, expert_usage, sample_entropy, strategies, save_path)

    plt.show()

    # 7. 生成分析报告
    print("=" * 70)
    print(f"{dataset_name} - 分析总结")
    print("=" * 70)

    n_total = len(router_data["sample_ids"])
    n_analyzed = len(demo_indices)
    n_selected = len(strategies["Quality"][0])

    print(f"数据集分析报告 - {dataset_name}")
    print(f"{'=' * 50}")
    print(f"总样本数量: {n_total}")
    print(f"分析样本数量: {n_analyzed}")
    print(f"选择样本数量: {n_selected} (选择率: {args.selection_ratio:.1%})")
    print()

    print("质量分析:")
    print(f"  平均质量分数: {quality_scores.mean():.4f} ± {quality_scores.std():.4f}")
    print(f"  质量分数范围: [{quality_scores.min():.4f}, {quality_scores.max():.4f}]")
    print()

    print("路由分析:")
    print(f"  专家负载平衡度: {1 - expert_usage.std() * len(expert_usage):.4f}")
    print(f"  平均路由熵: {sample_entropy.mean():.4f} ± {sample_entropy.std():.4f}")
    print(f"  最大可能熵: {np.log(len(expert_usage)):.4f}")
    print()

    print("距离分析:")
    non_zero_distances = distance_matrix[distance_matrix > 0]
    if len(non_zero_distances) > 0:
        print(f"  平均样本距离: {non_zero_distances.mean():.4f} ± {non_zero_distances.std():.4f}")
        print(f"  距离范围: [{non_zero_distances.min():.4f}, {non_zero_distances.max():.4f}]")
    else:
        print("  警告: 所有距离都为0或无效")
    print()

    print("选择策略效果:")
    for strategy_name, (local_indices, _) in strategies.items():
        quality_subset = quality_scores[demo_indices]
        selected_quality = quality_subset[local_indices]

        selected_distances = []
        for i in range(len(local_indices)):
            for j in range(i + 1, len(local_indices)):
                selected_distances.append(distance_matrix[local_indices[i], local_indices[j]])

        avg_diversity = np.mean(selected_distances) if selected_distances else 0

        print(f"  {strategy_name:>10}: 质量={selected_quality.mean():.4f}, 多样性={avg_diversity:.4f}")

    return {
        "dataset_name": dataset_name,
        "quality_scores": quality_scores,
        "expert_usage": expert_usage,
        "sample_entropy": sample_entropy,
        "strategies": strategies,
        "n_total": n_total,
    }


def main():
    parser = argparse.ArgumentParser(description="Select-MoE综合数据选择分析")
    parser.add_argument("router_data_path", help="路由数据文件路径(.pt格式)或包含多个router_data文件的目录")
    parser.add_argument("--max-samples", type=int, default=40, help="分析的最大样本数 (默认: 40)")
    parser.add_argument("--selection-ratio", type=float, default=0.2, help="选择比例 (默认: 0.2)")
    parser.add_argument("--save-plots", action="store_true", help="保存图片到文件")
    parser.add_argument("--output-dir", default="./outputs/visual_figs/comprehensive_analysis", help="图片保存目录")
    parser.add_argument("--dataset-filter", help="只分析匹配此模式的数据集 (支持通配符)")
    parser.add_argument("--aggregate-datasets", action="store_true", help="将多个数据集聚合为一个整体进行分析，而不是分别分析每个数据集")

    args = parser.parse_args()

    # 创建输出目录
    if args.save_plots:
        os.makedirs(args.output_dir, exist_ok=True)

    print("Select-MoE数据选择综合分析")
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

    # 根据是否聚合模式决定分析方式
    all_results = []

    if args.aggregate_datasets and len(all_router_data) > 1:
        # 聚合模式：合并所有数据集后统一分析
        print("\n启用聚合分析模式 - 将所有数据集合并为一个整体")
        aggregated_data = aggregate_router_data(all_router_data)
        result = analyze_single_dataset("聚合数据集", aggregated_data, args)
        all_results.append(result)

        # 为聚合模式添加额外的数据集构成信息
        print(f"\n{'=' * 70}")
        print("聚合数据集构成详情")
        print("=" * 70)
        for dataset_name, count in aggregated_data["dataset_sample_counts"].items():
            percentage = count / len(aggregated_data["sample_ids"]) * 100
            print(f"  {dataset_name}: {count} 样本 ({percentage:.1f}%)")
        print()

    else:
        # 原有模式：分别分析每个数据集
        if args.aggregate_datasets:
            print("只有一个数据集，聚合模式无效，使用标准分析模式")

        for dataset_name, router_data in all_router_data.items():
            result = analyze_single_dataset(dataset_name, router_data, args)
            all_results.append(result)

    # 生成总体分析报告 (仅在非聚合模式下有多个数据集时)
    if not args.aggregate_datasets and len(all_results) > 1:
        print(f"\n{'=' * 80}")
        print("总体分析报告")
        print("=" * 80)

        total_samples = sum(r["n_total"] for r in all_results)
        avg_quality_scores = [r["quality_scores"].mean() for r in all_results]
        avg_entropy_scores = [r["sample_entropy"].mean() for r in all_results]

        print(f"分析了 {len(all_results)} 个数据集，共 {total_samples} 个样本")
        print(f"平均质量分数范围: {min(avg_quality_scores):.4f} - {max(avg_quality_scores):.4f}")
        print(f"平均路由熵范围: {min(avg_entropy_scores):.4f} - {max(avg_entropy_scores):.4f}")
        print()

        # 按数据集展示统计信息
        print("各数据集统计:")
        print(f"{'数据集':<15} {'样本数':<8} {'平均质量':<10} {'平均熵':<10} {'专家平衡度':<12}")
        print("-" * 65)
        for result in all_results:
            balance = 1 - result["expert_usage"].std() * len(result["expert_usage"])
            print(
                f"{result['dataset_name']:<15} {result['n_total']:<8} {result['quality_scores'].mean():<10.4f} "
                f"{result['sample_entropy'].mean():<10.4f} {balance:<12.4f}"
            )

    print("\n" + "=" * 70)
    print("关键洞察:")
    print("1. 质量选择优化数据质量，但可能选择相似样本")
    print("2. 多样性选择确保样本覆盖面，但质量可能不是最高")
    print("3. Select-MoE结合两者优势，先质量筛选再多样性选择")
    print("4. 逐层余弦相似度有效衡量MoE路由模式的差异")
    print("5. FPS算法确保选择样本的最大化多样性分布")

    print("=" * 70)
    print("分析完成！")


if __name__ == "__main__":
    main()
