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
import torch

try:
    from cuml.manifold import TSNE as CumlTSNE

    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False

from sklearn.manifold import TSNE

from src.clustering import ClusterBasedSelection
from src.stages.selection import load_router_data


def optimize_tsne_parameters(n_samples, n_features):
    """
    根据数据集大小智能选择t-SNE参数

    Args:
        n_samples: 样本数
        n_features: 特征数

    Returns:
        dict: 优化的参数字典
    """
    # 根据数据集大小调整参数
    if n_samples < 1000:
        # 小数据集：使用精确方法，更高质量
        return {"method": "exact", "n_iter": 1500, "perplexity": min(30, n_samples // 3), "early_exaggeration": 12.0, "learning_rate": "auto"}
    elif n_samples < 5000:
        # 中等数据集：平衡速度和质量
        return {"method": "barnes_hut", "n_iter": 1000, "perplexity": min(50, n_samples // 4), "early_exaggeration": 12.0, "learning_rate": "auto"}
    else:
        # 大数据集：优先速度
        return {
            "method": "barnes_hut",
            "n_iter": 750,  # 减少迭代次数加速
            "perplexity": min(50, max(10, n_samples // 10)),
            "early_exaggeration": 8.0,  # 减小早期放大系数
            "learning_rate": 200.0,  # 固定学习率加速收敛
        }


def gpu_accelerated_tsne(features, n_components=2, perplexity=30, random_state=42, use_gpu=True):
    """
    GPU加速的t-SNE实现，自动回退到CPU版本

    Args:
        features: 输入特征矩阵 (numpy array)
        n_components: 输出维度
        perplexity: t-SNE perplexity参数
        random_state: 随机种子
        use_gpu: 是否尝试使用GPU加速

    Returns:
        numpy array: 降维后的坐标
    """
    n_samples, n_features = features.shape

    # 优化参数
    params = optimize_tsne_parameters(n_samples, n_features)
    adjusted_perplexity = min(perplexity, params["perplexity"])

    # 尝试GPU加速的cuML t-SNE
    if use_gpu and CUML_AVAILABLE:
        try:
            # cuML t-SNE参数
            tsne_gpu = CumlTSNE(
                n_components=n_components,
                perplexity=adjusted_perplexity,
                random_state=random_state,
                learning_rate=params["learning_rate"],
                n_iter=params["n_iter"],
                early_exaggeration=params["early_exaggeration"],
                method=params["method"],
            )

            coords_2d = tsne_gpu.fit_transform(features)

            # cuML返回的可能是cupy数组，转换为numpy
            if hasattr(coords_2d, "get"):
                coords_2d = coords_2d.get()  # cupy to numpy
            elif hasattr(coords_2d, "cpu"):
                coords_2d = coords_2d.cpu().numpy()  # torch to numpy

            return coords_2d

        except Exception as e:
            print(f"GPU t-SNE失败，回退到CPU版本: {str(e)}")

    # CPU版本的sklearn t-SNE（回退方案）
    # 注意: sklearn t-SNE不支持learning_rate='auto'和method参数
    learning_rate = 200.0 if params["learning_rate"] == "auto" else params["learning_rate"]
    method = "barnes_hut" if params["method"] == "barnes_hut" else "exact"

    tsne_cpu = TSNE(
        n_components=n_components,
        perplexity=adjusted_perplexity,
        random_state=random_state,
        n_iter=params["n_iter"],
        learning_rate=learning_rate,
        method=method,
    )

    coords_2d = tsne_cpu.fit_transform(features)
    return coords_2d


plt.rcParams["font.sans-serif"] = ["Maple Mono NF CN"]


def analyze_quality_gates(router_data, top_k=10):
    """分析质量门的输出分布"""
    quality_score = router_data["quality_score"]  # [N, L, 1]
    sample_ids = router_data["sample_ids"]

    print("=" * 70)
    print("质量门分析")
    print("=" * 70)

    # 计算每个样本的质量分数（使用sigmoid转换）
    quality_probs = torch.sigmoid(quality_score.float())  # [N, L, 1]
    good_probs = quality_probs.squeeze(-1)  # [N, L] - good概率

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
    moe_logits = router_data["moe_logits"]  # [N, L, E] - 已经是概率分布

    print("=" * 70)
    print("MoE路由分析")
    print("=" * 70)

    # 数据已经是概率分布格式，直接使用
    moe_probs = moe_logits.float()  # [N, L, E]
    n_samples, n_layers, n_experts = moe_probs.shape

    print(f"数据形状: {n_samples} 样本 × {n_layers} 层 × {n_experts} 专家")
    print()

    # 逐层计算专家使用统计和平衡度
    layer_expert_usage = []  # 每层的专家使用率
    layer_balance_scores = []  # 每层的平衡度
    layer_entropy_stats = []  # 每层的熵统计

    print("各层专家使用和平衡度分析:")
    print(f"{'层':<4} {'平均使用率':<10} {'使用率标准差':<12} {'平衡度':<8} {'平均熵':<8} {'熵标准差':<8}")
    print("-" * 60)

    for layer_idx in range(n_layers):
        # 当前层的数据: [N, E]
        layer_probs = moe_probs[:, layer_idx, :]  # [N, E]

        # 1. 计算当前层的专家使用统计
        layer_expert_usage_rates = layer_probs.mean(dim=0).numpy()  # [E] - 当前层每个专家的平均使用率
        layer_expert_usage.append(layer_expert_usage_rates)

        # 2. 计算当前层的平衡度
        layer_balance = 1 - layer_expert_usage_rates.std() * len(layer_expert_usage_rates)
        layer_balance_scores.append(layer_balance)

        # 3. 计算当前层每个样本的路由熵
        layer_entropy_per_sample = -torch.sum(layer_probs * torch.log(layer_probs + 1e-8), dim=-1).numpy()  # [N]
        layer_mean_entropy = layer_entropy_per_sample.mean()
        layer_std_entropy = layer_entropy_per_sample.std()
        layer_entropy_stats.append({"mean": layer_mean_entropy, "std": layer_std_entropy, "per_sample": layer_entropy_per_sample})

        # 输出当前层的统计信息
        print(
            f"{layer_idx:<4} {layer_expert_usage_rates.mean():<10.4f} {layer_expert_usage_rates.std():<12.4f} "
            f"{layer_balance:<8.4f} {layer_mean_entropy:<8.4f} {layer_std_entropy:<8.4f}"
        )

    print()

    # 计算整体统计（各层平均）
    overall_balance = np.mean(layer_balance_scores)
    overall_entropy_means = [stats["mean"] for stats in layer_entropy_stats]
    overall_avg_entropy = np.mean(overall_entropy_means)
    overall_entropy_std = np.std(overall_entropy_means)  # 层间熵的标准差

    # 计算每个样本在所有层的平均熵
    sample_avg_entropy = np.mean([stats["per_sample"] for stats in layer_entropy_stats], axis=0)  # [N]

    # 整合所有层的专家使用率（用于可视化和其他分析）
    # 注意：这里我们保持每层独立，但为了兼容性，我们计算一个总体的expert_usage
    # 这个值主要用于可视化，实际分析应该使用layer_expert_usage
    combined_expert_usage = np.mean(layer_expert_usage, axis=0)  # [E] - 各层平均后的专家使用率

    print("整体统计（各层平均）:")
    print(f"  模型层数: {n_layers}")
    print(f"  每层专家数: {n_experts}")
    print(f"  整体平衡度: {overall_balance:.4f} (各层平衡度的平均值)")
    print(f"  层间平衡度差异: {np.std(layer_balance_scores):.4f}")
    print()

    print("路由多样性统计（熵）:")
    print(f"  各层平均熵: {overall_avg_entropy:.4f}")
    print(f"  层间熵标准差: {overall_entropy_std:.4f}")
    print(f"  样本平均熵: {sample_avg_entropy.mean():.4f} ± {sample_avg_entropy.std():.4f}")
    print(f"  最大可能熵（单层）: {np.log(n_experts):.4f}")
    print()

    # 返回详细的层级数据以及兼容性数据
    return {
        "combined_expert_usage": combined_expert_usage,  # 兼容性：用于可视化
        "layer_expert_usage": layer_expert_usage,  # 新增：每层的专家使用率
        "layer_balance_scores": layer_balance_scores,  # 新增：每层的平衡度
        "overall_balance": overall_balance,  # 新增：整体平衡度
        "sample_avg_entropy": sample_avg_entropy,  # 兼容性：每个样本的平均熵
        "layer_entropy_stats": layer_entropy_stats,  # 新增：每层的熵统计
        "overall_entropy_stats": {
            "layer_avg_entropy": overall_avg_entropy,
            "layer_entropy_std": overall_entropy_std,
            "sample_avg_entropy_mean": sample_avg_entropy.mean(),
            "sample_avg_entropy_std": sample_avg_entropy.std(),
        },
    }


def perform_clustering_and_visualization(router_data, device=None, clustering_method="kmeans"):
    """
    使用GPU聚类算法进行聚类并准备2D投影可视化
    """
    moe_logits = router_data["moe_logits"]  # [N, L, E] - 已经是概率分布
    n_samples = len(moe_logits)

    # 如果没有指定设备，自动选择
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 准备聚类特征：将 [N, L, E] 展平为 [N, L*E]
    features_matrix = moe_logits.reshape(n_samples, -1).to(device=device, dtype=torch.float32)

    # 2. 使用GPU聚类算法
    cluster_selector = ClusterBasedSelection(device=device, debug_print=False)

    # 准备聚类参数
    clustering_params = {
        "auto_k": False,
        "k_range": [max(2, min(10, n_samples // 100)), min(50, n_samples // 10)],
        "k": 26,
        "max_iters": 300,
    }

    if clustering_method.lower() == "kmeans":
        cluster_labels, cluster_info = cluster_selector._kmeans_clustering(features_matrix, clustering_params)
    elif clustering_method.lower() == "hdbscan":
        hdbscan_params = {
            "min_cluster_size": max(10, n_samples // 100),
            "metric": "cosine",
            "use_gpu": True,
        }
        cluster_labels, cluster_info = cluster_selector._hdbscan_clustering(features_matrix, hdbscan_params)
    else:
        raise ValueError(f"不支持的聚类方法: {clustering_method}")

    # 3. 2D投影使用GPU加速的t-SNE
    features_cpu = features_matrix.cpu().numpy()

    coords_2d = gpu_accelerated_tsne(features_cpu, n_components=2, perplexity=min(30, n_samples // 4), random_state=42, use_gpu=True)

    return coords_2d, cluster_labels.cpu().numpy(), cluster_info


def create_simplified_visualization(router_data, coords_2d, cluster_labels, cluster_info, quality_scores, save_path=None):
    """创建简化的三个子图可视化"""
    sample_ids = router_data["sample_ids"]
    quality_score_raw = router_data["quality_score"]  # [N, L, 1]

    # 使用1x3布局
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    plt.subplots_adjust(wspace=0.3)

    # 子图1: 样本质量分数分布
    ax1 = axes[0]
    ax1.hist(quality_scores, bins=50, alpha=0.7, color="skyblue", edgecolor="black")
    ax1.set_xlabel("质量分数")
    ax1.set_ylabel("样本数量")
    ax1.set_title(f"样本质量分数分布\n总样本: {len(quality_scores)}个")
    ax1.grid(True, alpha=0.3)

    # 添加统计信息
    mean_score = quality_scores.mean()
    std_score = quality_scores.std()
    ax1.axvline(mean_score, color="red", linestyle="--", alpha=0.7, label=f"平均值: {mean_score:.3f}")
    ax1.legend()

    # 子图2: 各层质量分数分布
    ax2 = axes[1]
    quality_by_layer = torch.sigmoid(quality_score_raw.float()).squeeze(-1)  # [N, L]
    layer_avg_quality = quality_by_layer.mean(dim=0).numpy()  # [L]
    layer_std_quality = quality_by_layer.std(dim=0).numpy()  # [L]

    x_layers = range(len(layer_avg_quality))
    ax2.plot(x_layers, layer_avg_quality, "o-", linewidth=2, markersize=6, color="blue", label="平均质量")
    ax2.fill_between(x_layers, layer_avg_quality - layer_std_quality, layer_avg_quality + layer_std_quality, alpha=0.2, color="blue", label="标准差")
    ax2.set_xlabel("层索引")
    ax2.set_ylabel("平均质量分数")
    ax2.set_title("各层质量分数分布")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 子图3: 二级路由 2D 投影 + 聚类结果
    ax3 = axes[2]

    # 获取独特的聚类标签
    unique_labels = np.unique(cluster_labels)
    n_clusters = len(unique_labels[unique_labels >= 0])  # 排除噪声点(-1)

    # 为不同簇生成颜色
    colors = plt.cm.Set1(np.linspace(0, 1, max(n_clusters, 1)))

    # 绘制数据点 (减小点的半径)
    for i, label in enumerate(unique_labels):
        if label == -1:
            # 噪声点用灰色显示 (s 从 20 减小到 8)
            mask = cluster_labels == label
            ax3.scatter(coords_2d[mask, 0], coords_2d[mask, 1], c="gray", s=1, alpha=0.5, label=f"噪声 ({np.sum(mask)}个)", marker="x")
        else:
            # 正常簇用不同颜色 (s 从 30 减小到 12)
            mask = cluster_labels == label
            color = colors[i % len(colors)] if i < len(colors) else colors[i % len(colors)]
            ax3.scatter(coords_2d[mask, 0], coords_2d[mask, 1], c=[color], s=2, alpha=0.7, label=f"簇 {label} ({np.sum(mask)}个)")

    ax3.set_xlabel("第一主成分")
    ax3.set_ylabel("第二主成分")
    ax3.set_title(f"二级路由 2D 投影与聚类结果\n{cluster_info.get('method', 'Unknown')} | {n_clusters}个簇")
    ax3.grid(True, alpha=0.3)

    # 只在簇数不太多时显示图例
    if len(unique_labels) <= 10:
        ax3.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")

    # 构建标题信息
    title_info = f"Select-MoE 数据选择分析 - {router_data.get('dataset_name', 'Unknown')}"
    subtitle_info = f"总样本: {len(sample_ids)}, 平均质量: {mean_score:.3f} ± {std_score:.3f}"

    # 如果是聚合数据集，添加数据集构成信息
    if "source_datasets" in router_data:
        datasets_info = ", ".join(router_data["source_datasets"])
        if len(datasets_info) > 60:  # 如果太长则截断
            datasets_info = datasets_info[:57] + "..."
        subtitle_info += f"\n包含数据集: {datasets_info}"

    plt.suptitle(title_info + "\n" + subtitle_info, fontsize=14, y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ 已保存简化分析图: {save_path}")

    return fig


def detect_processing_mode(router_data_path):
    """智能检测处理模式并返回配置信息

    Returns:
        dict: {
            'mode': 'batch' | 'single_experiment' | 'single_file',
            'description': str,
            'auto_aggregate': bool,
            'paths': list  # 需要处理的路径列表
        }
    """
    if os.path.isfile(router_data_path):
        if router_data_path.endswith(".pt"):
            return {"mode": "single_file", "description": "单文件模式 - 分析单个数据集文件", "auto_aggregate": False, "paths": [router_data_path]}
        else:
            raise ValueError(f"不支持的文件格式: {router_data_path}")

    elif os.path.isdir(router_data_path):
        # 检查目录内容来判断模式
        dir_contents = os.listdir(router_data_path)

        # 检查是否包含 *_router_data.pt 文件（单实验模式）
        router_data_files = [f for f in dir_contents if f.endswith("_router_data.pt")]
        if router_data_files:
            return {
                "mode": "single_experiment",
                "description": "单实验合并模式 - 合并单个实验内的所有数据集",
                "auto_aggregate": True,
                "paths": [router_data_path],
            }

        # 检查是否包含实验子目录（批量处理模式）
        experiment_dirs = []
        for item in dir_contents:
            item_path = os.path.join(router_data_path, item)
            if os.path.isdir(item_path):
                # 检查子目录是否包含router_data目录
                router_data_subdir = os.path.join(item_path, "router_data")
                if os.path.isdir(router_data_subdir):
                    # 检查router_data目录是否包含.pt文件
                    router_files = glob.glob(os.path.join(router_data_subdir, "*_router_data.pt"))
                    if router_files:
                        experiment_dirs.append(router_data_subdir)

        if experiment_dirs:
            return {
                "mode": "batch",
                "description": f"批量处理模式 - 处理{len(experiment_dirs)}个实验，每个实验合并数据集",
                "auto_aggregate": True,
                "paths": experiment_dirs,
            }

        # 如果都不匹配，尝试查找嵌套的router_data目录
        nested_router_dirs = glob.glob(os.path.join(router_data_path, "*/router_data"))
        if nested_router_dirs:
            valid_dirs = []
            for router_dir in nested_router_dirs:
                router_files = glob.glob(os.path.join(router_dir, "*_router_data.pt"))
                if router_files:
                    valid_dirs.append(router_dir)

            if valid_dirs:
                return {
                    "mode": "batch",
                    "description": f"批量处理模式 - 处理{len(valid_dirs)}个实验，每个实验合并数据集",
                    "auto_aggregate": True,
                    "paths": valid_dirs,
                }

        raise ValueError(f"目录 {router_data_path} 中未找到有效的router数据文件或子实验目录")

    else:
        raise ValueError(f"路径不存在或不是有效的文件/目录: {router_data_path}")


def parse_experiment_config(experiment_path):
    """从实验路径解析配置信息用于文件命名

    Args:
        experiment_path: 实验目录路径，如 outputs/stage_2_selection/2025-08-11/03-52-40-batch=8_lr=0.001_loss=beta_moment_matching_tag=none/router_data

    Returns:
        str: 格式化的配置名称，如 '03-52-40-batch=8_lr=0.001_loss=beta_moment_matching_tag=none'
    """
    # 获取实验目录名（router_data的父目录）
    if experiment_path.endswith("router_data"):
        experiment_dir = os.path.dirname(experiment_path)
    else:
        experiment_dir = experiment_path

    # 提取目录名
    experiment_name = os.path.basename(experiment_dir)

    # 如果目录名包含时间戳和配置信息，直接使用
    if "-" in experiment_name and ("=" in experiment_name or "batch" in experiment_name or "lr" in experiment_name):
        return experiment_name

    # 否则使用完整路径的最后两级目录名
    parent_dir = os.path.basename(os.path.dirname(experiment_dir))
    return f"{parent_dir}_{experiment_name}"


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
            all_router_data[dataset_name] = load_router_data(file_path)

        return all_router_data
    else:
        raise ValueError(f"路径不是有效的.pt文件或目录: {router_data_path}")


def batch_process_experiments(experiment_paths, args):
    """批量处理多个实验的router数据并生成分析报告

    Args:
        experiment_paths: list of experiment router_data directory paths
        args: command line arguments

    Returns:
        list: 所有实验的分析结果
    """
    all_experiment_results = []

    experiment_output_dir = os.path.join(args.output_dir, "batch_analysis")
    os.makedirs(experiment_output_dir, exist_ok=True)
    args.output_dir = experiment_output_dir

    for experiment_path in experiment_paths:
        try:
            # 解析实验配置名称
            config_name = parse_experiment_config(experiment_path)

            # 加载该实验的所有router数据
            experiment_router_data = load_all_router_data_files(experiment_path)

            if len(experiment_router_data) == 0:
                continue

            # 如果有多个数据集，自动聚合
            if len(experiment_router_data) > 1:
                aggregated_data = aggregate_router_data(experiment_router_data)
            else:
                # 只有一个数据集，直接使用
                dataset_name, router_data = next(iter(experiment_router_data.items()))
                aggregated_data = router_data
                aggregated_data["dataset_name"] = f"{config_name} - {dataset_name}"

            # 分析该实验
            result = analyze_single_dataset(config_name, aggregated_data, args)
            result["config_name"] = config_name
            result["experiment_path"] = experiment_path
            all_experiment_results.append(result)

        except Exception:
            continue

    # 生成批量处理汇总报告
    if all_experiment_results:
        generate_batch_analysis_report(all_experiment_results, args)

    return all_experiment_results


def generate_batch_analysis_report(all_results, args):
    """生成批量分析的汇总报告"""
    print(f"\n\n{'=' * 90}")
    print("📈 批量分析汇总报告")
    print("=" * 90)

    successful_experiments = len(all_results)
    total_samples = sum(r["n_total"] for r in all_results)

    print(f"成功分析了 {successful_experiments} 个实验，共 {total_samples:,} 个样本")
    print()

    # 统计信息表
    print("各实验详细统计:")
    print(f"{'=' * 100}")
    print(f"{'Config Name':<50} {'Samples':<8} {'Quality':<10} {'Entropy':<10} {'Clusters':<8} {'Balance':<10}")
    print("-" * 100)

    quality_scores = []
    entropy_scores = []
    cluster_counts = []
    balance_scores = []

    for result in all_results:
        avg_quality = result["quality_scores"].mean()
        avg_entropy = result["sample_entropy"].mean()
        # 使用新的overall_balance值
        balance = result.get("overall_balance", 0.0)
        if balance == 0.0 and "expert_usage" in result:
            # 兼容旧版本的计算方式
            expert_usage_array = result["expert_usage"]
            if isinstance(expert_usage_array, np.ndarray):
                balance = 1 - expert_usage_array.std() * len(expert_usage_array)
            else:
                balance = 0.0

        quality_scores.append(avg_quality)
        entropy_scores.append(avg_entropy)
        cluster_counts.append(result["n_clusters"])
        balance_scores.append(balance)

        # 截断配置名称以适合显示
        config_display = result["config_name"][:47] + "..." if len(result["config_name"]) > 50 else result["config_name"]

        print(f"{config_display:<50} {result['n_total']:<8} {avg_quality:<10.4f} {avg_entropy:<10.4f} {result['n_clusters']:<8} {balance:<10.4f}")

    print("\n")

    # 汇总统计
    print("汇总统计:")
    print(f"Quality Score - Min: {min(quality_scores):.4f}, Max: {max(quality_scores):.4f}, Avg: {sum(quality_scores) / len(quality_scores):.4f}")
    print(f"Routing Entropy - Min: {min(entropy_scores):.4f}, Max: {max(entropy_scores):.4f}, Avg: {sum(entropy_scores) / len(entropy_scores):.4f}")
    print(f"Cluster Count - Min: {min(cluster_counts)}, Max: {max(cluster_counts)}, Avg: {sum(cluster_counts) / len(cluster_counts):.1f}")
    print(f"Expert Balance - Min: {min(balance_scores):.4f}, Max: {max(balance_scores):.4f}, Avg: {sum(balance_scores) / len(balance_scores):.4f}")

    # 保存报告到文件
    report_path = os.path.join(args.output_dir, "batch_analysis_report.txt")
    os.makedirs(args.output_dir, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Select-MoE 批量分析报告\n")
        f.write(f"{'=' * 50}\n")
        f.write(f"分析时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"成功分析: {successful_experiments} 个实验\n")
        f.write(f"总样本数: {total_samples:,} 个\n\n")

        f.write("各实验详情:\n")
        f.write("-" * 120 + "\n")
        for result in all_results:
            avg_quality = result["quality_scores"].mean()
            avg_entropy = result["sample_entropy"].mean()
            # 使用新的overall_balance值
            balance = result.get("overall_balance", 0.0)
            if balance == 0.0 and "expert_usage" in result:
                # 兼容旧版本的计算方式
                expert_usage_array = result["expert_usage"]
                if isinstance(expert_usage_array, np.ndarray):
                    balance = 1 - expert_usage_array.std() * len(expert_usage_array)
                else:
                    balance = 0.0

            f.write(f"Config: {result['config_name']}\n")
            f.write(f"  Path: {result['experiment_path']}\n")
            f.write(f"  Samples: {result['n_total']:,}, Quality: {avg_quality:.4f}, Entropy: {avg_entropy:.4f}\n")
            f.write(f"  Clusters: {result['n_clusters']}, Balance: {balance:.4f}\n\n")

    print(f"\n💾 详细报告已保存至: {report_path}")
    print(f"🖼️  所有实验的分析图表已保存在: {os.path.join(args.output_dir, 'batch_analysis')}")
    print("=" * 90)


def aggregate_router_data(all_router_data):
    """将多个数据集的router_data合并为一个整体数据集"""
    if len(all_router_data) == 1:
        # 只有一个数据集，直接返回
        dataset_name, router_data = next(iter(all_router_data.items()))
        router_data["dataset_name"] = f"{dataset_name} (单一数据集)"
        return router_data

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
        quality_score = router_data["quality_score"]  # [N, L, 1] - 新格式
        moe_logits = router_data["moe_logits"]  # [N, L, E]

        # 检查形状兼容性
        if reference_quality_shape is None:
            reference_quality_shape = quality_score.shape[1:]  # [L, 1]
            reference_moe_shape = moe_logits.shape[1:]  # [L, E]
        else:
            if quality_score.shape[1:] != reference_quality_shape:
                raise ValueError(f"数据集 {dataset_name} 的quality_score形状 {quality_score.shape[1:]} 与参考形状 {reference_quality_shape} 不兼容")
            if moe_logits.shape[1:] != reference_moe_shape:
                raise ValueError(f"数据集 {dataset_name} 的moe_logits形状 {moe_logits.shape[1:]} 与参考形状 {reference_moe_shape} 不兼容")

        sample_count = len(router_data["sample_ids"])
        sample_counts.append(sample_count)

        # 添加数据集前缀到sample_ids以避免冲突
        prefixed_sample_ids = [f"{dataset_name}_{sid}" for sid in router_data["sample_ids"]]

        all_quality_logits.append(quality_score)
        all_moe_logits.append(moe_logits)
        all_sample_ids.extend(prefixed_sample_ids)

    # 合并张量
    aggregated_quality_logits = torch.cat(all_quality_logits, dim=0)
    aggregated_moe_logits = torch.cat(all_moe_logits, dim=0)

    # 构建聚合后的router_data
    aggregated_data = {
        "quality_score": aggregated_quality_logits,  # 使用新的键名
        "moe_logits": aggregated_moe_logits,
        "sample_ids": all_sample_ids,
        "dataset_name": f"所有数据集聚合 ({len(dataset_names)}个数据集)",
        "source_datasets": dataset_names,
        "dataset_sample_counts": dict(zip(dataset_names, sample_counts, strict=True)),
    }

    return aggregated_data


def analyze_single_dataset(dataset_name, router_data, args, config_name=None):
    """分析单个数据集"""
    # 1. 分析质量门
    quality_scores, _ = analyze_quality_gates(router_data)

    # 2. 分析MoE路由
    routing_analysis = analyze_moe_routing(router_data)
    expert_usage = routing_analysis["combined_expert_usage"]  # 用于兼容性
    sample_entropy = routing_analysis["sample_avg_entropy"]
    overall_balance = routing_analysis["overall_balance"]

    # 3. 执行聚类和2D投影
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coords_2d, cluster_labels, cluster_info = perform_clustering_and_visualization(router_data, device=device, clustering_method=args.clustering_method)

    # 4. 创建简化可视化
    save_path = None

    # 根据配置名称或数据集名称生成保存路径
    if config_name and config_name != dataset_name:
        # 批量处理模式，使用配置名称
        safe_name = config_name.replace("/", "_").replace("\\", "_")
        save_path = os.path.join(args.output_dir, f"{safe_name}.png")
    else:
        # 单数据集分析模式
        if dataset_name == "聚合数据集" and "source_datasets" in router_data:
            # 聚合数据集使用特殊命名
            safe_dataset_name = f"aggregated_{'_'.join(router_data['source_datasets'])}"
            safe_dataset_name = safe_dataset_name.replace("/", "_").replace("\\", "_")
        else:
            safe_dataset_name = dataset_name.replace("/", "_").replace("\\", "_")
        save_path = os.path.join(args.output_dir, f"{safe_dataset_name}.png")

    create_simplified_visualization(router_data, coords_2d, cluster_labels, cluster_info, quality_scores, save_path)

    plt.show()

    # 5. 生成分析报告
    n_total = len(router_data["sample_ids"])
    n_clusters = len(np.unique(cluster_labels[cluster_labels >= 0]))

    return {
        "dataset_name": dataset_name,
        "quality_scores": quality_scores,
        "expert_usage": expert_usage,
        "sample_entropy": sample_entropy,
        "cluster_info": cluster_info,
        "n_total": n_total,
        "n_clusters": n_clusters,
        "overall_balance": overall_balance,  # 新增：整体平衡度
        "routing_analysis": routing_analysis,  # 新增：详细的路由分析结果
    }


def main():
    parser = argparse.ArgumentParser(description="Select-MoE智能数据选择分析")
    parser.add_argument("router_data_path", help="路由数据文件路径(.pt格式)或包含多个router_data文件的目录")
    parser.add_argument("--clustering-method", choices=["kmeans", "hdbscan"], default="kmeans", help="聚类方法 (默认: kmeans)")
    parser.add_argument("--output-dir", default="./outputs/visual_figs", help="图片保存目录")
    parser.add_argument("--dataset-filter", help="只分析匹配此模式的数据集 (支持通配符)")
    parser.add_argument("--aggregate-datasets", action="store_true", help="将多个数据集聚合为一个整体进行分析，而不是分别分析每个数据集")

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    print("Select-MoE智能数据选择综合分析")
    print("=" * 70)

    # 🔍 智能检测处理模式
    print(f"🔍 分析路径: {args.router_data_path}")

    try:
        mode_config = detect_processing_mode(args.router_data_path)
        print(f"✅ 检测到处理模式: {mode_config['mode']}")
        print(f"📄 {mode_config['description']}")
        print(f"🎯 待处理路径数量: {len(mode_config['paths'])}")
        print()

        # 根据模式自动设置聚合选项
        if mode_config["auto_aggregate"]:
            args.aggregate_datasets = True
            print("🔧 自动启用数据集聚合模式")

    except ValueError as e:
        print(f"❌ 路径检测失败: {e}")
        return

    # 🚀 根据检测到的模式执行相应的处理逻辑
    if mode_config["mode"] == "batch":
        # 批量处理模式：处理多个实验目录
        print("\n🔄 启动批量处理模式")
        batch_process_experiments(mode_config["paths"], args)

    elif mode_config["mode"] == "single_experiment":
        # 单实验模式：处理单个实验目录内的router_data
        print("\n⚙️ 启动单实验分析模式")

        router_data_path = mode_config["paths"][0]
        print(f"加载路由数据: {router_data_path}")
        all_router_data = load_all_router_data_files(router_data_path)

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
                balance = result.get("overall_balance", 0.0)
                if balance == 0.0 and "expert_usage" in result:
                    # 兼容旧版本的计算方式
                    expert_usage_array = result["expert_usage"]
                    if isinstance(expert_usage_array, np.ndarray):
                        balance = 1 - expert_usage_array.std() * len(expert_usage_array)
                    else:
                        balance = 0.0
                print(
                    f"{result['dataset_name']:<15} {result['n_total']:<8} {result['quality_scores'].mean():<10.4f} "
                    f"{result['sample_entropy'].mean():<10.4f} {balance:<12.4f}"
                )

    elif mode_config["mode"] == "single_file":
        # 单文件模式：直接分析单个.pt文件
        print("\n📁 启动单文件分析模式")

        router_data_path = mode_config["paths"][0]
        print(f"加载路由数据文件: {router_data_path}")
        all_router_data = load_all_router_data_files(router_data_path)

        # 单文件模式只有一个数据集
        dataset_name, router_data = next(iter(all_router_data.items()))
        analyze_single_dataset(dataset_name, router_data, args)

    else:
        print(f"❌ 不支持的处理模式: {mode_config['mode']}")
        return

    print("=" * 70)
    print("🎉 Select-MoE智能分析完成！")
    print()
    print("📊 关键洞察:")
    print("1. 质量分数反映了样本的整体数据质量")
    print("2. 各层质量分数分布展示了模型层级间的质量判断一致性")
    print("3. 聚类结果显示了MoE路由模式的数据分组特征")
    print("4. GPU加速的聚类算法有效处理大规模数据集")

    if mode_config["mode"] == "batch":
        print("5. 批量处理模式支持多实验对比分析")
        print(f"📁 批量分析结果保存在: {os.path.join(args.output_dir)}")

    print("=" * 70)


if __name__ == "__main__":
    main()
