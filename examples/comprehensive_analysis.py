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
    import umap

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

from src.clustering import ClusterBasedSelection
from src.stages.selection import load_router_data

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


def perform_clustering_and_visualization(router_data, device=None, clustering_method="kmeans"):
    """
    使用GPU聚类算法进行聚类并准备2D投影可视化
    """
    moe_logits = router_data["moe_logits"]  # [N, L, E] - 已经是概率分布

    print("=" * 70)
    print("执行聚类算法和2D投影")
    print("=" * 70)

    n_samples = len(moe_logits)
    print(f"处理 {n_samples} 个样本...")

    # 如果没有指定设备，自动选择
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 准备聚类特征：将 [N, L, E] 展平为 [N, L*E]
    features_matrix = moe_logits.reshape(n_samples, -1).to(device=device, dtype=torch.float32)
    print(f"聚类特征矩阵形状: {features_matrix.shape}")

    # 2. 使用GPU聚类算法
    cluster_selector = ClusterBasedSelection(device=device, debug_print=True)

    # 准备聚类参数
    clustering_params = {
        "auto_k": False,
        "k_range": [max(2, min(10, n_samples // 100)), min(50, n_samples // 10)],
        "k": 30,
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

    print(f"聚类完成: {cluster_info}")

    # 3. 2D投影（使用UMAP或PCA）
    print("开始2D投影...")
    features_cpu = features_matrix.cpu().numpy()

    # 根据数据大小选择降维方法
    if n_samples > 10000 or not UMAP_AVAILABLE:
        print("使用PCA进行降维...")
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=2, random_state=42)
        coords_2d = reducer.fit_transform(features_cpu)
    else:
        print("使用UMAP进行高质量降维...")
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, n_samples // 10))
        coords_2d = reducer.fit_transform(features_cpu)

    print(f"2D投影完成，坐标形状: {coords_2d.shape}")

    return coords_2d, cluster_labels.cpu().numpy(), cluster_info


def create_simplified_visualization(router_data, coords_2d, cluster_labels, cluster_info, quality_scores, expert_usage, save_path=None):
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

    # 绘制数据点
    for i, label in enumerate(unique_labels):
        if label == -1:
            # 噪声点用灰色显示
            mask = cluster_labels == label
            ax3.scatter(coords_2d[mask, 0], coords_2d[mask, 1], c="gray", s=20, alpha=0.5, label=f"噪声 ({np.sum(mask)}个)", marker="x")
        else:
            # 正常簇用不同颜色
            mask = cluster_labels == label
            color = colors[i % len(colors)] if i < len(colors) else colors[i % len(colors)]
            ax3.scatter(coords_2d[mask, 0], coords_2d[mask, 1], c=[color], s=30, alpha=0.7, label=f"簇 {label} ({np.sum(mask)}个)")

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

        print(f"  {dataset_name}: {sample_count} 样本, 形状 {quality_score.shape}")

    # 合并张量
    aggregated_quality_logits = torch.cat(all_quality_logits, dim=0)
    aggregated_moe_logits = torch.cat(all_moe_logits, dim=0)

    total_samples = sum(sample_counts)
    print("\n合并完成:")
    print(f"  总样本数: {total_samples}")
    print(f"  质量分数形状: {aggregated_quality_logits.shape}")
    print(f"  MoE logits形状: {aggregated_moe_logits.shape}")
    print(f"  包含数据集: {', '.join(dataset_names)}")
    print()

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


def analyze_single_dataset(dataset_name, router_data, args):
    """分析单个数据集"""
    print(f"\n{'=' * 80}")
    print(f"分析数据集: {dataset_name}")
    print(f"{'=' * 80}")

    print(f"总样本数: {len(router_data['sample_ids'])}")
    print(f"模型层数: {router_data['quality_score'].shape[1]}")
    print(f"专家数量: {router_data['moe_logits'].shape[2]}")
    print()

    # 1. 分析质量门
    quality_scores, quality_ranking = analyze_quality_gates(router_data)

    # 2. 分析MoE路由
    expert_usage, sample_entropy = analyze_moe_routing(router_data)

    # 3. 执行聚类和2D投影
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coords_2d, cluster_labels, cluster_info = perform_clustering_and_visualization(router_data, device=device, clustering_method=args.clustering_method)

    # 4. 创建简化可视化
    print("=" * 70)
    print(f"创建{dataset_name}数据集简化可视化...")
    print("=" * 70)

    save_path = None
    if dataset_name == "聚合数据集" and "source_datasets" in router_data:
        # 聚合数据集使用特殊命名
        safe_dataset_name = f"aggregated_{'_'.join(router_data['source_datasets'])}"
        safe_dataset_name = safe_dataset_name.replace("/", "_").replace("\\", "_")
    else:
        safe_dataset_name = dataset_name.replace("/", "_").replace("\\", "_")
    save_path = os.path.join(args.output_dir, f"simplified_analysis_{safe_dataset_name}.png")

    create_simplified_visualization(router_data, coords_2d, cluster_labels, cluster_info, quality_scores, expert_usage, save_path)

    plt.show()

    # 5. 生成分析报告
    print("=" * 70)
    print(f"{dataset_name} - 分析总结")
    print("=" * 70)

    n_total = len(router_data["sample_ids"])
    n_clusters = len(np.unique(cluster_labels[cluster_labels >= 0]))

    print(f"数据集分析报告 - {dataset_name}")
    print(f"{'=' * 50}")
    print(f"总样本数量: {n_total}")
    print(f"聚类方法: {cluster_info.get('method', 'Unknown')}")
    print(f"有效簇数: {n_clusters}")
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

    print("聚类分析:")
    print(f"  聚类方法: {cluster_info.get('method', 'Unknown')}")
    print(f"  有效簇数: {n_clusters}")
    if "optimal_k" in cluster_info:
        print(f"  最优k值: {cluster_info['optimal_k']}")
    print()

    return {
        "dataset_name": dataset_name,
        "quality_scores": quality_scores,
        "expert_usage": expert_usage,
        "sample_entropy": sample_entropy,
        "cluster_info": cluster_info,
        "n_total": n_total,
        "n_clusters": n_clusters,
    }


def main():
    parser = argparse.ArgumentParser(description="Select-MoE简化数据选择分析")
    parser.add_argument("router_data_path", help="路由数据文件路径(.pt格式)或包含多个router_data文件的目录")
    parser.add_argument("--clustering-method", choices=["kmeans", "hdbscan"], default="kmeans", help="聚类方法 (默认: kmeans)")
    parser.add_argument("--output-dir", default="./outputs/visual_figs/simplified_analysis", help="图片保存目录")
    parser.add_argument("--dataset-filter", help="只分析匹配此模式的数据集 (支持通配符)")
    parser.add_argument("--aggregate-datasets", action="store_true", help="将多个数据集聚合为一个整体进行分析，而不是分别分析每个数据集")

    args = parser.parse_args()

    # 创建输出目录
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

    print("=" * 70)
    print("Select-MoE简化分析完成！")
    print()
    print("关键洞察:")
    print("1. 质量分数反映了样本的整体数据质量")
    print("2. 各层质量分数分布展示了模型层级间的质量判断一致性")
    print("3. 聚类结果显示了MoE路由模式的数据分组特征")
    print("4. GPU加速的聚类算法有效处理大规模数据集")
    print("=" * 70)


if __name__ == "__main__":
    main()
