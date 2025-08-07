#!/usr/bin/env python3
"""
Select-MoE数据选择综合可视化分析脚本

本脚本提供了对Select-MoE数据选择过程的全面分析和可视化，包括：
1. 质量门分析：展示样本的质量分数分布
2. MoE路由分析：分析专家选择模式  
3. Wasserstein距离计算：样本间相似性度量
4. FPS算法应用：多样性选择过程
5. 选择结果对比：质量选择 vs 多样性选择

这个脚本帮助理解整个Select-MoE数据选择管道的工作原理。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from scipy.stats import wasserstein_distance
import pandas as pd
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler

from src.stages.selection import (
    load_router_data, 
    calculate_quality_score_from_gates,
    compute_batch_wasserstein_distance_gpu,
    farthest_point_sampling
)
import argparse


plt.rcParams['font.sans-serif'] = ['Maple Mono NF CN'] 

def analyze_quality_gates(router_data, top_k=10):
    """分析质量门的输出分布"""
    quality_logits = router_data['quality_logits']  # [N, L, 2]
    sample_ids = router_data['sample_ids']
    
    print("=" * 70)
    print("质量门分析")
    print("=" * 70)
    
    # 计算每个样本的质量分数（good概率的平均值）
    quality_probs = torch.softmax(quality_logits.float(), dim=-1)  # [N, L, 2]
    good_probs = quality_probs[:, :, 0]  # [N, L] - good类别的概率
    
    # 每个样本在所有层的平均good概率作为质量分数
    sample_quality_scores = good_probs.mean(dim=1).numpy()  # [N]
    
    print(f"质量分数统计:")
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
        idx = sorted_indices[-(i+1)]
        print(f"  {sample_ids[idx]}: {sample_quality_scores[idx]:.4f}")
    print()
    
    return sample_quality_scores, sorted_indices


def analyze_moe_routing(router_data):
    """分析MoE路由模式"""
    moe_logits = router_data['moe_logits']  # [N, L, E]
    
    print("=" * 70)
    print("MoE路由分析")
    print("=" * 70)
    
    # 转换为概率分布
    moe_probs = torch.softmax(moe_logits.float(), dim=-1)  # [N, L, E]
    
    # 计算专家使用统计
    expert_usage = moe_probs.mean(dim=(0, 1)).numpy()  # [E] - 每个专家的平均使用率
    
    print(f"专家使用统计:")
    print(f"  专家数量: {len(expert_usage)}")
    print(f"  平均使用率: {expert_usage.mean():.4f}")
    print(f"  使用率标准差: {expert_usage.std():.4f}")
    print(f"  负载平衡度: {1 - expert_usage.std()*len(expert_usage):.4f}")
    print()
    
    # 分析路由多样性
    # 计算每个样本的路由熵（跨专家维度）
    entropy_per_layer = -torch.sum(moe_probs * torch.log(moe_probs + 1e-8), dim=-1)  # [N, L]
    sample_avg_entropy = entropy_per_layer.mean(dim=1).numpy()  # [N]
    
    print(f"路由多样性统计（熵）:")
    print(f"  平均熵: {sample_avg_entropy.mean():.4f}")
    print(f"  熵标准差: {sample_avg_entropy.std():.4f}")
    print(f"  最大可能熵: {np.log(len(expert_usage)):.4f}")
    print()
    
    return expert_usage, sample_avg_entropy


def compute_comprehensive_distances(router_data, max_samples=50):
    """计算样本间的综合距离矩阵"""
    moe_logits = router_data['moe_logits']  # [N, L, E]
    
    # 限制样本数量
    n_samples = min(len(moe_logits), max_samples)
    selected_indices = np.linspace(0, len(moe_logits)-1, n_samples, dtype=int)
    
    moe_subset = moe_logits[selected_indices]  # [n_samples, L, E]
    distance_matrix = np.zeros((n_samples, n_samples))
    
    print("=" * 70)
    print("Wasserstein距离计算")
    print("=" * 70)
    print(f"计算{n_samples}个样本的距离矩阵...")
    
    # 尝试使用GPU加速计算
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if device.type == 'cuda':
        print("使用GPU加速计算...")
        # 转换为概率并移至GPU
        moe_probs = torch.softmax(moe_subset.float(), dim=-1).to(device)
        
        # 批量计算距离矩阵
        gpu_distance_matrix = compute_batch_wasserstein_distance_gpu(moe_probs, moe_probs)
        distance_matrix = gpu_distance_matrix.cpu().numpy()
    else:
        print("使用CPU计算...")
        # CPU计算
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                # 转换为概率分布
                probs_i = torch.softmax(moe_subset[i].float(), dim=-1)
                probs_j = torch.softmax(moe_subset[j].float(), dim=-1)
                
                # 计算总Wasserstein距离
                total_dist = 0
                for layer_idx in range(probs_i.shape[0]):
                    prob1 = probs_i[layer_idx].numpy()
                    prob2 = probs_j[layer_idx].numpy()
                    expert_indices = np.arange(len(prob1))
                    layer_dist = wasserstein_distance(expert_indices, expert_indices, prob1, prob2)
                    total_dist += layer_dist
                
                distance_matrix[i, j] = total_dist
                distance_matrix[j, i] = total_dist
    
    print(f"✓ 距离矩阵计算完成")
    print(f"  平均距离: {distance_matrix[distance_matrix > 0].mean():.4f}")
    print(f"  距离标准差: {distance_matrix[distance_matrix > 0].std():.4f}")
    print()
    
    return distance_matrix, selected_indices


def perform_selection_comparison(router_data, distance_matrix, demo_indices, 
                               quality_scores, selection_ratio=0.2):
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
    
    print(f"1. 质量选择（选择质量分数最高的样本）:")
    for i, global_idx in enumerate(quality_selected_global):
        local_idx = quality_selected_local[i]
        print(f"   样本{global_idx} (本地{local_idx}): 质量分数 {quality_subset[local_idx]:.4f}")
    print()
    
    # 2. 多样性选择：使用FPS算法
    fps_selected_local = farthest_point_sampling(distance_matrix, n_select, seed=42)
    fps_selected_global = demo_indices[fps_selected_local]
    
    print(f"2. 多样性选择（FPS算法）:")
    for i, local_idx in enumerate(fps_selected_local):
        global_idx = fps_selected_global[i]
        print(f"   样本{global_idx} (本地{local_idx}): 质量分数 {quality_subset[local_idx]:.4f}")
    print()
    
    # 3. 随机选择作为基准
    np.random.seed(42)
    random_selected_local = np.random.choice(n_total, n_select, replace=False)
    random_selected_global = demo_indices[random_selected_local]
    
    print(f"3. 随机选择（基准）:")
    for i, local_idx in enumerate(random_selected_local):
        global_idx = random_selected_global[i]
        print(f"   样本{global_idx} (本地{local_idx}): 质量分数 {quality_subset[local_idx]:.4f}")
    print()
    
    # 计算选择结果的统计指标
    strategies = {
        'Quality': (quality_selected_local, quality_selected_global),
        'Diversity': (fps_selected_local, fps_selected_global), 
        'Random': (random_selected_local, random_selected_global)
    }
    
    print("选择结果统计:")
    print("-" * 50)
    
    for strategy_name, (local_indices, global_indices) in strategies.items():
        # 质量分数统计
        selected_quality = quality_subset[local_indices]
        avg_quality = selected_quality.mean()
        
        # 多样性统计（选中样本间的平均距离）
        selected_distances = []
        for i in range(len(local_indices)):
            for j in range(i+1, len(local_indices)):
                selected_distances.append(distance_matrix[local_indices[i], local_indices[j]])
        
        avg_diversity = np.mean(selected_distances) if selected_distances else 0
        
        print(f"{strategy_name:>10}: 平均质量={avg_quality:.4f}, 平均距离={avg_diversity:.4f}")
    
    print()
    
    return strategies


def create_comprehensive_visualization(router_data, distance_matrix, demo_indices,
                                    quality_scores, expert_usage, sample_entropy,
                                    strategies, save_path=None):
    """创建综合可视化图表"""
    sample_ids = router_data['sample_ids']
    moe_logits = router_data['moe_logits']
    quality_logits = router_data['quality_logits']
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. 质量分数分布 (2x4 grid, position 1-2)
    ax1 = plt.subplot(4, 4, 1)
    ax1.hist(quality_scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('质量分数')
    ax1.set_ylabel('样本数量')
    ax1.set_title('样本质量分数分布')
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(4, 4, 2)
    quality_by_layer = torch.softmax(quality_logits.float(), dim=-1)[:, :, 0]  # [N, L]
    layer_avg_quality = quality_by_layer.mean(dim=0).numpy()  # [L]
    ax2.plot(range(len(layer_avg_quality)), layer_avg_quality, 'o-', linewidth=2)
    ax2.set_xlabel('层索引')
    ax2.set_ylabel('平均质量分数')
    ax2.set_title('各层质量分数分布')
    ax2.grid(True, alpha=0.3)
    
    # 2. MoE专家使用分析 (position 3-4)
    ax3 = plt.subplot(4, 4, 3)
    ax3.bar(range(len(expert_usage)), expert_usage, alpha=0.7, color='lightgreen')
    ax3.set_xlabel('专家索引')
    ax3.set_ylabel('平均使用率')
    ax3.set_title('专家使用率分布')
    ax3.grid(True, alpha=0.3)
    
    ax4 = plt.subplot(4, 4, 4)
    ax4.hist(sample_entropy, bins=30, alpha=0.7, color='orange', edgecolor='black')
    ax4.set_xlabel('路由熵')
    ax4.set_ylabel('样本数量')
    ax4.set_title('样本路由多样性分布')
    ax4.grid(True, alpha=0.3)
    
    # 3. 距离矩阵热力图 (position 5-8, span 2x2)
    ax5 = plt.subplot(4, 4, (5, 8))
    # 只显示距离矩阵的上三角部分
    mask = np.triu(np.ones_like(distance_matrix, dtype=bool), k=1)
    sns.heatmap(distance_matrix, mask=~mask, cmap='viridis', square=True, 
                cbar_kws={'label': 'Wasserstein距离'}, ax=ax5)
    ax5.set_title(f'样本间Wasserstein距离矩阵\n({len(demo_indices)}个样本)')
    
    # 4. 2D投影和选择结果 (position 9-12, span 2x2)
    ax6 = plt.subplot(4, 4, (9, 12))
    
    # 使用MDS进行2D投影
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, 
              max_iter=3000, eps=1e-9)
    coords_2d = mds.fit_transform(distance_matrix)
    
    # 绘制所有点
    ax6.scatter(coords_2d[:, 0], coords_2d[:, 1], c='lightgray', s=50, alpha=0.6, 
               label='未选择')
    
    # 绘制不同策略选择的点
    colors = {'Quality': 'red', 'Diversity': 'blue', 'Random': 'green'}
    markers = {'Quality': 's', 'Diversity': '*', 'Random': '^'}
    
    for strategy_name, (local_indices, _) in strategies.items():
        selected_coords = coords_2d[local_indices]
        ax6.scatter(selected_coords[:, 0], selected_coords[:, 1], 
                   c=colors[strategy_name], marker=markers[strategy_name], 
                   s=120, label=f'{strategy_name}选择', alpha=0.8, edgecolors='black')
    
    ax6.set_xlabel('第一主成分')
    ax6.set_ylabel('第二主成分')
    ax6.set_title('样本2D投影与选择结果对比')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 5. 选择策略对比表格 (position 13-16)
    ax7 = plt.subplot(4, 4, (13, 16))
    ax7.axis('off')
    
    # 计算统计数据
    stats_data = []
    for strategy_name, (local_indices, global_indices) in strategies.items():
        quality_subset = quality_scores[demo_indices]
        selected_quality = quality_subset[local_indices]
        
        # 多样性统计
        selected_distances = []
        for i in range(len(local_indices)):
            for j in range(i+1, len(local_indices)):
                selected_distances.append(distance_matrix[local_indices[i], local_indices[j]])
        
        avg_diversity = np.mean(selected_distances) if selected_distances else 0
        min_diversity = min(selected_distances) if selected_distances else 0
        max_diversity = max(selected_distances) if selected_distances else 0
        
        stats_data.append({
            'Strategy': strategy_name,
            'Avg Quality': f"{selected_quality.mean():.4f}",
            'Quality Std': f"{selected_quality.std():.4f}",
            'Avg Distance': f"{avg_diversity:.4f}",
            'Min Distance': f"{min_diversity:.4f}",
            'Max Distance': f"{max_diversity:.4f}"
        })
    
    # 创建表格
    df_stats = pd.DataFrame(stats_data)
    table = ax7.table(cellText=df_stats.values, colLabels=df_stats.columns,
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # 设置表格样式
    for i in range(len(df_stats.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(df_stats) + 1):
        for j in range(len(df_stats.columns)):
            if i == 1:  # Quality row
                table[(i, j)].set_facecolor('#FFCDD2')
            elif i == 2:  # Diversity row  
                table[(i, j)].set_facecolor('#BBDEFB')
            else:  # Random row
                table[(i, j)].set_facecolor('#C8E6C9')
    
    ax7.set_title('选择策略性能对比', fontsize=14, pad=20)
    
    plt.suptitle(f'Select-MoE数据选择综合分析\n数据集: {router_data["dataset_name"]}, '
                f'总样本: {len(sample_ids)}, 分析样本: {len(demo_indices)}', 
                fontsize=16, y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 已保存综合分析图: {save_path}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Select-MoE综合数据选择分析')
    parser.add_argument('router_data_path', help='路由数据文件路径(.pt格式)')
    parser.add_argument('--max-samples', type=int, default=40, help='分析的最大样本数 (默认: 40)')
    parser.add_argument('--selection-ratio', type=float, default=0.2, help='选择比例 (默认: 0.2)')
    parser.add_argument('--save-plots', action='store_true', help='保存图片到文件')
    parser.add_argument('--output-dir', default='./outputs/visual_figs/comprehensive_analysis', help='图片保存目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    if args.save_plots:
        os.makedirs(args.output_dir, exist_ok=True)
    
    print("Select-MoE数据选择综合分析")
    print("=" * 70)
    
    # 1. 加载数据
    print(f"加载路由数据: {args.router_data_path}")
    router_data = load_router_data(args.router_data_path)
    
    print(f"数据集: {router_data['dataset_name']}")
    print(f"总样本数: {len(router_data['sample_ids'])}")
    print(f"模型层数: {router_data['moe_logits'].shape[1]}")
    print(f"专家数量: {router_data['moe_logits'].shape[2]}")
    print()
    
    # 2. 分析质量门
    quality_scores, quality_ranking = analyze_quality_gates(router_data)
    
    # 3. 分析MoE路由
    expert_usage, sample_entropy = analyze_moe_routing(router_data)
    
    # 4. 计算距离矩阵
    distance_matrix, demo_indices = compute_comprehensive_distances(
        router_data, args.max_samples)
    
    # 5. 对比选择策略
    strategies = perform_selection_comparison(
        router_data, distance_matrix, demo_indices, 
        quality_scores, args.selection_ratio)
    
    # 6. 创建综合可视化
    print("=" * 70)
    print("创建综合可视化...")
    print("=" * 70)
    
    save_path = os.path.join(args.output_dir, 'comprehensive_analysis.png') if args.save_plots else None
    fig = create_comprehensive_visualization(
        router_data, distance_matrix, demo_indices, 
        quality_scores, expert_usage, sample_entropy,
        strategies, save_path)
    
    plt.show()
    
    # 7. 生成分析报告
    print("=" * 70)
    print("分析总结")
    print("=" * 70)
    
    dataset_name = router_data['dataset_name']
    n_total = len(router_data['sample_ids'])
    n_analyzed = len(demo_indices)
    n_selected = len(strategies['Quality'][0])
    
    print(f"数据集分析报告 - {dataset_name}")
    print(f"{'='*50}")
    print(f"总样本数量: {n_total}")
    print(f"分析样本数量: {n_analyzed}")
    print(f"选择样本数量: {n_selected} (选择率: {args.selection_ratio:.1%})")
    print()
    
    print(f"质量分析:")
    print(f"  平均质量分数: {quality_scores.mean():.4f} ± {quality_scores.std():.4f}")
    print(f"  质量分数范围: [{quality_scores.min():.4f}, {quality_scores.max():.4f}]")
    print()
    
    print(f"路由分析:")
    print(f"  专家负载平衡度: {1 - expert_usage.std()*len(expert_usage):.4f}")
    print(f"  平均路由熵: {sample_entropy.mean():.4f} ± {sample_entropy.std():.4f}")
    print(f"  最大可能熵: {np.log(len(expert_usage)):.4f}")
    print()
    
    print(f"距离分析:")
    non_zero_distances = distance_matrix[distance_matrix > 0]
    print(f"  平均样本距离: {non_zero_distances.mean():.4f} ± {non_zero_distances.std():.4f}")
    print(f"  距离范围: [{non_zero_distances.min():.4f}, {non_zero_distances.max():.4f}]")
    print()
    
    print("选择策略效果:")
    for strategy_name, (local_indices, _) in strategies.items():
        quality_subset = quality_scores[demo_indices]
        selected_quality = quality_subset[local_indices]
        
        selected_distances = []
        for i in range(len(local_indices)):
            for j in range(i+1, len(local_indices)):
                selected_distances.append(distance_matrix[local_indices[i], local_indices[j]])
        
        avg_diversity = np.mean(selected_distances) if selected_distances else 0
        
        print(f"  {strategy_name:>10}: 质量={selected_quality.mean():.4f}, 多样性={avg_diversity:.4f}")
    
    print()
    print("关键洞察:")
    print("1. 质量选择优化数据质量，但可能选择相似样本")
    print("2. 多样性选择确保样本覆盖面，但质量可能不是最高")
    print("3. Select-MoE结合两者优势，先质量筛选再多样性选择")
    print("4. Wasserstein距离有效衡量MoE路由模式的差异")
    print("5. FPS算法确保选择样本的最大化多样性分布")
    
    print("=" * 70)
    print("分析完成！")


if __name__ == "__main__":
    main()