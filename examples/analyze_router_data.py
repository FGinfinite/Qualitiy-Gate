#!/usr/bin/env python3
"""
示例脚本：分析重构后的路由数据
演示如何加载和使用新的路由数据格式
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.stages.selection import load_router_data, get_sample_router_info


def analyze_router_data(router_data_path: str):
    """分析路由数据的示例函数"""

    # 1. 加载完整的路由数据
    router_data = load_router_data(router_data_path)

    # 2. 获取基本信息
    print(f"数据集名称: {router_data['dataset_name']}")
    print(f"样本总数: {router_data['num_samples']}")
    print(f"质量门logits形状: {router_data['quality_logits'].shape}")
    print(f"MoE路由logits形状: {router_data['moe_logits'].shape}")
    print(f"样本ID示例: {router_data['sample_ids'][:3] if router_data['sample_ids'] else '无'}")

    # 3. 分析第一个样本的详细信息
    if router_data["sample_ids"]:
        first_sample_id = router_data["sample_ids"][0]
        sample_info = get_sample_router_info(router_data, first_sample_id)

        print(f"\n=== 样本 {first_sample_id} 详细信息 ===")
        print(f"样本ID: {sample_info['sample_id']}")
        print(f"数据集: {sample_info['dataset_name']}")
        print(f"在数据集中的位置: {sample_info['position_in_dataset']}")
        print(f"质量门logits形状: {sample_info['quality_logits'].shape}")  # [num_layers, 2]
        print(f"MoE路由logits形状: {sample_info['moe_logits'].shape}")  # [num_layers, num_experts]

        # 4. 计算质量分数（示例）
        quality_logits = sample_info["quality_logits"]  # [num_layers, 2]
        # 质量门logits已经在序列维度进行了平均
        quality_probs = torch.softmax(quality_logits, dim=-1)  # [num_layers, 2]
        good_probs = quality_probs[:, 0]  # [num_layers] - 好数据概率

        # 按层计算质量分数
        overall_quality_score = good_probs.mean().item()

        print(f"整体质量分数: {overall_quality_score:.6f}")
        print(f"各层质量分数: {good_probs.tolist()[:5]}...")  # 前5层

        # 5. 分析MoE路由分布
        moe_probs = sample_info["moe_logits"]  # [num_layers, num_experts]
        expert_usage = moe_probs.mean(dim=0)  # [num_experts] - 各专家使用频率

        print(f"专家使用分布 (前5个): {expert_usage.tolist()[:5]}")
        print(f"最常用专家: {expert_usage.argmax().item()}")
        print(f"专家使用熵: {torch.distributions.Categorical(expert_usage).entropy().item():.4f}")


if __name__ == "__main__":
    # 示例用法
    if len(sys.argv) > 1:
        router_data_path = sys.argv[1]
        analyze_router_data(router_data_path)
    else:
        print("用法: python analyze_router_data.py <router_data_path>")
        print(
            "例如: python analyze_router_data.py outputs/stage_2_selection/2024-12-07/10-30-45/router_data/oasst1_router_data.pt"
        )
        print("\n也可以通过样本ID查询特定样本:")
        print('python -c "')
        print("from src.stages.selection import load_router_data, get_sample_router_info")
        print("data = load_router_data('path/to/oasst1_router_data.pt')")
        print("info = get_sample_router_info(data, 'oasst1_25460')")
        print("print(info)")
        print('"')
