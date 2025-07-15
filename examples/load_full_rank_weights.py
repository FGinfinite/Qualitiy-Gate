#!/usr/bin/env python3
"""
示例：加载全秩微调权重到预训练模型

使用方法:
    python examples/load_full_rank_weights.py --checkpoint_path outputs/stage_1_pretrain/xxx/full_rank_weights.pt
"""
import os
import sys

# 获取当前脚本所在目录的父目录（即项目根目录）
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)  # 将项目根目录添加到sys.path的最前面

import argparse
import torch
from src.models.select_moe import SelectMoeForCausalLM, register_select_moe
from src.training.full_rank_finetuning import load_full_rank_weights

def main():
    parser = argparse.ArgumentParser(description="加载全秩微调权重到预训练模型")
    parser.add_argument("--checkpoint_path", type=str, required=True, 
                       help="全秩微调权重的检查点路径")
    parser.add_argument("--model_path", type=str, 
                       default="converted_models/select_moe_converted_OLMoE-1B-7B-0125",
                       help="预训练模型路径")
    
    args = parser.parse_args()
    
    # 注册 Select-MoE 模型
    register_select_moe()
    
    # 加载预训练模型
    print(f"正在加载预训练模型: {args.model_path}")
    model = SelectMoeForCausalLM.from_pretrained(
        args.model_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    )
    print("预训练模型加载完成")
    
    # 加载全秩微调权重
    print(f"正在加载全秩微调权重: {args.checkpoint_path}")
    load_full_rank_weights(model, args.checkpoint_path)
    print("全秩微调权重加载完成")
    
    # 模型现在已经包含了微调后的权重，可以用于推理
    print("模型已准备就绪，可以用于推理")

if __name__ == "__main__":
    main()