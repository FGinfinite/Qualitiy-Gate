#!/usr/bin/env python3
"""
示例：从Trainer checkpoint加载模型权重

使用方法:
    # 方法1: 直接加载完整模型
    python examples/load_trainer_checkpoint.py --checkpoint_dir outputs/stage_1_pretrain/2025-07-16/01-10-13/checkpoint-28 --method full_model
    
    # 方法2: 加载到预训练模型（所有权重）
    python examples/load_trainer_checkpoint.py --checkpoint_dir outputs/stage_1_pretrain/2025-07-16/01-10-13/checkpoint-28 --method full_weights
    
    # 方法3: 只加载目标模块权重
    python examples/load_trainer_checkpoint.py --checkpoint_dir outputs/stage_1_pretrain/2025-07-16/01-10-13/checkpoint-28 --method target_only --target_modules "mlp.gate"
"""

import argparse
import torch
from src.models.select_moe import SelectMoeForCausalLM, register_select_moe
from src.training.full_rank_finetuning import (
    load_trainer_checkpoint_full_rank,
    load_trainer_checkpoint_full_model
)

def main():
    parser = argparse.ArgumentParser(description="从Trainer checkpoint加载模型权重")
    parser.add_argument("--checkpoint_dir", type=str, required=True, 
                       help="Trainer checkpoint目录路径")
    parser.add_argument("--method", type=str, 
                       choices=["full_model", "full_weights", "target_only"],
                       default="full_model",
                       help="加载方法")
    parser.add_argument("--model_path", type=str, 
                       default="converted_models/select_moe_converted_OLMoE-1B-7B-0125",
                       help="预训练模型路径（方法2和3需要）")
    parser.add_argument("--target_modules", type=str, nargs="+",
                       default=["mlp.gate"],
                       help="目标模块名称（方法3需要）")
    
    args = parser.parse_args()
    
    # 注册 Select-MoE 模型
    register_select_moe()
    
    if args.method == "full_model":
        # 方法1：直接加载完整模型
        print("=== 方法1：直接加载完整模型 ===")
        model = load_trainer_checkpoint_full_model(args.checkpoint_dir)
        print("模型加载完成，可直接用于推理")
        
    elif args.method == "full_weights":
        # 方法2：加载到预训练模型（所有权重）
        print("=== 方法2：加载所有权重到预训练模型 ===")
        print(f"正在加载预训练模型: {args.model_path}")
        model = SelectMoeForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )
        print("预训练模型加载完成")
        
        # 加载checkpoint权重
        load_trainer_checkpoint_full_rank(model, args.checkpoint_dir)
        print("所有权重加载完成")
        
    elif args.method == "target_only":
        # 方法3：只加载目标模块权重
        print("=== 方法3：只加载目标模块权重 ===")
        print(f"正在加载预训练模型: {args.model_path}")
        model = SelectMoeForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )
        print("预训练模型加载完成")
        
        # 只加载目标模块权重
        print(f"目标模块: {args.target_modules}")
        load_trainer_checkpoint_full_rank(
            model, 
            args.checkpoint_dir, 
            target_patterns=args.target_modules
        )
        print("目标模块权重加载完成")
    
    print("\n=== 模型信息 ===")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数数量: {total_params:,}")
    print("模型已准备就绪，可用于推理或进一步训练")

if __name__ == "__main__":
    main()