#!/usr/bin/env python3
"""
训练诊断脚本

用于诊断阶段1训练中梯度为0的问题。
检查：
1. 哪些参数被设置为可训练
2. 这些参数在前向传播中是否被使用
3. 梯度是否正确计算
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from transformers import AutoTokenizer

from src.models.quality_gate_model import QualityGateForCausalLM
from src.training.full_rank_finetuning import setup_full_rank_training

# 质量门控参数模式
QUALITY_GATE_PATTERNS = [
    "quality_gate",
    ".gate.weight",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="converted_models/quality_gate_Qwen3-1.7B-Base")
    args = parser.parse_args()

    print("=" * 80)
    print("训练诊断工具")
    print("=" * 80)

    # 加载模型
    print(f"\n正在加载模型: {args.model_path}")
    model = QualityGateForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float32, device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("✓ 模型加载完成")

    # 设置full_rank训练
    print("\n正在设置full_rank训练...")
    setup_full_rank_training(model, QUALITY_GATE_PATTERNS, mode="parameter")

    # 列出所有可训练参数
    print("\n可训练参数列表:")
    print("-" * 80)
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
            print(f"  {name}: {param.shape} ({param.numel():,} params)")

    print(f"\n总可训练参数: {len(trainable_params)}")

    # 创建一个简单的输入
    print("\n正在测试前向传播...")
    test_input = "The capital of France is"
    inputs = tokenizer(test_input, return_tensors="pt")

    # 前向传播
    model.train()
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss

    print(f"✓ 前向传播成功，Loss: {loss.item():.4f}")

    # 反向传播
    print("\n正在测试反向传播...")
    loss.backward()

    # 检查梯度
    print("\n检查梯度:")
    print("-" * 80)
    has_gradient = False
    zero_gradient_params = []
    nonzero_gradient_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                print(f"  ⚠️  {name}: 梯度为 None!")
            elif param.grad.abs().sum().item() == 0:
                zero_gradient_params.append(name)
            else:
                nonzero_gradient_params.append(name)
                has_gradient = True
                grad_norm = param.grad.norm().item()
                print(f"  ✓ {name}: 梯度范数 = {grad_norm:.6e}")

    if zero_gradient_params:
        print(f"\n梯度为0的参数 ({len(zero_gradient_params)}):")
        for name in zero_gradient_params[:10]:
            print(f"  - {name}")
        if len(zero_gradient_params) > 10:
            print(f"  ... (还有 {len(zero_gradient_params) - 10} 个)")

    # 总结
    print("\n" + "=" * 80)
    print("诊断结果")
    print("=" * 80)

    if has_gradient:
        print(f"✅ 成功：发现 {len(nonzero_gradient_params)} 个参数有非零梯度")
        print("训练应该可以正常进行")
    else:
        print("❌ 问题：所有可训练参数的梯度都为0或None")
        print("\n可能的原因：")
        print("  1. 这些参数在前向传播中没有被使用")
        print("  2. 损失计算有问题")
        print("  3. 模型架构问题（detach或stop_gradient）")

        # 详细检查模型架构
        print("\n正在检查quality_gate层...")
        for name, module in model.named_modules():
            if "quality_gate" in name:
                print(f"\n  模块: {name}")
                print(f"    类型: {type(module).__name__}")
                if hasattr(module, "gate"):
                    gate = module.gate
                    print(f"    gate参数: {gate.weight.shape}")
                    print(f"    requires_grad: {gate.weight.requires_grad}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
