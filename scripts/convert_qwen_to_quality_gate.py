#!/usr/bin/env python3
# Copyright 2024 Quality-Gate Project. All rights reserved.
# Licensed under the Apache License, Version 2.0

"""
将 Qwen3 基座模型转换为 Quality-Gate 模型

功能：
1. 加载 Qwen3 预训练模型（仅支持 Qwen3，不支持 Qwen2/Qwen2.5）
2. 创建 Quality-Gate 配置并添加质量门控层
3. 复制 Qwen3 基座权重到 Quality-Gate 模型
4. 保存为标准 HuggingFace 模型格式

注意：此脚本专门为 Qwen3 设计，与 Quality-Gate 模型架构完全匹配。
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 获取项目根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.quality_gate_model import (
    QualityGateConfig,
    QualityGateForCausalLM,
    register_quality_gate,
)


def set_random_seed(seed):
    """设置所有随机数生成器的种子以确保可重现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def convert_and_save_model(
    model_name="Qwen/Qwen3-1.7B",
    save_path=None,
    device="cpu",
    quality_gate_init_mean=0.0,
    quality_gate_init_std=0.02,
    quality_loss_weight=1.0,
    quality_loss_type="sigmoid",
    seed=42,
):
    """
    将 Qwen3 预训练模型转换为 Quality-Gate 模型并保存

    Args:
        model_name: Qwen3 模型名称或本地路径（仅支持 Qwen3）
        save_path: 本地保存路径
        device: 使用的设备
        quality_gate_init_mean: 质量门控初始化均值
        quality_gate_init_std: 质量门控初始化标准差
        quality_loss_weight: 质量损失权重
        quality_loss_type: 质量损失类型 ("sigmoid", "linear", "beta")
        seed: 随机种子

    注意：此函数仅支持 Qwen3 模型，不支持 Qwen2/Qwen2.5
    """
    # 设置随机种子
    set_random_seed(seed)

    print("=" * 80)
    print("Qwen3 → Quality-Gate 模型转换")
    print(f"源模型: {model_name}")
    print(f"设备: {device}")
    print(f"随机种子: {seed}")
    print("=" * 80)

    # 注册 Quality-Gate 模型
    register_quality_gate()
    print("✓ Quality-Gate 模型已注册")

    try:
        # 1. 加载原始 Qwen3 模型
        print(f"\n1. 加载 Qwen3 模型: {model_name}")
        print("   这可能需要几分钟...")

        original_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device if device != "cpu" else None,
        )
        original_config = original_model.config

        # 验证模型类型
        model_type = type(original_model).__name__
        if "Qwen2" in model_type:
            raise ValueError(f"❌ 检测到 Qwen2/Qwen2.5 模型 ({model_type})。\n   本脚本仅支持 Qwen3 模型。\n   请使用 Qwen3 系列模型（如 Qwen/Qwen3-1.7B）。")

        print("✓ Qwen3 模型加载成功!")
        print(f"  - 模型类型: {model_type}")
        print(f"  - 层数: {original_config.num_hidden_layers}")
        print(f"  - 隐藏层大小: {original_config.hidden_size}")
        print(f"  - 词汇表大小: {original_config.vocab_size}")
        print(f"  - 总参数量: {sum(p.numel() for p in original_model.parameters()):,}")

        # 2. 创建 Quality-Gate 配置
        print("\n2. 创建 Quality-Gate 配置...")
        quality_gate_config_dict = original_config.to_dict()
        quality_gate_config_dict.update(
            {
                "model_type": "quality_gate",
                "quality_gate_init_mean": quality_gate_init_mean,
                "quality_gate_init_std": quality_gate_init_std,
                "quality_loss_weight": quality_loss_weight,
                "quality_loss_type": quality_loss_type,
                # 默认损失参数
                "sample_wise_averaging": True,
                "full_sequence_prediction": True,
                "mask_special_tokens": True,
            }
        )

        quality_gate_config = QualityGateConfig(**quality_gate_config_dict)

        print("✓ Quality-Gate 配置创建成功!")
        print(f"  - 质量门控初始化: mean={quality_gate_init_mean}, std={quality_gate_init_std}")
        print(f"  - 质量损失权重: {quality_loss_weight}")
        print(f"  - 质量损失类型: {quality_loss_type}")

        # 3. 转换模型
        print("\n3. 转换为 Quality-Gate 架构...")

        # 创建新的 Quality-Gate 模型
        quality_gate_model = QualityGateForCausalLM(quality_gate_config)

        # 复制原始模型的权重
        original_state_dict = original_model.state_dict()
        quality_gate_state_dict = quality_gate_model.state_dict()

        copied_count = 0
        new_params = []
        missing_params = []

        for key in quality_gate_state_dict:
            if key in original_state_dict:
                # 直接复制 Qwen3 权重
                quality_gate_state_dict[key].copy_(original_state_dict[key])
                copied_count += 1
            elif "quality_gate" in key:
                # 质量门控参数（新增的）
                new_params.append(key)
            else:
                # 不应该发生：Qwen3 和 Quality-Gate 架构应该完全匹配
                missing_params.append(key)

        print(f"✓ 复制 Qwen3 权重: {copied_count}/{len(original_state_dict)}")
        print(f"✓ 新增质量门控参数: {len(new_params)} 个")

        # 验证架构匹配
        if missing_params:
            print(f"⚠️  警告: {len(missing_params)} 个参数未找到匹配")
            for param in missing_params[:5]:
                print(f"    - {param}")
            if len(missing_params) > 5:
                print(f"    ... 还有 {len(missing_params) - 5} 个")
            print("   这可能表示模型架构不匹配，请确认使用的是 Qwen3 模型。")

        # 显示质量门控参数示例
        if new_params:
            print(f"  质量门控层分布: 每层 1 个质量门控")
            print(f"  示例参数: {new_params[0]}")

        print("✓ Quality-Gate 模型创建完成!")

        # 移动到设备
        if device != "cpu":
            quality_gate_model = quality_gate_model.to(device)

        # 4. 验证转换
        print("\n4. 验证转换...")

        # 测试前向传播
        batch_size, seq_len = 1, 8
        test_input = torch.randint(0, quality_gate_config.vocab_size, (batch_size, seq_len))
        if device != "cpu":
            test_input = test_input.to(device)

        with torch.no_grad():
            outputs = quality_gate_model(test_input, output_router_logits=False)

        print("✓ 前向传播测试成功!")
        print(f"  - 输出形状: {outputs.logits.shape}")

        # 测试质量门控输出
        with torch.no_grad():
            outputs_with_router = quality_gate_model(test_input, output_router_logits=True)

        print("✓ 质量门控输出测试成功!")
        if hasattr(outputs_with_router, "router_logits") and outputs_with_router.router_logits:
            print(f"  - 质量门控层数: {len(outputs_with_router.router_logits)}")
            first_layer_router = outputs_with_router.router_logits[0]
            print(f"  - 质量门控输出形状: {first_layer_router.shape} [batch, seq_len, 1]")
        else:
            raise ValueError("❌ 质量门控输出验证失败：未找到 router_logits")

        # 5. 保存模型
        print("\n5. 保存转换后的模型...")

        if save_path is None:
            save_path = f"./converted_models/quality_gate_{model_name.split('/')[-1]}"

        os.makedirs(save_path, exist_ok=True)

        # 保存模型和配置
        quality_gate_model.save_pretrained(save_path)
        print(f"✓ 模型已保存到: {save_path}")

        # 保存分词器
        print("保存分词器...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(save_path)
        print("✓ 分词器已保存")

        # 6. 测试加载保存的模型
        print("\n6. 测试从保存路径加载...")

        test_loaded_model = QualityGateForCausalLM.from_pretrained(save_path)
        print("✓ 模型加载成功!")
        print(f"  - 模型类型: {type(test_loaded_model).__name__}")
        print(f"  - 配置类型: {type(test_loaded_model.config).__name__}")

        # 快速功能测试
        if device != "cpu":
            test_loaded_model = test_loaded_model.to(device)

        with torch.no_grad():
            test_outputs = test_loaded_model(test_input, output_router_logits=True)

        print("✓ 加载模型前向传播测试成功!")
        print(f"  - 输出形状: {test_outputs.logits.shape}")

        # 7. 转换完成
        print("\n" + "=" * 80)
        print("🎉 Qwen3 → Quality-Gate 转换完成!")
        print("=" * 80)

        print(f"\n📁 保存路径: {os.path.abspath(save_path)}")

        print("\n📝 使用方法:")
        print("   1. 更新配置文件 (configs/stage_1_warmup.yaml):")
        print(f"      selector_model.path: '{save_path}'")
        print("")
        print("   2. 运行训练:")
        print("      CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_stage_1.sh")
        print("")
        print("   3. 加载模型代码:")
        print("      from src.models.quality_gate_model import QualityGateForCausalLM, register_quality_gate")
        print("      register_quality_gate()")
        print(f"      model = QualityGateForCausalLM.from_pretrained('{save_path}')")

        print("\n✅ 转换内容:")
        print(f"   ✓ Qwen3 基座权重: {copied_count} 个参数")
        print(f"   ✓ 质量门控层: {len(new_params)} 个参数")
        print(f"   ✓ 总层数: {quality_gate_config.num_hidden_layers} 层")
        print("   ✓ 可直接用于训练和推理")

        return save_path

    except Exception as e:
        print(f"\n❌ 转换失败: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    """主转换脚本 - 仅支持 Qwen3"""
    parser = argparse.ArgumentParser(
        description="将 Qwen3 预训练模型转换为 Quality-Gate 格式",
        epilog="注意：此脚本仅支持 Qwen3 模型，不支持 Qwen2/Qwen2.5",
    )

    # 模型参数
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help="Qwen3 模型名称或路径（仅支持 Qwen3）",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="本地保存路径",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="使用的设备 (cpu, cuda, cuda:0 等)",
    )

    # Quality-Gate 配置
    parser.add_argument(
        "--quality-gate-init-mean",
        type=float,
        default=0.0,
        help="质量门控初始化均值",
    )
    parser.add_argument(
        "--quality-gate-init-std",
        type=float,
        default=0.02,
        help="质量门控初始化标准差",
    )
    parser.add_argument(
        "--quality-loss-weight",
        type=float,
        default=1.0,
        help="质量损失权重",
    )
    parser.add_argument(
        "--quality-loss-type",
        type=str,
        default="sigmoid",
        choices=["sigmoid", "linear", "beta"],
        help="质量损失类型",
    )

    # 随机种子
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认: 42）",
    )

    args = parser.parse_args()

    # 验证设备
    if args.device != "cpu" and not torch.cuda.is_available():
        print("⚠️  CUDA 不可用，回退到 CPU")
        args.device = "cpu"
    elif args.device.startswith("cuda:"):
        device_id = int(args.device.split(":")[1])
        if device_id >= torch.cuda.device_count():
            print(f"⚠️  设备 {args.device} 不可用，回退到 CPU")
            args.device = "cpu"

    print(f"使用设备: {args.device}")
    if args.device != "cpu":
        print(f"GPU 内存: {torch.cuda.get_device_properties(args.device).total_memory / 1024**3:.1f} GB")

    # 转换并保存
    result_path = convert_and_save_model(
        model_name=args.model,
        save_path=args.save_path,
        device=args.device,
        quality_gate_init_mean=args.quality_gate_init_mean,
        quality_gate_init_std=args.quality_gate_init_std,
        quality_loss_weight=args.quality_loss_weight,
        quality_loss_type=args.quality_loss_type,
        seed=args.seed,
    )

    if result_path:
        print(f"\n✅ 成功! 模型已保存到: {result_path}")
        exit(0)
    else:
        print("\n❌ 转换失败!")
        exit(1)


if __name__ == "__main__":
    main()
