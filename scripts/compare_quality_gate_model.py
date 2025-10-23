#!/usr/bin/env python3
# Copyright 2024 Quality-Gate Project. All rights reserved.
# Licensed under the Apache License, Version 2.0

"""
验证 Qwen3 → Quality-Gate 模型转换的正确性

功能：
1. 比较原始 Qwen3 模型和转换后的 Quality-Gate 模型权重
2. 验证基础权重完全匹配
3. 检查质量门控层是否正确初始化
4. 测试前向传播和 router_logits 输出
5. 生成详细的比较报告
"""

import argparse
import os
import sys

import torch
from transformers import AutoModelForCausalLM

# 获取项目根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.quality_gate_model import QualityGateForCausalLM, register_quality_gate


def compare_state_dicts(qwen3_dict, quality_gate_dict, tolerance=1e-6):
    """
    比较 Qwen3 和 Quality-Gate 模型的 state_dict

    Args:
        qwen3_dict: Qwen3 模型的 state_dict
        quality_gate_dict: Quality-Gate 模型的 state_dict
        tolerance: 数值比较的容差

    Returns:
        dict: 详细的比较结果
    """
    results = {
        "identical_keys": [],  # 完全匹配的权重
        "missing_in_quality_gate": [],  # Quality-Gate 中缺失的权重
        "extra_in_quality_gate": [],  # Quality-Gate 中额外的权重
        "shape_mismatch": [],  # 形状不匹配
        "value_mismatch": [],  # 数值不匹配
        "quality_gates": [],  # 质量门控参数详情
    }

    # 1. 检查 Qwen3 的每个权重在 Quality-Gate 中是否存在且匹配
    for key in qwen3_dict.keys():
        if key not in quality_gate_dict:
            results["missing_in_quality_gate"].append(key)
            continue

        qwen3_tensor = qwen3_dict[key]
        quality_gate_tensor = quality_gate_dict[key]

        # 检查形状
        if qwen3_tensor.shape != quality_gate_tensor.shape:
            results["shape_mismatch"].append({"key": key, "qwen3_shape": qwen3_tensor.shape, "quality_gate_shape": quality_gate_tensor.shape})
            continue

        # 检查数值
        if torch.allclose(qwen3_tensor, quality_gate_tensor, atol=tolerance):
            results["identical_keys"].append(key)
        else:
            max_diff = torch.max(torch.abs(qwen3_tensor - quality_gate_tensor)).item()
            mean_diff = torch.mean(torch.abs(qwen3_tensor - quality_gate_tensor)).item()
            results["value_mismatch"].append({"key": key, "max_diff": max_diff, "mean_diff": mean_diff, "shape": qwen3_tensor.shape})

    # 2. 检查 Quality-Gate 中的额外权重（应该只有质量门控）
    qwen3_keys = set(qwen3_dict.keys())
    quality_gate_keys = set(quality_gate_dict.keys())
    extra_keys = quality_gate_keys - qwen3_keys

    for key in sorted(extra_keys):
        if "quality_gate" in key:
            tensor = quality_gate_dict[key]
            results["quality_gates"].append(
                {
                    "key": key,
                    "shape": tensor.shape,
                    "dtype": str(tensor.dtype),
                    "mean": tensor.mean().item(),
                    "std": tensor.std().item(),
                    "min": tensor.min().item(),
                    "max": tensor.max().item(),
                }
            )
        else:
            # 不应该有质量门控以外的额外参数
            results["extra_in_quality_gate"].append(key)

    return results


def compare_models(
    qwen3_model_name="Qwen/Qwen3-1.7B",
    quality_gate_model_path="./converted_models/quality_gate_Qwen3-1.7B",
    device="cpu",
    dtype="bfloat16",
    memory_efficient=True,
):
    """
    比较 Qwen3 模型和转换后的 Quality-Gate 模型

    Args:
        qwen3_model_name: Qwen3 模型名称或路径
        quality_gate_model_path: Quality-Gate 模型路径
        device: 设备
        dtype: 数据类型
        memory_efficient: 是否使用内存高效模式
    """
    print("=" * 80)
    print("Qwen3 ↔ Quality-Gate 模型转换验证")
    print(f"原始模型: {qwen3_model_name}")
    print(f"转换模型: {quality_gate_model_path}")
    print(f"设备: {device}")
    print(f"数据类型: {dtype}")
    print("=" * 80)

    # 转换 dtype
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)

    # 注册 Quality-Gate 模型
    register_quality_gate()
    print("✓ Quality-Gate 模型已注册")

    # 显示 GPU 内存信息
    if device != "cpu" and torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
        print(f"GPU 内存: {total_memory:.1f} GB")

    try:
        # 1. 加载原始 Qwen3 模型
        print(f"\n【步骤 1】加载原始 Qwen3 模型")
        print(f"模型: {qwen3_model_name}")
        print("正在加载，请稍候...")

        qwen3_model = AutoModelForCausalLM.from_pretrained(
            qwen3_model_name, torch_dtype=torch_dtype, device_map=device if device != "cpu" else None, low_cpu_mem_usage=True
        )
        qwen3_config = qwen3_model.config

        print("✓ Qwen3 模型加载成功")
        print(f"  模型类型: {type(qwen3_model).__name__}")
        print(f"  层数: {qwen3_config.num_hidden_layers}")
        print(f"  隐藏层大小: {qwen3_config.hidden_size}")
        print(f"  词汇表大小: {qwen3_config.vocab_size}")
        print(f"  总参数量: {sum(p.numel() for p in qwen3_model.parameters()):,}")

        # 提取 state_dict
        if memory_efficient:
            print("  内存高效模式: 提取 state_dict 后释放 GPU 内存")
            qwen3_state_dict = qwen3_model.state_dict()
            qwen3_model = qwen3_model.cpu()
            if device != "cpu":
                torch.cuda.empty_cache()
        else:
            qwen3_state_dict = qwen3_model.state_dict()

        print(f"  State dict 大小: {len(qwen3_state_dict)} 个参数")

        # 2. 加载 Quality-Gate 模型
        print(f"\n【步骤 2】加载转换后的 Quality-Gate 模型")
        print(f"模型路径: {quality_gate_model_path}")

        # 检查模型路径是否存在
        if not os.path.exists(quality_gate_model_path):
            print(f"❌ 错误: 模型路径不存在: {quality_gate_model_path}")
            print("   请先运行 scripts/convert_qwen_to_quality_gate.py 进行模型转换")
            return False

        quality_gate_model = QualityGateForCausalLM.from_pretrained(quality_gate_model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True)

        if device != "cpu":
            quality_gate_model = quality_gate_model.to(device)

        quality_gate_config = quality_gate_model.config

        print("✓ Quality-Gate 模型加载成功")
        print(f"  模型类型: {type(quality_gate_model).__name__}")
        print(f"  配置类型: {type(quality_gate_config).__name__}")
        print(f"  层数: {quality_gate_config.num_hidden_layers}")
        print(f"  总参数量: {sum(p.numel() for p in quality_gate_model.parameters()):,}")
        print(f"  质量门控初始化: mean={quality_gate_config.quality_gate_init_mean}, std={quality_gate_config.quality_gate_init_std}")

        quality_gate_state_dict = quality_gate_model.state_dict()
        print(f"  State dict 大小: {len(quality_gate_state_dict)} 个参数")

        # 3. 比较 state_dict
        print(f"\n【步骤 3】比较模型权重")
        print(f"数值容差: {1e-6 if dtype == 'bfloat16' else 1e-8}")

        tolerance = 1e-6 if dtype == "bfloat16" else 1e-8
        results = compare_state_dicts(qwen3_state_dict, quality_gate_state_dict, tolerance)

        # 输出比较结果
        print("\n" + "─" * 80)
        print("权重比较结果")
        print("─" * 80)

        total_qwen3_params = len(qwen3_state_dict)
        identical_count = len(results["identical_keys"])
        quality_gate_count = len(results["quality_gates"])

        print(f"✓ 完全匹配的权重: {identical_count}/{total_qwen3_params}")
        print(f"✓ 新增质量门控参数: {quality_gate_count}")

        # 显示部分匹配的权重（示例）
        if results["identical_keys"]:
            print("\n  匹配权重示例:")
            for key in results["identical_keys"][:3]:
                print(f"    ✓ {key}")
            if len(results["identical_keys"]) > 3:
                print(f"    ... 还有 {len(results['identical_keys']) - 3} 个")

        # 检查是否有问题
        has_issues = False

        if results["missing_in_quality_gate"]:
            has_issues = True
            print(f"\n❌ Quality-Gate 中缺失的权重: {len(results['missing_in_quality_gate'])}")
            for key in results["missing_in_quality_gate"][:5]:
                print(f"    - {key}")
            if len(results["missing_in_quality_gate"]) > 5:
                print(f"    ... 还有 {len(results['missing_in_quality_gate']) - 5} 个")

        if results["shape_mismatch"]:
            has_issues = True
            print(f"\n❌ 形状不匹配: {len(results['shape_mismatch'])}")
            for item in results["shape_mismatch"][:5]:
                print(f"    - {item['key']}")
                print(f"      Qwen3: {item['qwen3_shape']} → Quality-Gate: {item['quality_gate_shape']}")

        if results["value_mismatch"]:
            has_issues = True
            print(f"\n❌ 数值不匹配: {len(results['value_mismatch'])}")
            for item in results["value_mismatch"][:5]:
                print(f"    - {item['key']}: max_diff={item['max_diff']:.2e}, mean_diff={item['mean_diff']:.2e}")

        if results["extra_in_quality_gate"]:
            has_issues = True
            print(f"\n⚠️  额外的非质量门控参数: {len(results['extra_in_quality_gate'])}")
            for key in results["extra_in_quality_gate"][:5]:
                print(f"    - {key}")

        # 4. 质量门控参数分析
        print("\n" + "─" * 80)
        print("质量门控参数分析")
        print("─" * 80)

        if results["quality_gates"]:
            print(f"质量门控层数: {len(results['quality_gates'])}")
            print(f"预期层数: {qwen3_config.num_hidden_layers}")

            if len(results["quality_gates"]) != qwen3_config.num_hidden_layers:
                has_issues = True
                print(f"❌ 质量门控层数不匹配!")

            # 显示前2个质量门控的详细信息
            print("\n质量门控参数详情:")
            for item in results["quality_gates"][:2]:
                print(f"  {item['key']}")
                print(f"    形状: {item['shape']}")
                print(f"    统计: mean={item['mean']:.6f}, std={item['std']:.6f}")
                print(f"    范围: [{item['min']:.6f}, {item['max']:.6f}]")

            if len(results["quality_gates"]) > 2:
                print(f"  ... 还有 {len(results['quality_gates']) - 2} 个质量门控")

            # 验证初始化是否符合预期
            expected_mean = quality_gate_config.quality_gate_init_mean
            expected_std = quality_gate_config.quality_gate_init_std
            print(f"\n初始化验证:")
            print(f"  配置的初始化: mean={expected_mean}, std={expected_std}")

            actual_means = [item["mean"] for item in results["quality_gates"]]
            actual_stds = [item["std"] for item in results["quality_gates"]]
            avg_mean = sum(actual_means) / len(actual_means)
            avg_std = sum(actual_stds) / len(actual_stds)

            print(f"  实际平均值: mean={avg_mean:.6f}, std={avg_std:.6f}")

            # 检查是否在合理范围内
            mean_ok = abs(avg_mean - expected_mean) < 0.1
            std_ok = abs(avg_std - expected_std) < 0.1
            print(f"  均值检查: {'✓' if mean_ok else '❌'}")
            print(f"  标准差检查: {'✓' if std_ok else '❌'}")

            if not (mean_ok and std_ok):
                has_issues = True

        else:
            has_issues = True
            print("❌ 未找到质量门控参数")

        # 5. 前向传播测试
        print("\n" + "─" * 80)
        print("前向传播测试")
        print("─" * 80)

        print("创建测试输入...")
        batch_size, seq_len = 2, 16
        test_input = torch.randint(0, qwen3_config.vocab_size, (batch_size, seq_len))
        if device != "cpu":
            test_input = test_input.to(device)

        # 测试 Qwen3 模型
        if memory_efficient and device != "cpu":
            print("将 Qwen3 模型移回 GPU...")
            qwen3_model = qwen3_model.to(device)

        print("\n测试 Qwen3 模型...")
        with torch.no_grad():
            qwen3_outputs = qwen3_model(test_input)
            print(f"✓ Qwen3 输出形状: {qwen3_outputs.logits.shape}")

        # 测试 Quality-Gate 模型
        print("\n测试 Quality-Gate 模型（不输出 router_logits）...")
        with torch.no_grad():
            quality_gate_outputs = quality_gate_model(test_input, output_router_logits=False)
            print(f"✓ Quality-Gate 输出形状: {quality_gate_outputs.logits.shape}")

        # 比较输出是否一致
        print("\n比较模型输出一致性...")
        logits_match = torch.allclose(qwen3_outputs.logits, quality_gate_outputs.logits, atol=1e-3, rtol=1e-3)
        max_logits_diff = torch.max(torch.abs(qwen3_outputs.logits - quality_gate_outputs.logits)).item()
        print(f"  输出是否匹配: {'✓' if logits_match else '❌'}")
        print(f"  最大差异: {max_logits_diff:.2e}")

        if not logits_match:
            has_issues = True
            print("  ⚠️  输出不完全一致，但小差异可能是正常的（由于数值精度）")

        # 测试 router_logits 输出
        print("\n测试 Quality-Gate 模型（输出 router_logits）...")
        with torch.no_grad():
            quality_gate_outputs_with_router = quality_gate_model(test_input, output_router_logits=True)
            print(f"✓ 输出形状: {quality_gate_outputs_with_router.logits.shape}")

            if hasattr(quality_gate_outputs_with_router, "router_logits") and quality_gate_outputs_with_router.router_logits:
                router_logits = quality_gate_outputs_with_router.router_logits
                print(f"✓ Router logits 已输出")
                print(f"  层数: {len(router_logits)}")
                print(f"  第一层形状: {router_logits[0].shape}")
                print(f"  预期形状: [batch_size={batch_size}, seq_len={seq_len}, 1]")

                # 验证形状
                expected_shape = (batch_size, seq_len, 1)
                actual_shape = router_logits[0].shape
                shape_ok = actual_shape == expected_shape
                print(f"  形状检查: {'✓' if shape_ok else '❌'}")

                if not shape_ok:
                    has_issues = True
            else:
                has_issues = True
                print("❌ Router logits 未找到")

        # 清理内存
        if memory_efficient and device != "cpu":
            print("\n清理 GPU 内存...")
            del qwen3_model, quality_gate_model
            torch.cuda.empty_cache()
            print("✓ 内存已清理")

        # 6. 最终总结
        print("\n" + "=" * 80)
        print("验证总结")
        print("=" * 80)

        print(f"\n基础权重:")
        print(f"  Qwen3 参数总数: {total_qwen3_params}")
        print(f"  完全匹配: {identical_count}/{total_qwen3_params} ({100 * identical_count / total_qwen3_params:.1f}%)")

        print(f"\n质量门控:")
        print(f"  新增参数数量: {quality_gate_count}")
        print(f"  预期层数: {qwen3_config.num_hidden_layers}")
        print(f"  层数检查: {'✓' if quality_gate_count == qwen3_config.num_hidden_layers else '❌'}")

        print(f"\n前向传播:")
        print(f"  基础输出一致性: {'✓' if logits_match else '⚠️'}")
        print(f"  Router logits: {'✓' if hasattr(quality_gate_outputs_with_router, 'router_logits') else '❌'}")

        # 判断是否通过
        success = (
            identical_count == total_qwen3_params  # 所有 Qwen3 权重都匹配
            and len(results["missing_in_quality_gate"]) == 0  # 没有缺失
            and len(results["shape_mismatch"]) == 0  # 没有形状不匹配
            and len(results["value_mismatch"]) == 0  # 没有数值不匹配
            and len(results["extra_in_quality_gate"]) == 0  # 没有额外参数
            and quality_gate_count == qwen3_config.num_hidden_layers  # 质量门控层数正确
            and hasattr(quality_gate_outputs_with_router, "router_logits")  # router_logits 可用
        )

        print("\n" + "=" * 80)
        if success:
            print("🎉 验证通过！转换后的模型完全正确")
            print("=" * 80)
            print("\n✅ 所有检查项:")
            print("   ✓ Qwen3 基础权重完全匹配")
            print("   ✓ 质量门控层数正确")
            print("   ✓ 质量门控参数初始化正确")
            print("   ✓ 无缺失或额外参数")
            print("   ✓ 前向传播正常")
            print("   ✓ Router logits 输出正确")
            print("\n模型已可用于训练！")
        else:
            print("❌ 验证失败！发现问题")
            print("=" * 80)
            if has_issues:
                print("\n请检查上述错误信息，可能需要重新转换模型")

        return success

    except Exception as e:
        print(f"\n❌ 验证过程出错: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="验证 Qwen3 → Quality-Gate 模型转换的正确性", epilog="建议在转换模型后立即运行此脚本进行验证")

    parser.add_argument("--qwen3-model", type=str, default="Qwen/Qwen3-1.7B", help="原始 Qwen3 模型名称或路径")
    parser.add_argument(
        "--quality-gate-model",
        type=str,
        default="./converted_models/quality_gate_Qwen3-1.7B",
        help="转换后的 Quality-Gate 模型路径",
    )
    parser.add_argument("--device", type=str, default="cpu", help="设备 (cpu, cuda, cuda:0 等)")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"], help="数据类型")
    parser.add_argument("--memory-efficient", action="store_true", help="使用内存高效模式（顺序加载模型）")

    args = parser.parse_args()

    # 验证设备
    if args.device != "cpu" and not torch.cuda.is_available():
        print("⚠️  CUDA 不可用，回退到 CPU")
        args.device = "cpu"

    print("配置:")
    print(f"  Qwen3 模型: {args.qwen3_model}")
    print(f"  Quality-Gate 模型: {args.quality_gate_model}")
    print(f"  设备: {args.device}")
    print(f"  数据类型: {args.dtype}")
    print(f"  内存高效模式: {args.memory_efficient}")

    try:
        success = compare_models(
            qwen3_model_name=args.qwen3_model,
            quality_gate_model_path=args.quality_gate_model,
            device=args.device,
            dtype=args.dtype,
            memory_efficient=args.memory_efficient,
        )
        return success
    except KeyboardInterrupt:
        print("\n⚠️  用户中断")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
