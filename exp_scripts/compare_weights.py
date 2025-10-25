#!/usr/bin/env python3
"""
模型权重对比脚本

该脚本用于对比两个full_rank_weights.pt文件的相似性，
分析不同训练配置下保存的模型权重是否一致。

使用方法:
    python exp_scripts/compare_weights.py \
        --file1 outputs/stage_1_warmup/.../full_rank_weights.pt \
        --file2 outputs/stage_1_warmup/.../full_rank_weights.pt

    # 或者使用快捷模式，自动定位权重文件
    python exp_scripts/compare_weights.py \
        --dir1 outputs/stage_1_warmup/.../batch=16_lr=0.001... \
        --dir2 outputs/stage_1_warmup/.../batch=16_lr=0.0003...
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def load_weights(file_path: Path) -> Dict[str, Any]:
    """加载权重文件"""
    try:
        weights = torch.load(file_path, map_location="cpu")
        print(f"✓ 成功加载: {file_path}")
        return weights
    except Exception as e:
        print(f"✗ 加载失败 {file_path}: {e}")
        sys.exit(1)


def analyze_weights_structure(weights: Dict[str, Any], label: str = "Weights"):
    """分析权重结构"""
    print(f"\n{'=' * 80}")
    print(f"{label} 结构分析")
    print(f"{'=' * 80}")

    if isinstance(weights, dict):
        print("数据类型: 字典")
        print(f"键数量: {len(weights)}")

        # 统计参数量
        total_params = 0
        tensor_count = 0
        non_tensor_count = 0

        for key, value in weights.items():
            if isinstance(value, torch.Tensor):
                tensor_count += 1
                total_params += value.numel()
            else:
                non_tensor_count += 1

        print(f"Tensor参数数量: {tensor_count}")
        print(f"非Tensor项数量: {non_tensor_count}")
        print(f"总参数量: {total_params:,} ({total_params / 1e6:.2f}M)")

        # 显示前几个键
        print(f"\n前10个键:")
        for i, key in enumerate(sorted(weights.keys())[:10]):
            value = weights[key]
            if isinstance(value, torch.Tensor):
                print(f"  {i + 1}. {key}: {value.shape} ({value.dtype})")
            else:
                print(f"  {i + 1}. {key}: {type(value).__name__}")

        if len(weights) > 10:
            print(f"  ... (还有 {len(weights) - 10} 个键)")
    else:
        print(f"数据类型: {type(weights)}")


def compare_tensor_weights(tensor1: torch.Tensor, tensor2: torch.Tensor, key: str, verbose: bool = False) -> Dict[str, Any]:
    """对比两个权重tensor"""
    result = {
        "key": key,
        "identical": False,
        "shape_match": False,
        "dtype_match": False,
        "max_abs_diff": None,
        "mean_abs_diff": None,
        "relative_diff": None,
        "cosine_similarity": None,
        "exact_match_ratio": None,
    }

    # 检查形状
    result["shape_match"] = tensor1.shape == tensor2.shape
    if not result["shape_match"]:
        if verbose:
            print(f"  ⚠️  {key}: 形状不匹配 {tensor1.shape} vs {tensor2.shape}")
        return result

    # 检查数据类型
    result["dtype_match"] = tensor1.dtype == tensor2.dtype

    # 完全相同检查
    result["identical"] = torch.equal(tensor1, tensor2)

    # 转换为float进行数值比较
    t1 = tensor1.float()
    t2 = tensor2.float()

    # 数值差异分析
    diff = torch.abs(t1 - t2)
    result["max_abs_diff"] = diff.max().item()
    result["mean_abs_diff"] = diff.mean().item()

    # 相对差异（避免除零）
    magnitude = torch.abs(t1).mean()
    if magnitude > 1e-10:
        result["relative_diff"] = result["mean_abs_diff"] / magnitude.item()

    # 精确匹配比例
    exact_match = (tensor1 == tensor2).float().mean().item()
    result["exact_match_ratio"] = exact_match

    # 余弦相似度（flatten后计算）
    t1_flat = t1.flatten()
    t2_flat = t2.flatten()
    if t1_flat.numel() > 0 and t1_flat.norm() > 1e-10 and t2_flat.norm() > 1e-10:
        cosine_sim = torch.nn.functional.cosine_similarity(t1_flat.unsqueeze(0), t2_flat.unsqueeze(0))
        result["cosine_similarity"] = cosine_sim.item()

    return result


def compare_weights(weights1: Dict, weights2: Dict, verbose: bool = False) -> Dict[str, Any]:
    """对比两个权重字典"""
    print(f"\n{'=' * 80}")
    print("权重对比分析")
    print(f"{'=' * 80}")

    comparison = {
        "keys_match": False,
        "identical": False,
        "weight_comparisons": {},
        "statistics": {
            "total_params": 0,
            "identical_params": 0,
            "different_params": 0,
            "avg_max_diff": 0.0,
            "avg_mean_diff": 0.0,
            "avg_cosine_sim": 0.0,
        },
    }

    # 检查键是否一致
    keys1 = set(weights1.keys())
    keys2 = set(weights2.keys())
    comparison["keys_match"] = keys1 == keys2

    print("\n键对比:")
    print(f"  文件1键数量: {len(keys1)}")
    print(f"  文件2键数量: {len(keys2)}")
    print(f"  键一致: {'✓ 是' if comparison['keys_match'] else '✗ 否'}")

    if not comparison["keys_match"]:
        only_in_1 = keys1 - keys2
        only_in_2 = keys2 - keys1
        if only_in_1:
            print(f"  仅在文件1中的键数量: {len(only_in_1)}")
            if verbose:
                for key in sorted(only_in_1)[:5]:
                    print(f"    - {key}")
                if len(only_in_1) > 5:
                    print(f"    ... (还有 {len(only_in_1) - 5} 个)")
        if only_in_2:
            print(f"  仅在文件2中的键数量: {len(only_in_2)}")
            if verbose:
                for key in sorted(only_in_2)[:5]:
                    print(f"    - {key}")
                if len(only_in_2) > 5:
                    print(f"    ... (还有 {len(only_in_2) - 5} 个)")

    # 对比每个权重
    common_keys = keys1 & keys2
    all_identical = True

    print(f"\n正在对比 {len(common_keys)} 个公共键...")

    max_diffs = []
    mean_diffs = []
    cosine_sims = []
    total_params = 0
    identical_params = 0

    for key in sorted(common_keys):
        value1 = weights1[key]
        value2 = weights2[key]

        if isinstance(value1, torch.Tensor) and isinstance(value2, torch.Tensor):
            result = compare_tensor_weights(value1, value2, key, verbose=verbose)
            comparison["weight_comparisons"][key] = result

            if result["shape_match"]:
                params = value1.numel()
                total_params += params

                if result["identical"]:
                    identical_params += params
                else:
                    all_identical = False

                if result["max_abs_diff"] is not None:
                    max_diffs.append(result["max_abs_diff"])
                if result["mean_abs_diff"] is not None:
                    mean_diffs.append(result["mean_abs_diff"])
                if result["cosine_similarity"] is not None:
                    cosine_sims.append(result["cosine_similarity"])
            else:
                all_identical = False

        elif isinstance(value1, dict) and isinstance(value2, dict):
            # 递归处理嵌套字典
            if verbose:
                print(f"  发现嵌套字典: {key} (包含 {len(value1)} 个子键)")
            # 递归对比嵌套的权重
            nested_comparison = compare_weights(value1, value2, verbose=False)
            comparison["weight_comparisons"][key] = {
                "key": key,
                "type": "nested_dict",
                "identical": nested_comparison["identical"],
                "nested_stats": nested_comparison["statistics"],
            }
            # 合并统计数据
            total_params += nested_comparison["statistics"]["total_params"]
            identical_params += nested_comparison["statistics"]["identical_params"]

            if nested_comparison["statistics"].get("avg_max_diff"):
                max_diffs.append(nested_comparison["statistics"]["avg_max_diff"])
            if nested_comparison["statistics"].get("avg_mean_diff"):
                mean_diffs.append(nested_comparison["statistics"]["avg_mean_diff"])
            if nested_comparison["statistics"].get("avg_cosine_sim"):
                cosine_sims.append(nested_comparison["statistics"]["avg_cosine_sim"])

            if not nested_comparison["identical"]:
                all_identical = False
        else:
            # 其他类型直接比较
            try:
                identical = value1 == value2
            except Exception:
                # 如果比较失败，认为不相同
                identical = False

            comparison["weight_comparisons"][key] = {
                "key": key,
                "identical": identical,
                "type1": type(value1).__name__,
                "type2": type(value2).__name__,
            }
            if not identical:
                all_identical = False

    comparison["identical"] = all_identical and comparison["keys_match"]

    # 计算统计数据
    comparison["statistics"]["total_params"] = total_params
    comparison["statistics"]["identical_params"] = identical_params
    comparison["statistics"]["different_params"] = total_params - identical_params

    if max_diffs:
        comparison["statistics"]["avg_max_diff"] = np.mean(max_diffs)
        comparison["statistics"]["max_of_max_diff"] = np.max(max_diffs)
    if mean_diffs:
        comparison["statistics"]["avg_mean_diff"] = np.mean(mean_diffs)
    if cosine_sims:
        comparison["statistics"]["avg_cosine_sim"] = np.mean(cosine_sims)
        comparison["statistics"]["min_cosine_sim"] = np.min(cosine_sims)

    return comparison


def print_comparison_summary(comparison: Dict[str, Any], verbose: bool = False):
    """打印对比摘要"""
    print(f"\n{'=' * 80}")
    print("对比摘要")
    print(f"{'=' * 80}")

    stats = comparison["statistics"]

    if comparison["identical"]:
        print("\n✅ 结论: 两个权重文件完全一致（包括所有参数和结构）")
    else:
        print("\n⚠️ 结论: 两个权重文件存在差异")

        print("\n参数统计:")
        print(f"  总参数量: {stats['total_params']:,} ({stats['total_params'] / 1e6:.2f}M)")
        print(f"  相同参数: {stats['identical_params']:,} ({stats['identical_params'] / 1e6:.2f}M)")
        print(f"  不同参数: {stats['different_params']:,} ({stats['different_params'] / 1e6:.2f}M)")

        if stats["total_params"] > 0:
            identical_ratio = stats["identical_params"] / stats["total_params"]
            print(f"  相同比例: {identical_ratio * 100:.2f}%")

        if stats.get("avg_max_diff") is not None:
            print(f"\n差异统计:")
            print(f"  平均最大差异: {stats['avg_max_diff']:.6e}")
            print(f"  最大的最大差异: {stats.get('max_of_max_diff', 0):.6e}")
            print(f"  平均均值差异: {stats.get('avg_mean_diff', 0):.6e}")

        if stats.get("avg_cosine_sim") is not None:
            print(f"\n相似度统计:")
            print(f"  平均余弦相似度: {stats['avg_cosine_sim']:.6f}")
            print(f"  最小余弦相似度: {stats.get('min_cosine_sim', 0):.6f}")

        # 找出差异最大的权重
        tensor_results = [
            r
            for r in comparison["weight_comparisons"].values()
            if isinstance(r, dict) and "max_abs_diff" in r and r["max_abs_diff"] is not None and not r["identical"]
        ]

        if tensor_results and verbose:
            print("\n差异最大的前10个权重:")
            sorted_by_diff = sorted(tensor_results, key=lambda x: x.get("max_abs_diff", 0), reverse=True)[:10]
            for i, result in enumerate(sorted_by_diff, 1):
                key = result["key"]
                max_diff = result["max_abs_diff"]
                mean_diff = result["mean_abs_diff"]
                cosine = result.get("cosine_similarity", "N/A")
                exact_match = result.get("exact_match_ratio", 0) * 100

                print(f"\n  {i}. {key}")
                print(f"     最大差异: {max_diff:.6e}")
                print(f"     平均差异: {mean_diff:.6e}")
                print(f"     精确匹配: {exact_match:.2f}%")
                if isinstance(cosine, float):
                    print(f"     余弦相似度: {cosine:.6f}")

    print("\n" + "=" * 80)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="对比两个模型权重文件的相似性")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file1", type=str, help="第一个权重文件路径")
    group.add_argument("--dir1", type=str, help="第一个实验目录（自动查找full_rank_weights.pt）")

    parser.add_argument("--file2", type=str, help="第二个权重文件路径")
    parser.add_argument("--dir2", type=str, help="第二个实验目录（自动查找full_rank_weights.pt）")
    parser.add_argument("--weight-file", type=str, default="full_rank_weights.pt", help="权重文件名（默认: full_rank_weights.pt）")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细的对比信息")
    parser.add_argument("--analyze", "-a", action="store_true", help="显示权重结构分析")

    args = parser.parse_args()

    # 确定文件路径
    if args.file1:
        file1 = Path(args.file1)
        file2 = Path(args.file2) if args.file2 else None
    else:
        # 从目录构建路径
        file1 = Path(args.dir1) / args.weight_file
        if args.dir2:
            file2 = Path(args.dir2) / args.weight_file
        else:
            file2 = None

    if file2 is None:
        print("❌ 错误: 必须指定第二个文件（--file2）或目录（--dir2）")
        sys.exit(1)

    # 检查文件存在性
    if not file1.exists():
        print(f"❌ 错误: 文件不存在: {file1}")
        sys.exit(1)

    if not file2.exists():
        print(f"❌ 错误: 文件不存在: {file2}")
        sys.exit(1)

    print(f"\n{'=' * 80}")
    print("模型权重相似性对比工具")
    print(f"{'=' * 80}")
    print(f"\n文件1: {file1}")
    print(f"文件2: {file2}")

    # 加载权重
    weights1 = load_weights(file1)
    weights2 = load_weights(file2)

    # 结构分析（可选）
    if args.analyze:
        analyze_weights_structure(weights1, "文件1")
        analyze_weights_structure(weights2, "文件2")

    # 对比权重
    comparison = compare_weights(weights1, weights2, verbose=args.verbose)

    # 打印摘要
    print_comparison_summary(comparison, verbose=args.verbose)


if __name__ == "__main__":
    main()
