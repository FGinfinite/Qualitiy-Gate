#!/usr/bin/env python3
"""
Router Data对比脚本

该脚本用于对比两个router_data.pt文件的相似性，分析不同实验配置下
质量门控的输出是否一致。

使用方法:
    python exp_scripts/compare_router_data.py \
        --file1 outputs/stage_2_selection/.../router_data/gsm8k_router_data.pt \
        --file2 outputs/stage_2_selection/.../router_data/gsm8k_router_data.pt

    # 或者使用快捷模式，自动比较两个实验目录
    python exp_scripts/compare_router_data.py \
        --dir1 outputs/stage_2_selection/.../batch=16_lr=0.0003... \
        --dir2 outputs/stage_2_selection/.../batch=16_lr=0.0005... \
        --dataset gsm8k
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def load_router_data(file_path: Path) -> Dict[str, Any]:
    """加载router_data.pt文件"""
    try:
        data = torch.load(file_path, map_location="cpu")
        print(f"✓ 成功加载: {file_path}")
        return data
    except Exception as e:
        print(f"✗ 加载失败 {file_path}: {e}")
        sys.exit(1)


def analyze_structure(data: Dict[str, Any], label: str = "Data"):
    """分析数据结构"""
    print(f"\n{'=' * 80}")
    print(f"{label} 结构分析")
    print(f"{'=' * 80}")

    if isinstance(data, dict):
        print("数据类型: 字典")
        print(f"键数量: {len(data)}")
        print(f"键列表: {list(data.keys())}")

        for key, value in data.items():
            print(f"\n键: {key}")
            if isinstance(value, torch.Tensor):
                print("  类型: Tensor")
                print(f"  形状: {value.shape}")
                print(f"  数据类型: {value.dtype}")
                print(f"  设备: {value.device}")
                if value.numel() > 0:
                    print(f"  范围: [{value.min().item():.6f}, {value.max().item():.6f}]")
                    print(f"  均值: {value.mean().item():.6f}")
                    print(f"  标准差: {value.std().item():.6f}")
            elif isinstance(value, list):
                print("  类型: 列表")
                print(f"  长度: {len(value)}")
                if len(value) > 0:
                    print(f"  元素类型: {type(value[0])}")
                    if isinstance(value[0], (int, float)):
                        arr = np.array(value)
                        print(f"  范围: [{arr.min():.6f}, {arr.max():.6f}]")
                        print(f"  均值: {arr.mean():.6f}")
                        print(f"  标准差: {arr.std():.6f}")
            elif isinstance(value, dict):
                print("  类型: 字典")
                print(f"  子键: {list(value.keys())}")
            else:
                print(f"  类型: {type(value)}")
                print(f"  值: {value}")
    else:
        print(f"数据类型: {type(data)}")


def compare_tensors(tensor1: torch.Tensor, tensor2: torch.Tensor, key: str) -> Dict[str, Any]:
    """详细对比两个tensor"""
    result = {
        "key": key,
        "identical": False,
        "shape_match": False,
        "dtype_match": False,
        "max_abs_diff": None,
        "mean_abs_diff": None,
        "cosine_similarity": None,
        "pearson_correlation": None,
        "exact_match_ratio": None,
    }

    # 检查形状
    result["shape_match"] = tensor1.shape == tensor2.shape
    if not result["shape_match"]:
        return result

    # 检查数据类型
    result["dtype_match"] = tensor1.dtype == tensor2.dtype

    # 转换为float进行比较
    t1 = tensor1.float()
    t2 = tensor2.float()

    # 完全相同检查
    result["identical"] = torch.equal(tensor1, tensor2)

    # 数值差异分析
    diff = torch.abs(t1 - t2)
    result["max_abs_diff"] = diff.max().item()
    result["mean_abs_diff"] = diff.mean().item()

    # 精确匹配比例
    exact_match = (tensor1 == tensor2).float().mean().item()
    result["exact_match_ratio"] = exact_match

    # 余弦相似度（flatten后计算）
    t1_flat = t1.flatten()
    t2_flat = t2.flatten()
    if t1_flat.numel() > 0:
        cosine_sim = torch.nn.functional.cosine_similarity(t1_flat.unsqueeze(0), t2_flat.unsqueeze(0))
        result["cosine_similarity"] = cosine_sim.item()

        # Pearson相关系数
        if t1_flat.std() > 0 and t2_flat.std() > 0:
            t1_centered = t1_flat - t1_flat.mean()
            t2_centered = t2_flat - t2_flat.mean()
            pearson = (t1_centered * t2_centered).mean() / (t1_centered.std() * t2_centered.std())
            result["pearson_correlation"] = pearson.item()

    return result


def compare_lists(list1: list, list2: list, key: str) -> Dict[str, Any]:
    """对比两个列表"""
    result = {
        "key": key,
        "identical": False,
        "length_match": False,
        "exact_match_ratio": None,
    }

    # 检查长度
    result["length_match"] = len(list1) == len(list2)
    if not result["length_match"]:
        return result

    # 完全相同检查
    result["identical"] = list1 == list2

    # 如果是数值列表，计算相似度
    if len(list1) > 0 and isinstance(list1[0], (int, float)):
        arr1 = np.array(list1)
        arr2 = np.array(list2)

        # 精确匹配比例
        exact_match = (arr1 == arr2).mean()
        result["exact_match_ratio"] = exact_match

        # 数值差异
        result["max_abs_diff"] = np.abs(arr1 - arr2).max()
        result["mean_abs_diff"] = np.abs(arr1 - arr2).mean()

    return result


def compare_router_data(data1: Dict, data2: Dict) -> Dict[str, Any]:
    """对比两个router_data字典"""
    print(f"\n{'=' * 80}")
    print("数据对比分析")
    print(f"{'=' * 80}")

    comparison = {
        "keys_match": False,
        "identical": False,
        "field_comparisons": {},
    }

    # 检查键是否一致
    keys1 = set(data1.keys())
    keys2 = set(data2.keys())
    comparison["keys_match"] = keys1 == keys2

    print("\n键对比:")
    print(f"  文件1键: {sorted(keys1)}")
    print(f"  文件2键: {sorted(keys2)}")
    print(f"  键一致: {'✓ 是' if comparison['keys_match'] else '✗ 否'}")

    if not comparison["keys_match"]:
        only_in_1 = keys1 - keys2
        only_in_2 = keys2 - keys1
        if only_in_1:
            print(f"  仅在文件1中: {sorted(only_in_1)}")
        if only_in_2:
            print(f"  仅在文件2中: {sorted(only_in_2)}")

    # 对比每个字段
    common_keys = keys1 & keys2
    all_identical = True

    for key in sorted(common_keys):
        value1 = data1[key]
        value2 = data2[key]

        if isinstance(value1, torch.Tensor) and isinstance(value2, torch.Tensor):
            result = compare_tensors(value1, value2, key)
            comparison["field_comparisons"][key] = result
            if not result["identical"]:
                all_identical = False

        elif isinstance(value1, list) and isinstance(value2, list):
            result = compare_lists(value1, value2, key)
            comparison["field_comparisons"][key] = result
            if not result["identical"]:
                all_identical = False

        else:
            # 直接比较
            identical = value1 == value2
            comparison["field_comparisons"][key] = {
                "key": key,
                "identical": identical,
                "type1": type(value1).__name__,
                "type2": type(value2).__name__,
            }
            if not identical:
                all_identical = False

    comparison["identical"] = all_identical and comparison["keys_match"]

    return comparison


def print_comparison_results(comparison: Dict[str, Any]):
    """打印对比结果"""
    print(f"\n{'=' * 80}")
    print("详细对比结果")
    print(f"{'=' * 80}")

    overall_identical = comparison["identical"]
    print(f"\n总体结论: {'✓ 完全一致' if overall_identical else '✗ 存在差异'}")

    if not comparison["field_comparisons"]:
        print("\n无可比较的字段")
        return

    print("\n字段对比详情:")
    print("-" * 80)

    for key, result in comparison["field_comparisons"].items():
        print(f"\n字段: {key}")
        print(f"  完全相同: {'✓ 是' if result.get('identical', False) else '✗ 否'}")

        if "shape_match" in result:
            # Tensor对比结果
            print(f"  形状匹配: {'✓' if result['shape_match'] else '✗'}")
            if result["shape_match"]:
                if result.get("exact_match_ratio") is not None:
                    print(f"  精确匹配率: {result['exact_match_ratio'] * 100:.2f}%")
                if result.get("max_abs_diff") is not None:
                    print(f"  最大绝对差异: {result['max_abs_diff']:.6e}")
                if result.get("mean_abs_diff") is not None:
                    print(f"  平均绝对差异: {result['mean_abs_diff']:.6e}")
                if result.get("cosine_similarity") is not None:
                    print(f"  余弦相似度: {result['cosine_similarity']:.6f}")
                if result.get("pearson_correlation") is not None:
                    print(f"  Pearson相关: {result['pearson_correlation']:.6f}")

        elif "length_match" in result:
            # 列表对比结果
            print(f"  长度匹配: {'✓' if result['length_match'] else '✗'}")
            if result["length_match"] and result.get("exact_match_ratio") is not None:
                print(f"  精确匹配率: {result['exact_match_ratio'] * 100:.2f}%")
                if result.get("max_abs_diff") is not None:
                    print(f"  最大绝对差异: {result['max_abs_diff']:.6e}")
                if result.get("mean_abs_diff") is not None:
                    print(f"  平均绝对差异: {result['mean_abs_diff']:.6e}")

        else:
            # 其他类型
            if "type1" in result:
                print(f"  类型1: {result['type1']}")
                print(f"  类型2: {result['type2']}")


def generate_summary(comparison: Dict[str, Any]) -> str:
    """生成简要总结"""
    summary = []
    summary.append("\n" + "=" * 80)
    summary.append("对比摘要")
    summary.append("=" * 80)

    if comparison["identical"]:
        summary.append("\n✅ 结论: 两个文件完全一致（包括所有数值和结构）")
    else:
        summary.append("\n⚠️ 结论: 两个文件存在差异")

        # 统计相似度
        tensor_results = [r for r in comparison["field_comparisons"].values() if "cosine_similarity" in r and r["shape_match"]]

        if tensor_results:
            summary.append("\nTensor字段统计:")

            identical_count = sum(1 for r in tensor_results if r["identical"])
            summary.append(f"  完全相同的字段: {identical_count}/{len(tensor_results)}")

            # 平均相似度
            avg_cosine = np.mean([r["cosine_similarity"] for r in tensor_results if r["cosine_similarity"] is not None])
            avg_pearson = np.mean([r["pearson_correlation"] for r in tensor_results if r["pearson_correlation"] is not None])
            avg_exact_match = np.mean([r["exact_match_ratio"] for r in tensor_results if r["exact_match_ratio"] is not None])

            summary.append(f"  平均余弦相似度: {avg_cosine:.6f}")
            summary.append(f"  平均Pearson相关: {avg_pearson:.6f}")
            summary.append(f"  平均精确匹配率: {avg_exact_match * 100:.2f}%")

            # 找出差异最大的字段
            max_diff_field = max(tensor_results, key=lambda r: r.get("max_abs_diff", 0))
            if max_diff_field["max_abs_diff"] > 0:
                summary.append(f"\n  差异最大的字段: {max_diff_field['key']}")
                summary.append(f"    最大绝对差异: {max_diff_field['max_abs_diff']:.6e}")
                summary.append(f"    平均绝对差异: {max_diff_field['mean_abs_diff']:.6e}")

    summary.append("\n" + "=" * 80)
    return "\n".join(summary)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="对比两个router_data.pt文件的相似性")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file1", type=str, help="第一个.pt文件路径")
    group.add_argument("--dir1", type=str, help="第一个实验目录（使用--dataset指定数据集）")

    parser.add_argument("--file2", type=str, help="第二个.pt文件路径")
    parser.add_argument("--dir2", type=str, help="第二个实验目录（使用--dataset指定数据集）")
    parser.add_argument("--dataset", type=str, default="gsm8k", help="数据集名称（与--dir1/--dir2配合使用）")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细的结构分析")

    args = parser.parse_args()

    # 确定文件路径
    if args.file1:
        file1 = Path(args.file1)
        file2 = Path(args.file2) if args.file2 else None
    else:
        # 从目录构建路径
        file1 = Path(args.dir1) / "router_data" / f"{args.dataset}_router_data.pt"
        if args.dir2:
            file2 = Path(args.dir2) / "router_data" / f"{args.dataset}_router_data.pt"
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
    print("Router Data 相似性对比工具")
    print(f"{'=' * 80}")
    print(f"\n文件1: {file1}")
    print(f"文件2: {file2}")

    # 加载数据
    data1 = load_router_data(file1)
    data2 = load_router_data(file2)

    # 详细结构分析（可选）
    if args.verbose:
        analyze_structure(data1, "文件1")
        analyze_structure(data2, "文件2")

    # 对比数据
    comparison = compare_router_data(data1, data2)

    # 打印详细结果
    print_comparison_results(comparison)

    # 打印总结
    summary = generate_summary(comparison)
    print(summary)


if __name__ == "__main__":
    main()
