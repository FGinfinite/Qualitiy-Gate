#!/usr/bin/env python3
"""
批量质量分数分布可视化脚本

该脚本自动扫描指定日期目录下的所有实验结果，
逐个调用可视化脚本分析其质量分数分布，
并将结果保存到统一的输出目录。

使用方法:
    python exp_scripts/batch_visualize_quality_distribution.py \
        --stage-2-output outputs/stage_2_selection \
        --date 2025-10-31 \
        --output-root outputs/visual/router

    # 完整参数示例
    python exp_scripts/batch_visualize_quality_distribution.py \
        --stage-2-output outputs/stage_2_selection \
        --date 2025-10-31 \
        --output-root outputs/visual/router \
        --bin-width 0.01 \
        --analyze
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List


def find_router_data_dirs(stage2_output: str, date: str) -> List[Path]:
    """
    扫描指定日期目录下所有包含router_data的实验目录

    Args:
        stage2_output: stage2输出根目录
        date: 日期字符串 (YYYY-MM-DD)

    Returns:
        包含router_data的实验目录路径列表
    """
    stage2_path = Path(stage2_output)
    date_path = stage2_path / date

    if not date_path.exists():
        print(f"❌ 错误: 日期目录不存在: {date_path}")
        return []

    print(f"📂 扫描目录: {date_path}")

    # 查找所有包含router_data目录的实验
    router_dirs = []
    for exp_dir in sorted(date_path.glob("*")):
        if exp_dir.is_dir() and (exp_dir / "router_data").exists():
            router_dirs.append(exp_dir / "router_data")  # 返回router_data子目录而非实验目录

    print(f"✅ 找到 {len(router_dirs)} 个实验目录:")
    for exp_dir in router_dirs:
        print(f"   - {exp_dir.name}")

    return router_dirs


def get_visualization_script_path() -> Path:
    """获取可视化脚本的路径"""
    script_path = Path(__file__).parent / "visualize_quality_distribution.py"
    if not script_path.exists():
        # 尝试在根目录查找
        script_path = Path(__file__).parent.parent / "visual_analyse" / "visualize_quality_distribution.py"
    return script_path


def run_single_visualization(
    script_path: Path,
    router_data_dir: Path,
    output_image_path: Path,
    bin_width: float,
    analyze: bool,
) -> bool:
    """
    运行单个可视化脚本

    Args:
        script_path: 可视化脚本路径
        router_data_dir: router_data目录
        output_image_path: 输出图片路径
        bin_width: bin宽度
        analyze: 是否进行分离度分析

    Returns:
        是否成功
    """
    cmd = [
        sys.executable,
        str(script_path),
        "--router-data-dir",
        str(router_data_dir),
        "--output",
        str(output_image_path),
        "--bin-width",
        str(bin_width),
    ]

    if analyze:
        cmd.append("--analyze")

    print(f"\n{'=' * 80}")
    print(f"🚀 运行可视化: {router_data_dir.name}")
    print(f"   输出: {output_image_path.name}")
    print(f"{'=' * 80}")

    try:
        subprocess.run(
            cmd,
            capture_output=False,
            text=True,
            check=True,
            cwd=str(Path(__file__).parent.parent),  # 设置工作目录为项目根目录
        )
        print(f"✅ 完成: {router_data_dir.name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 失败: {router_data_dir.name}")
        print(f"   错误代码: {e.returncode}")
        return False
    except Exception as e:
        print(f"❌ 异常: {router_data_dir.name}")
        print(f"   错误信息: {e}")
        return False


def batch_visualize_quality_distribution(
    stage2_output: str,
    date: str,
    output_root: str,
    bin_width: float = 0.01,
    analyze: bool = False,
):
    """
    批量可视化质量分数分布

    Args:
        stage2_output: stage2输出根目录
        date: 日期字符串
        output_root: 输出根目录
        bin_width: 直方图bin宽度
        analyze: 是否进行分离度分析
    """
    # 创建输出目录
    output_root = Path(output_root)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = output_root / f"{date}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 输出目录: {output_dir}")

    # 查找所有router_data目录
    router_dirs = find_router_data_dirs(stage2_output, date)

    if not router_dirs:
        print("⚠️  未找到任何包含router_data的目录，退出")
        return

    # 获取可视化脚本路径
    script_path = get_visualization_script_path()
    if not script_path.exists():
        print(f"❌ 错误: 未找到可视化脚本: {script_path}")
        return

    print(f"\n📜 使用可视化脚本: {script_path}")

    # 统计结果
    results = {
        "date": date,
        "timestamp": timestamp,
        "total_experiments": len(router_dirs),
        "successful": 0,
        "failed": 0,
        "experiments": [],
    }

    # 逐个运行可视化
    for i, router_dir in enumerate(router_dirs, 1):
        # 构建输出图片路径
        exp_name = router_dir.parent.name  # 使用实验目录名（router_data的父目录）
        output_image = output_dir / f"{exp_name}.png"

        print(f"\n[{i}/{len(router_dirs)}] 处理实验: {exp_name}")

        # 运行可视化
        success = run_single_visualization(
            script_path=script_path,
            router_data_dir=router_dir,
            output_image_path=output_image,
            bin_width=bin_width,
            analyze=analyze,
        )

        # 记录结果
        results["experiments"].append(
            {
                "name": exp_name,
                "router_data_dir": str(router_dir),
                "output_image": str(output_image),
                "success": success,
            }
        )

        if success:
            results["successful"] += 1
        else:
            results["failed"] += 1

    # 保存结果摘要
    results_file = output_dir / "batch_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 打印摘要
    print("\n" + "=" * 80)
    print("🎉 批量处理完成!")
    print("=" * 80)
    print(f"📅 日期: {date}")
    print(f"⏱️  时间: {timestamp}")
    print(f"📊 总计实验: {results['total_experiments']}")
    print(f"✅ 成功: {results['successful']}")
    print(f"❌ 失败: {results['failed']}")
    print(f"📁 输出目录: {output_dir}")
    print(f"📄 结果文件: {results_file}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="批量质量分数分布可视化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法
  python exp_scripts/batch_visualize_quality_distribution.py \\
      --stage-2-output outputs/stage_2_selection \\
      --date 2025-10-31

  # 指定输出目录和分析选项
  python exp_scripts/batch_visualize_quality_distribution.py \\
      --stage-2-output outputs/stage_2_selection \\
      --date 2025-10-31 \\
      --output-root outputs/visual/router \\
      --bin-width 0.02 \\
      --analyze
        """,
    )

    parser.add_argument("--stage-2-output", type=str, default="outputs/stage_2_selection", help="stage2输出根目录 (默认: outputs/stage_2_selection)")

    parser.add_argument("--date", type=str, required=True, help="要分析的日期 (格式: YYYY-MM-DD)")

    parser.add_argument("--output-root", type=str, default="outputs/visual/router", help="输出根目录 (默认: outputs/visual/router)")

    parser.add_argument("--bin-width", type=float, default=0.01, help="直方图bin宽度 (默认: 0.01即1%)")

    parser.add_argument("--analyze", action="store_true", help="打印详细的分离度分析")

    args = parser.parse_args()

    # 验证日期格式
    try:
        datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        print(f"❌ 错误: 日期格式无效，请使用 YYYY-MM-DD 格式")
        sys.exit(1)

    print("=" * 80)
    print("🎯 批量质量分数分布可视化")
    print("=" * 80)
    print(f"📂 Stage2输出目录: {args.stage_2_output}")
    print(f"📅 分析日期: {args.date}")
    print(f"📁 输出根目录: {args.output_root}")
    print(f"📊 Bin宽度: {args.bin_width} ({args.bin_width * 100:.2f}%)")
    print(f"🔍 分离度分析: {'是' if args.analyze else '否'}")

    # 运行批量可视化
    batch_visualize_quality_distribution(
        stage2_output=args.stage_2_output,
        date=args.date,
        output_root=args.output_root,
        bin_width=args.bin_width,
        analyze=args.analyze,
    )


if __name__ == "__main__":
    main()
