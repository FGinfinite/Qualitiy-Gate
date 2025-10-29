#!/usr/bin/env python3
"""
生成faiss缓存文件脚本

用途：在指定目录生成faiss缓存文件
使用场景：
  - 测试faiss缓存文件生成
  - 测试faiss缓存文件使用
  - 模拟faiss缓存文件使用情况
"""

import argparse
import os
from pathlib import Path


def parse_size(size_str: str) -> int:
    """
    解析文件大小字符串

    Args:
        size_str: 大小字符串，如 "10MB", "1GB", "500KB"

    Returns:
        字节数

    Examples:
        >>> parse_size("10MB")
        10485760
        >>> parse_size("1GB")
        1073741824
    """
    size_str = size_str.upper().strip()
    # 按单位长度从长到短排序，避免 "GB" 被 "B" 先匹配
    units = [
        ("TB", 1024**4),
        ("GB", 1024**3),
        ("MB", 1024**2),
        ("KB", 1024),
        ("B", 1),
    ]

    # 提取数字和单位
    for unit, multiplier in units:
        if size_str.endswith(unit):
            try:
                number = float(size_str[: -len(unit)])
                return int(number * multiplier)
            except ValueError:
                raise ValueError(f"无效的大小格式: {size_str}")

    # 如果没有单位，假设是字节
    try:
        return int(size_str)
    except ValueError:
        raise ValueError(f"无效的大小格式: {size_str}")


def format_size(size_bytes: int) -> str:
    """
    格式化字节数为可读字符串

    Args:
        size_bytes: 字节数

    Returns:
        可读的大小字符串
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def generate_trash_files(
    output_dir: str,
    total_size: str,
    num_files: int,
):
    """
    生成垃圾文件

    Args:
        output_dir: 输出目录
        total_size: 总大小（如 "1GB"）
        num_files: 文件数量
    """
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 查找已有文件的最大序号
    existing_files = list(output_path.glob("faiss_cache_*.pt"))
    start_index = 0

    if existing_files:
        # 提取所有序号
        indices = []
        for f in existing_files:
            try:
                # 文件名格式: faiss_cache_N.pt
                index = int(f.stem.split("_")[-1])
                indices.append(index)
            except (ValueError, IndexError):
                continue

        if indices:
            start_index = max(indices) + 1
            print(f"🔍 发现 {len(existing_files)} 个已有文件，最大序号: {max(indices)}")
            print(f"📌 从序号 {start_index} 开始生成新文件")
    else:
        print("📌 未发现已有文件，从序号 0 开始生成")

    # 解析大小
    total_bytes = parse_size(total_size)
    avg_file_size = total_bytes // num_files

    print(f"📁 输出目录: {output_path.absolute()}")
    print(f"📊 目标总大小: {format_size(total_bytes)}")
    print(f"📄 文件数量: {num_files}")
    print(f"📏 每个文件大小: {format_size(avg_file_size)}")
    print("-" * 60)

    total_written = 0
    files_created = 0

    try:
        for i in range(num_files):
            # 计算当前文件大小（固定平均值，最后一个文件写入剩余空间）
            if i == num_files - 1:
                file_size = total_bytes - total_written
            else:
                file_size = avg_file_size

            # 生成文件名（使用 start_index 偏移）
            file_index = start_index + i
            filename = f"faiss_cache_{file_index}.pt"
            filepath = output_path / filename

            # 生成并写入数据
            print(f"⚙️  生成 {filename} ({format_size(file_size)})...", end=" ", flush=True)

            # 对于大文件，分块写入以避免内存问题
            chunk_size = 100 * 1024 * 1024  # 100MB chunks
            with open(filepath, "wb") as f:
                written = 0
                while written < file_size:
                    chunk = min(chunk_size, file_size - written)
                    data = os.urandom(chunk)
                    f.write(data)
                    written += chunk

            total_written += file_size
            files_created += 1
            print("✅")

            # 检查是否已达到目标
            if total_written >= total_bytes:
                break

    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")

    print("-" * 60)
    print(f"✅ 完成! 创建 {files_created} 个文件")
    print(f"📊 总大小: {format_size(total_written)} / {format_size(total_bytes)}")
    print(f"📁 位置: {output_path.absolute()}")


def main():
    parser = argparse.ArgumentParser(
        description="生成垃圾文件以占用磁盘空间 (固定文件名: faiss_cache_0.pt ~ faiss_cache_N.pt)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 生成总共 1GB 的垃圾文件，分为 100 个文件 (默认输出到 index_cache/)
  python generate_trash.py -s 1GB -n 100

  # 生成 100GB，分为 100 个文件
  python generate_trash.py -s 100GB -n 100

  # 生成 500MB，分为 10 个文件，保存到自定义目录
  python generate_trash.py -s 500MB -n 10 -o /data/cache
        """,
    )

    parser.add_argument(
        "-s",
        "--size",
        type=str,
        required=True,
        help="总大小 (如: 1GB, 500MB, 100GB)",
    )

    parser.add_argument(
        "-n",
        "--num-files",
        type=int,
        required=True,
        help="生成的文件数量",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="outputs/index_cache",
        help="输出目录 (默认: outputs/index_cache)",
    )

    args = parser.parse_args()

    # 生成垃圾文件
    generate_trash_files(
        output_dir=args.output,
        total_size=args.size,
        num_files=args.num_files,
    )


if __name__ == "__main__":
    main()
