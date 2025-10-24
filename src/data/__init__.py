# src/data/__init__.py
"""
数据处理模块

提供数据加载、编码、格式转换和统计工具。
"""

# 核心加载功能
# 数据工具
from .data_utils import compute_string_lengths, get_data_statistics, sort_dataset_by_string_length, temp_seed
from .dataset_loader import load_and_prepare_dataset, load_selected_data, load_single_dataset

# 编码功能
from .encoders import encode_data, encode_with_messages_format

# 格式转换
from .format_converters import get_format_converter, infer_format_type

__all__ = [
    # 加载功能
    "load_and_prepare_dataset",
    "load_selected_data",
    "load_single_dataset",
    # 编码功能
    "encode_data",
    "encode_with_messages_format",
    # 数据工具
    "get_data_statistics",
    "temp_seed",
    "compute_string_lengths",
    "sort_dataset_by_string_length",
    # 格式转换
    "get_format_converter",
    "infer_format_type",
]
