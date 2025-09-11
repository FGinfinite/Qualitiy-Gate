# share_dataset/__init__.py
"""
共享内存数据集加载模块

提供基于内存映射的高性能数据集共享方案，支持多进程间快速数据集访问。
主要用于加速HuggingFace数据集的重复加载。
"""

from .client import LoadResult, SharedDatasetClient
from .config import SharedMemoryConfig
from .manager import DatasetServerManager

__all__ = [
    "SharedDatasetClient",
    "DatasetServerManager",
    "LoadResult",
    "SharedMemoryConfig",
]
