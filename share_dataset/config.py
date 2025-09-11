# share_dataset/config.py
"""
共享内存配置管理
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class SharedMemoryConfig:
    """共享内存配置类"""

    # 默认文件路径
    default_mmap_file: str = "/tmp/shared_dataset.mmap"
    default_pid_file: str = "/tmp/shared_dataset_server.pid"
    default_status_file: str = "/tmp/shared_dataset_server.status"

    # 服务器配置
    server_timeout: int = 30  # 服务器响应超时时间（秒）
    auto_start: bool = False  # 是否自动启动服务器

    # 数据集配置
    sample_limit: Optional[int] = None  # 样本数量限制，None表示使用完整数据集

    def __post_init__(self):
        """从环境变量覆盖配置"""
        self.default_mmap_file = os.getenv("SHARED_DATASET_MMAP_FILE", self.default_mmap_file)
        self.default_pid_file = os.getenv("SHARED_DATASET_PID_FILE", self.default_pid_file)
        self.default_status_file = os.getenv("SHARED_DATASET_STATUS_FILE", self.default_status_file)

        if os.getenv("SHARED_DATASET_TIMEOUT"):
            self.server_timeout = int(os.getenv("SHARED_DATASET_TIMEOUT"))

        if os.getenv("SHARED_DATASET_AUTO_START"):
            self.auto_start = os.getenv("SHARED_DATASET_AUTO_START").lower() == "true"

        if os.getenv("SHARED_DATASET_SAMPLE_LIMIT"):
            self.sample_limit = int(os.getenv("SHARED_DATASET_SAMPLE_LIMIT"))


# 全局配置实例
default_config = SharedMemoryConfig()
