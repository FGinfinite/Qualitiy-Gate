# share_dataset/client.py
"""
共享内存数据集客户端接口
"""

import logging
import mmap
import os
import pickle
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .config import SharedMemoryConfig, default_config


class LoadResult:
    """加载结果状态码"""

    SUCCESS = "success"
    SERVER_UNAVAILABLE = "server_unavailable"
    DATASET_NOT_FOUND = "dataset_not_found"
    LOAD_FAILED = "load_failed"


@dataclass
class DatasetLoadResult:
    """数据集加载结果"""

    status: str
    dataset: Optional[Any] = None
    message: str = ""
    load_time: float = 0.0


class SharedDatasetClient:
    """共享内存数据集客户端"""

    def __init__(self, config: Optional[SharedMemoryConfig] = None):
        self.config = config or default_config
        self.logger = logging.getLogger(__name__)

    def is_server_available(self) -> bool:
        """检查服务器是否可用"""
        try:
            # 检查PID文件
            if not os.path.exists(self.config.default_pid_file):
                return False

            # 检查进程是否存在
            with open(self.config.default_pid_file, "r") as f:
                pid = int(f.read().strip())

            # 发送信号0检查进程存在性
            try:
                os.kill(pid, 0)
            except OSError:
                return False

            # 检查内存映射文件
            if not os.path.exists(self.config.default_mmap_file):
                return False

            # 检查状态文件
            if os.path.exists(self.config.default_status_file):
                with open(self.config.default_status_file, "r") as f:
                    status = f.read().strip()
                    if status != "ready":
                        return False

            return True

        except Exception as e:
            self.logger.debug(f"检查服务器状态失败: {e}")
            return False

    def load_shared_dataset(self, dataset_config: Dict, timeout: Optional[int] = None) -> DatasetLoadResult:
        """
        从共享内存加载数据集

        Args:
            dataset_config: 数据集配置字典，包含name等信息
            timeout: 超时时间，覆盖默认配置

        Returns:
            DatasetLoadResult: 加载结果
        """
        start_time = time.time()
        timeout = timeout or self.config.server_timeout

        # 检查服务器是否可用
        if not self.is_server_available():
            return DatasetLoadResult(status=LoadResult.SERVER_UNAVAILABLE, message="共享内存服务器不可用", load_time=time.time() - start_time)

        try:
            # 等待服务器准备就绪
            ready_time = 0
            while ready_time < timeout:
                if os.path.exists(self.config.default_status_file):
                    with open(self.config.default_status_file, "r") as f:
                        status = f.read().strip()
                        if status == "ready":
                            break
                time.sleep(0.1)
                ready_time += 0.1
            else:
                return DatasetLoadResult(status=LoadResult.SERVER_UNAVAILABLE, message="服务器准备超时", load_time=time.time() - start_time)

            # 加载共享数据集
            dataset = self._load_from_mmap()
            if dataset is None:
                return DatasetLoadResult(status=LoadResult.LOAD_FAILED, message="从内存映射加载数据集失败", load_time=time.time() - start_time)

            # 验证数据集是否匹配请求的配置
            dataset_name = dataset_config.get("name", "")
            if not self._validate_dataset(dataset, dataset_name):
                return DatasetLoadResult(
                    status=LoadResult.DATASET_NOT_FOUND, message=f"数据集 {dataset_name} 不匹配共享数据集", load_time=time.time() - start_time
                )

            load_time = time.time() - start_time
            self.logger.info(f"从共享内存成功加载数据集，耗时: {load_time:.4f}秒")

            return DatasetLoadResult(status=LoadResult.SUCCESS, dataset=dataset, message=f"成功加载 {len(dataset)} 个样本", load_time=load_time)

        except Exception as e:
            self.logger.error(f"从共享内存加载数据集失败: {e}")
            return DatasetLoadResult(status=LoadResult.LOAD_FAILED, message=f"加载失败: {str(e)}", load_time=time.time() - start_time)

    def _load_from_mmap(self) -> Optional[Any]:
        """从内存映射文件加载数据集"""
        try:
            with open(self.config.default_mmap_file, "rb") as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    dataset = pickle.load(mm)
                    return dataset
        except Exception as e:
            self.logger.error(f"内存映射加载失败: {e}")
            return None

    def _validate_dataset(self, dataset: Any, expected_name: str) -> bool:
        """验证数据集是否匹配期望"""
        try:
            # 简单验证：检查是否包含openhermes
            if "openhermes" in expected_name.lower():
                # 对于OpenHermes数据集，检查是否有messages字段
                if hasattr(dataset, "__getitem__") and len(dataset) > 0:
                    sample = dataset[0]
                    if isinstance(sample, dict) and "messages" in sample:
                        return True
            return True  # 默认通过验证
        except Exception:
            return False

    def get_server_info(self) -> Dict:
        """获取服务器信息"""
        info = {
            "available": self.is_server_available(),
            "mmap_file": self.config.default_mmap_file,
            "pid_file": self.config.default_pid_file,
            "status_file": self.config.default_status_file,
        }

        if info["available"]:
            try:
                # 获取PID
                with open(self.config.default_pid_file, "r") as f:
                    info["pid"] = int(f.read().strip())

                # 获取内存映射文件大小
                if os.path.exists(self.config.default_mmap_file):
                    size = os.path.getsize(self.config.default_mmap_file)
                    info["mmap_size_mb"] = size / (1024 * 1024)

                # 获取状态
                if os.path.exists(self.config.default_status_file):
                    with open(self.config.default_status_file, "r") as f:
                        info["status"] = f.read().strip()
            except Exception as e:
                info["error"] = str(e)

        return info
