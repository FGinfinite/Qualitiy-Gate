# share_dataset/server.py
"""
共享内存数据集服务器实现
"""

import logging
import os
import pickle
import signal
import sys
import time
from functools import partial
from typing import Optional

from datasets import Dataset, load_dataset


class SharedDatasetServer:
    """共享内存数据集服务器"""

    def __init__(self, config: Optional = None):
        from .config import default_config

        self.config = config or default_config
        self.logger = logging.getLogger(__name__)
        self.dataset = None
        self.mmap_file = None
        self.running = False
        self.setup_signal_handlers()

    def setup_signal_handlers(self):
        """设置信号处理器"""
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        """信号处理器"""
        self.logger.info(f"收到信号 {signum}，准备关闭服务器...")
        self.cleanup()
        sys.exit(0)

    def load_openhermes_dataset(self, sample_limit: Optional[int] = None) -> Dataset:
        """
        加载OpenHermes-2.5数据集

        Args:
            sample_limit: 样本数量限制，None表示使用完整数据集

        Returns:
            加载的数据集
        """
        self.logger.info("开始加载OpenHermes-2.5数据集...")

        try:
            # 加载数据集
            dataset = load_dataset("teknium/OpenHermes-2.5", split="train")
            self.logger.info(f"原始数据集大小: {len(dataset)}")

            # 限制样本数量（用于测试）
            if sample_limit and sample_limit < len(dataset):
                dataset = dataset.select(range(sample_limit))
                self.logger.info(f"限制数据集大小为: {len(dataset)}")

            # 转换格式
            convert_fn = partial(self.convert_openhermes_format)
            dataset = dataset.map(
                convert_fn,
                desc="转换OpenHermes格式",
                load_from_cache_file=True,
            )

            self.logger.info(f"数据集格式转换完成，最终大小: {len(dataset)}")
            return dataset

        except Exception as e:
            self.logger.error(f"加载OpenHermes数据集失败: {e}")
            raise

    def convert_openhermes_format(self, example):
        """转换OpenHermes格式到项目标准格式"""
        if "conversations" not in example:
            raise ValueError("OpenHermes数据必须包含'conversations'字段")

        messages = []
        for msg in example["conversations"]:
            # 角色映射
            role_map = {"human": "user", "gpt": "assistant", "system": "system"}

            if msg["from"] not in role_map:
                continue

            role = role_map[msg["from"]]
            content = msg["value"]

            # 过滤空的system消息
            if role == "system" and not content.strip():
                continue

            messages.append({"role": role, "content": content.strip() if content else ""})

        if not messages:
            raise ValueError("转换后没有有效的消息")

        # 生成唯一ID
        example_id = f"openhermes_{example.get('id', hash(str(example)) % 1000000)}"

        return {"dataset": "openhermes", "id": example_id, "messages": messages}

    def create_memory_mapping(self):
        """创建内存映射文件"""
        try:
            self.logger.info("开始创建内存映射文件...")

            # 清理已存在的文件
            if os.path.exists(self.config.default_mmap_file):
                os.remove(self.config.default_mmap_file)

            # 序列化数据集
            serialized_data = pickle.dumps(self.dataset)
            data_size = len(serialized_data)

            self.logger.info(f"数据集序列化完成，大小: {data_size / (1024 * 1024):.2f} MB")

            # 写入内存映射文件
            with open(self.config.default_mmap_file, "wb") as f:
                f.write(serialized_data)

            self.logger.info(f"内存映射文件创建完成: {self.config.default_mmap_file}")

        except Exception as e:
            self.logger.error(f"创建内存映射文件失败: {e}")
            raise

    def write_pid_file(self):
        """写入PID文件"""
        with open(self.config.default_pid_file, "w") as f:
            f.write(str(os.getpid()))

    def write_status_file(self, status: str):
        """写入状态文件"""
        with open(self.config.default_status_file, "w") as f:
            f.write(status)

    def cleanup(self):
        """清理资源"""
        self.logger.info("开始清理服务器资源...")

        files_to_clean = [
            self.config.default_pid_file,
            self.config.default_status_file,
            self.config.default_mmap_file,
        ]

        for file_path in files_to_clean:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    self.logger.info(f"已删除文件: {file_path}")
                except Exception as e:
                    self.logger.error(f"删除文件失败 {file_path}: {e}")

        self.running = False
        self.logger.info("服务器清理完成")

    def start(self, sample_limit: Optional[int] = None):
        """启动服务器"""
        try:
            self.logger.info("启动共享内存数据集服务器...")

            # 写入状态文件
            self.write_status_file("loading")

            # 写入PID文件
            self.write_pid_file()

            # 加载数据集
            sample_limit = sample_limit or self.config.sample_limit
            self.dataset = self.load_openhermes_dataset(sample_limit)

            # 创建内存映射
            self.create_memory_mapping()

            # 更新状态为就绪
            self.write_status_file("ready")

            self.running = True
            self.logger.info("服务器启动成功，等待客户端连接...")

            # 保持服务器运行
            try:
                while self.running:
                    time.sleep(1)
            except KeyboardInterrupt:
                self.logger.info("收到键盘中断...")

        except Exception as e:
            self.logger.error(f"服务器启动失败: {e}")
            self.cleanup()
            raise
        finally:
            self.cleanup()


def main():
    """服务器主入口"""
    import argparse

    from .config import SharedMemoryConfig

    # 设置日志
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="共享内存数据集服务器")
    parser.add_argument("--sample-limit", type=int, help="样本数量限制")
    parser.add_argument("--mmap-file", help="内存映射文件路径")
    parser.add_argument("--pid-file", help="PID文件路径")

    args = parser.parse_args()

    # 创建配置
    config = SharedMemoryConfig()
    if args.mmap_file:
        config.default_mmap_file = args.mmap_file
    if args.pid_file:
        config.default_pid_file = args.pid_file

    # 启动服务器
    server = SharedDatasetServer(config)
    server.start(sample_limit=args.sample_limit)


if __name__ == "__main__":
    main()
