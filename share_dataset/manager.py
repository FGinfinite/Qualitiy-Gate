# share_dataset/manager.py
"""
共享内存数据集服务器管理工具
"""

import logging
import os
import subprocess
import time
from typing import Dict, Optional


class DatasetServerManager:
    """数据集服务器管理器"""

    def __init__(self, config: Optional = None):
        from .client import SharedDatasetClient
        from .config import default_config

        self.config = config or default_config
        self.client = SharedDatasetClient(self.config)
        self.logger = logging.getLogger(__name__)

    def is_server_running(self) -> bool:
        """检查服务器是否运行中"""
        return self.client.is_server_available()

    def get_server_pid(self) -> Optional[int]:
        """获取服务器PID"""
        try:
            if os.path.exists(self.config.default_pid_file):
                with open(self.config.default_pid_file, "r") as f:
                    return int(f.read().strip())
        except Exception:
            pass
        return None

    def start_server(self, sample_limit: Optional[int] = None, background: bool = True, timeout: int = 1200) -> bool:
        """
        启动服务器

        Args:
            sample_limit: 样本数量限制
            background: 是否后台运行
            timeout: 启动超时时间

        Returns:
            bool: 启动是否成功
        """
        if self.is_server_running():
            self.logger.info("服务器已在运行中")
            return True

        try:
            self.logger.info("启动共享内存数据集服务器...")

            # 清理残留文件
            self._cleanup_stale_files()

            # 构建启动命令
            import sys

            cmd = [
                sys.executable,
                "-m",
                "share_dataset.server",
                "--mmap-file",
                self.config.default_mmap_file,
                "--pid-file",
                self.config.default_pid_file,
            ]

            if sample_limit:
                cmd.extend(["--sample-limit", str(sample_limit)])

            # 启动服务器进程
            if background:
                # 后台启动
                with open("/tmp/shared_dataset_server.log", "w") as log_file:
                    process = subprocess.Popen(
                        cmd,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        preexec_fn=os.setsid,  # 创建新进程组
                    )

                # 等待启动完成
                start_time = time.time()
                while time.time() - start_time < timeout:
                    if self.is_server_running():
                        self.logger.info("服务器启动成功")
                        return True
                    time.sleep(1)

                self.logger.error("服务器启动超时")
                return False
            else:
                # 前台启动（用于调试）
                process = subprocess.run(cmd)
                return process.returncode == 0

        except Exception as e:
            self.logger.error(f"启动服务器失败: {e}")
            return False

    def stop_server(self, timeout: int = 10) -> bool:
        """
        停止服务器

        Args:
            timeout: 停止超时时间

        Returns:
            bool: 停止是否成功
        """
        if not self.is_server_running():
            self.logger.info("服务器未运行")
            self._cleanup_stale_files()
            return True

        try:
            pid = self.get_server_pid()
            if not pid:
                self.logger.error("无法获取服务器PID")
                return False

            self.logger.info(f"停止服务器进程 {pid}...")

            # 尝试优雅关闭
            os.kill(pid, 15)  # SIGTERM

            # 等待进程结束
            start_time = time.time()
            while time.time() - start_time < timeout:
                if not self.is_server_running():
                    self.logger.info("服务器已停止")
                    return True
                time.sleep(0.5)

            # 强制关闭
            self.logger.warning("优雅关闭超时，强制终止进程")
            os.kill(pid, 9)  # SIGKILL

            # 清理文件
            self._cleanup_stale_files()

            return True

        except ProcessLookupError:
            self.logger.info("进程已不存在")
            self._cleanup_stale_files()
            return True
        except Exception as e:
            self.logger.error(f"停止服务器失败: {e}")
            return False

    def restart_server(self, sample_limit: Optional[int] = None) -> bool:
        """重启服务器"""
        self.logger.info("重启服务器...")
        if not self.stop_server():
            return False
        time.sleep(2)
        return self.start_server(sample_limit=sample_limit)

    def get_server_status(self) -> Dict:
        """获取服务器状态信息"""
        status = {
            "running": self.is_server_running(),
            "pid": self.get_server_pid(),
            "config": {
                "mmap_file": self.config.default_mmap_file,
                "pid_file": self.config.default_pid_file,
                "status_file": self.config.default_status_file,
            },
        }

        # 获取详细信息
        if status["running"]:
            try:
                info = self.client.get_server_info()
                status.update(info)
            except Exception as e:
                status["error"] = str(e)

        return status

    def _cleanup_stale_files(self):
        """清理残留文件"""
        files_to_clean = [
            self.config.default_pid_file,
            self.config.default_status_file,
        ]

        for file_path in files_to_clean:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    self.logger.debug(f"清理残留文件: {file_path}")
                except Exception as e:
                    self.logger.warning(f"清理文件失败 {file_path}: {e}")


def main():
    """管理器主入口"""
    import argparse

    # 设置日志
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="共享内存数据集服务器管理工具")
    parser.add_argument("action", choices=["start", "stop", "restart", "status"], help="管理操作")
    parser.add_argument("--sample-limit", type=int, help="样本数量限制")
    parser.add_argument("--foreground", action="store_true", help="前台运行")

    args = parser.parse_args()

    # 创建管理器
    manager = DatasetServerManager()

    # 执行操作
    if args.action == "start":
        success = manager.start_server(sample_limit=args.sample_limit, background=not args.foreground)
        exit(0 if success else 1)

    elif args.action == "stop":
        success = manager.stop_server()
        exit(0 if success else 1)

    elif args.action == "restart":
        success = manager.restart_server(sample_limit=args.sample_limit)
        exit(0 if success else 1)

    elif args.action == "status":
        status = manager.get_server_status()
        print("服务器状态:")
        for key, value in status.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
