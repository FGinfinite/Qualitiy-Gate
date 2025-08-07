# src/utils/logging_utils.py
"""
日志工具模块

提供统一的日志记录功能，包括：
- HydraLoggingCallback: 将Trainer日志路由到Hydra日志系统
- 全局日志获取函数
- 日志配置工具函数
"""

import logging
from typing import Optional
from transformers import TrainerCallback, logging as transformers_logging


class HydraLoggingCallback(TrainerCallback):
    """自定义回调，将Trainer日志路由到Hydra日志系统"""

    def __init__(self, logger_name: Optional[str] = None):
        """
        初始化回调

        Args:
            logger_name: 可选的日志器名称，如果不提供则使用默认的__name__
        """
        if logger_name:
            self.logger = logging.getLogger(logger_name)
        else:
            self.logger = get_logger(__name__)

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """训练过程中发生日志记录时调用"""
        if logs is not None:
            # 为Hydra日志系统格式化训练指标
            log_message = []
            for key, value in logs.items():
                if isinstance(value, float):
                    log_message.append(f"{key}: {value:.4f}")
                else:
                    log_message.append(f"{key}: {value}")

            # 记录到Hydra日志
            if log_message:
                self.logger.info(f"步骤 {state.global_step}: {', '.join(log_message)}")

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self.logger.info(f"训练开始 - 总步数: {state.max_steps}, 训练轮数: {args.num_train_epochs}")

    def on_train_end(self, args, state, control, model=None, **kwargs):
        self.logger.info("训练完成")

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        self.logger.info(f"完成第 {int(state.epoch)} 轮训练")


def get_logger(name: str = __name__) -> logging.Logger:
    """
    获取配置好的日志器实例

    Args:
        name: 日志器名称

    Returns:
        配置好的日志器实例
    """
    return logging.getLogger(name)


def configure_transformers_logging(hydra_logger: Optional[logging.Logger] = None) -> logging.Logger:
    """
    配置transformers日志以与Hydra日志系统集成

    Args:
        hydra_logger: 可选的Hydra日志器实例，如果不提供则使用根日志器

    Returns:
        配置好的transformers日志器
    """
    # 设置适当的详细程度
    transformers_logging.set_verbosity_info()

    # 启用默认处理程序和显式格式化
    transformers_logging.enable_default_handler()
    transformers_logging.enable_explicit_format()

    if hydra_logger is None:
        hydra_logger = logging.getLogger()

    # 获取transformers日志器
    transformers_logger = transformers_logging.get_logger("transformers.trainer")

    # 清除现有处理程序以避免重复日志
    transformers_logger.handlers.clear()

    # 将Hydra的处理程序添加到transformers日志器
    for handler in hydra_logger.handlers:
        transformers_logger.addHandler(handler)

    # 设置与Hydra日志器相同的级别
    transformers_logger.setLevel(hydra_logger.level)

    return transformers_logger


def setup_training_logging(logger_name: str = __name__) -> tuple[logging.Logger, HydraLoggingCallback]:
    """
    设置训练日志的便捷函数

    Args:
        logger_name: 日志器名称

    Returns:
        (logger, callback)的元组
    """
    logger = get_logger(logger_name)

    # 配置transformers日志
    configure_transformers_logging(logger)

    # 创建自定义回调
    callback = HydraLoggingCallback(logger_name)

    return logger, callback


# 便捷的全局日志器实例
def info(message: str, logger_name: str = __name__):
    """记录info级别消息的便捷函数"""
    get_logger(logger_name).info(message)


def warning(message: str, logger_name: str = __name__):
    """记录warning级别消息的便捷函数"""
    get_logger(logger_name).warning(message)


def error(message: str, logger_name: str = __name__):
    """记录error级别消息的便捷函数"""
    get_logger(logger_name).error(message)


def debug(message: str, logger_name: str = __name__):
    """记录debug级别消息的便捷函数"""
    get_logger(logger_name).debug(message)
