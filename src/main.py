import os
import sys

import hydra
from omegaconf import DictConfig

# 获取当前脚本所在目录的父目录（即项目根目录）
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)  # 将项目根目录添加到sys.path的最前面

from src.stages import pretrain


@hydra.main(config_path="../configs", config_name="stage_1_pretrain", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entry point for the application.
    Selects and runs the appropriate stage based on the configuration.
    """
    stage = cfg.get("stage", "pretrain")  # Default to pretrain if not specified

    if stage == "pretrain":
        pretrain(cfg)
    elif stage == "select":
        pass
    elif stage == "finetune":
        pass
    elif stage == "evaluate":
        pass
    else:
        raise ValueError(f"Unknown stage: {stage}")


if __name__ == "__main__":
    main()
