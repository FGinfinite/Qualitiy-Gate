import os
import sys

import hydra
from omegaconf import DictConfig

# 获取当前脚本所在目录的父目录（即项目根目录）
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)  # 将项目根目录添加到sys.path的最前面

from src.stages import pretrain, selection, finetune


@hydra.main(config_path="../configs", config_name="stage_1_pretrain", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entry point for the application.
    Selects and runs the appropriate stage based on the configuration.
    """
    if cfg.stage == "pretrain":
        pretrain.train(cfg)
    elif cfg.stage == "selection":
        selection.select(cfg)
    elif cfg.stage == "finetune":
        finetune.train(cfg)
    elif cfg.stage == "evaluate":
        # evaluate.run(cfg)  # Placeholder for the evaluation stage
        pass
    else:
        raise ValueError(f"Unknown stage: {cfg.stage}")


if __name__ == "__main__":
    main()
