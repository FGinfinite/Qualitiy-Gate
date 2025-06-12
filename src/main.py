import hydra
from omegaconf import DictConfig

from src.stages import (
    pretrain,
    select_data,
    finetune_target_model,
    evaluate_model
)


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
        select_data(cfg)
    elif stage == "finetune":
        finetune_target_model(cfg)
    elif stage == "evaluate":
        evaluate_model(cfg)
    else:
        raise ValueError(f"Unknown stage: {stage}")


if __name__ == "__main__":
    main()