import json
import logging
from pathlib import Path

import hydra
import numpy as np
from lm_eval.evaluator import simple_evaluate
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def make_json_serializable(obj):
    """
    递归地将对象中不可序列化的Numpy类型转换为JSON兼容的Python原生类型。
    """
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.dtype):
        return str(obj)
    try:
        json.dumps(obj)
        return obj
    except (TypeError, OverflowError):
        return str(obj)


def save_results(results: dict, output_path: str | Path):
    """将评估结果保存到指定的JSON文件。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serializable_results = make_json_serializable(results)

    log.info(f"将评估结果保存到 '{output_path}'")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=4, ensure_ascii=False)
    log.info("评估结果已成功保存。")


@hydra.main(version_base=None, config_path="../../configs", config_name="stage_4_evaluate")
def main(cfg: DictConfig) -> None:
    """
    使用lm-evaluation-harness执行模型评估的主函数。
    不再手动加载模型，而是将模型参数委托给lm-eval处理。
    """
    log.info("======== 开始第四阶段：模型评估 ========")

    # 将hydra配置中的模型参数转换为字典
    model_args = OmegaConf.to_container(cfg.eval.model, resolve=True)

    log.info("使用 lm-eval-harness 评估模型...")
    log.info(f"模型参数: {model_args}")
    log.info(f"任务: {cfg.eval.tasks}")
    log.info(f"批处理大小: {cfg.eval.batch_size}")
    log.info(f"设备: {cfg.eval.device}")

    # 调用simple_evaluate，让它处理模型加载
    results = simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=list(cfg.eval.tasks),
        batch_size=cfg.eval.batch_size,
        device=cfg.eval.device,
        log_samples=True,
    )

    save_results(results, cfg.output.results_path)
    log.info("======== 第四阶段：模型评估完成 ========")


if __name__ == "__main__":
    main()
