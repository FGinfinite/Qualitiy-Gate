import json
import logging
from pathlib import Path

import torch
from omegaconf import DictConfig
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval import simple_evaluate

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_model(cfg: DictConfig) -> None:
    """
    Evaluates the performance of a fine-tuned model by merging LoRA weights
    and running lm-eval-harness.

    Args:
        cfg: Hydra configuration object containing evaluation parameters.
    """
    logger.info("🚀 Starting Stage 4: Model Evaluation")

    # 1. Load the base model and tokenizer
    logger.info(f"Loading base model from: {cfg.eval.model.base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.eval.model.base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.eval.model.base_model_path)
    logger.info("Base model and tokenizer loaded successfully.")

    # 2. Load and merge the LoRA adapter
    logger.info(f"Loading LoRA adapter from: {cfg.eval.model.adapter_path}")
    model = PeftModel.from_pretrained(base_model, cfg.eval.model.adapter_path)
    model = model.merge_and_unload()
    logger.info("✅ Successfully merged LoRA adapter into the base model.")

    # 3. Run evaluation using lm-eval-harness
    tasks = list(cfg.eval.tasks)
    logger.info(f"Running evaluation on tasks: {tasks}")
    
    results = simple_evaluate(
        model=model,
        tokenizer=tokenizer,
        tasks=tasks,
        batch_size=cfg.eval.batch_size,
        device="auto", # lm-eval will handle device placement
        limit=cfg.eval.limit if "limit" in cfg.eval else None,
    )

    # 4. Process and save the results
    output_path = Path(cfg.output.results_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"💾 Saving evaluation results to: {output_path}")

    # Convert results to a serializable format
    dumped_results = json.dumps(results, indent=2, default=str)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(dumped_results)

    logger.info(f"📝 Evaluation results:\n{json.dumps(results['results'], indent=2)}")
    logger.info("🏁 Stage 4: Model Evaluation finished.")