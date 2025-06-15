# src/stages/selection.py
import json
import logging
import os
from typing import Dict, List, Tuple

import hydra
import torch
from datasets import concatenate_datasets, load_dataset
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_model_and_tokenizer(
    model_path: str,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Loads a pre-trained model and its tokenizer from a local checkpoint.
    """
    model_kwargs = {
        "low_cpu_mem_usage": True,
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def calculate_quality_score(
    logits: torch.Tensor, quality_expert_indices: List[int]
) -> float:
    """
    Calculates the quality score for a single data point based on router logits.
    """
    quality_logits = logits[quality_expert_indices]
    return torch.sum(quality_logits).item()


@hydra.main(
    config_path="../../configs", config_name="stage_2_selection", version_base=None
)
def select_data(cfg: DictConfig) -> None:
    """
    Main function for the data selection stage, managed by Hydra.
    """
    log = logging.getLogger(__name__)
    log.info("--- Starting Stage 2: Data Selection ---")

    # 1. Load Model and Tokenizer
    log.info(f"Loading model from checkpoint: {cfg.model_checkpoint_path}")
    model, tokenizer = get_model_and_tokenizer(cfg.model_checkpoint_path)
    num_experts = model.config.num_experts
    num_hidden_layers = model.config.num_hidden_layers

    # 2. Define Expert Roles
    num_trash_experts = int(num_experts * cfg.trash_expert_ratio)
    quality_expert_indices = list(range(num_trash_experts, num_experts))
    log.info(f"Total experts: {num_experts}")
    log.info(
        f"Trash experts: {num_trash_experts} (Indices 0 to {num_trash_experts - 1})"
    )
    log.info(
        f"Quality experts: {len(quality_expert_indices)} (Indices {num_trash_experts} to {num_experts - 1})"
    )

    # 3. Load and Prepare Dataset
    log.info(f"Loading datasets: {cfg.dataset.paths}")
    all_datasets = [load_dataset(path, split="train") for path in cfg.dataset.paths]
    dataset = concatenate_datasets(all_datasets).shuffle(seed=cfg.dataset.seed)

    if cfg.dataset.subset_ratio < 1.0:
        dataset = dataset.select(range(int(len(dataset) * cfg.dataset.subset_ratio)))
    log.info(f"Total samples to score: {len(dataset)}")

    def format_text(example: Dict) -> Dict:
        """Formats different dataset structures into a single text string."""
        if "instruction" in example and "response" in example:
            instruction = example.get("instruction", "")
            context = example.get("context", "")
            response = example.get("response", "")
            text = f"Instruction: {instruction}"
            if context:
                text += f"\nContext: {context}"
            text += f"\nResponse: {response}"
            return {"text": text}
        conversation_data = example.get("conversations") or example.get("conversation")
        if conversation_data:
            return {
                "text": "\n".join(
                    [
                        f"{turn.get('from', turn.get('role', ''))}: {turn.get('value', turn.get('content', ''))}"
                        for turn in conversation_data
                    ]
                )
            }
        if "text" in example:
            return {"text": example["text"]}
        raise ValueError(f"Unsupported dataset format: {example.keys()}")

    dataset = dataset.map(format_text, remove_columns=dataset.column_names)
    dataloader = DataLoader(dataset, batch_size=cfg.data_process.batch_size)

    # 4. Score Data
    scored_data = []
    log.info("Scoring dataset...")
    for i, batch in enumerate(tqdm(dataloader)):
        texts = batch["text"]
        if i == 0:
            log.info(
                f"Batch 0, first 3 texts:\n1: {texts[0][:100]}...\n2: {texts[1][:100]}...\n3: {texts[2][:100]}..."
            )

        inputs = tokenizer(
            texts,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            max_length=512,
        )
        inputs = {
            k: v.to(model.device)
            if k != "input_ids"
            else v.to(model.device, dtype=torch.long)
            for k, v in inputs.items()
        }

        with torch.no_grad():
            outputs = model(**inputs, output_router_logits=True)

        # FIX: Correctly reshape and average logits to ensure sample independence
        batch_size = len(texts)
        sequence_length = inputs["input_ids"].shape[1]

        all_router_logits = torch.stack(
            outputs.router_logits
        )  # Shape: (num_layers, bs * seq_len, num_experts)

        # Reshape to separate samples before averaging: (num_layers, bs, seq_len, num_experts)
        reshaped_logits_per_layer = all_router_logits.view(
            num_hidden_layers, batch_size, sequence_length, num_experts
        )

        # Average across layers, then across tokens for a robust sample score
        # Avg across layers -> (bs, seq_len, num_experts)
        avg_layer_logits = torch.mean(reshaped_logits_per_layer, dim=0)
        # Avg across sequence length -> (bs, num_experts)
        batch_logits = torch.mean(avg_layer_logits, dim=1)

        if i == 0:
            log.info(f"Batch 0, averaged logits (first 3 samples):\n{batch_logits[:3]}")

        for j in range(batch_size):
            quality_score = calculate_quality_score(
                batch_logits[j], quality_expert_indices
            )
            if i == 0 and j < 3:
                log.info(f"Sample {j} quality score: {quality_score}")
            scored_data.append(
                {
                    "source_dataset": list(cfg.dataset.paths),
                    "source_index": i * cfg.data_process.batch_size + j,
                    "text": texts[j],
                    "quality_score": quality_score,
                }
            )

    # 5. Filter and Save Data
    log.info("Filtering and saving data...")
    scored_data.sort(key=lambda x: x["quality_score"], reverse=True)
    num_to_select = int(len(scored_data) * cfg.selection_percentage)
    selected_data = scored_data[:num_to_select]

    log.info(
        f"Selected top {len(selected_data)} samples ({cfg.selection_percentage * 100:.2f}%)"
    )
    if len(selected_data) > 3:
        log.info(f"Top 3 scores: {[d['quality_score'] for d in selected_data[:3]]}")
        log.info(f"Bottom 3 scores: {[d['quality_score'] for d in selected_data[-3:]]}")

    # The output path is now directly managed by Hydra.
    # We get the absolute path to avoid any potential issues with relative paths.
    output_path = hydra.utils.to_absolute_path(cfg.output_path)
    log.info(f"Ensuring output directory exists for: {output_path}")
    
    # The directory is automatically created by Hydra, but we ensure it for safety.
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in selected_data:
            f.write(json.dumps(item) + "\n")

    log.info(f"Filtered data saved to: {output_path}")
    log.info("--- Stage 2: Data Selection Completed ---")


if __name__ == "__main__":
    select_data()
