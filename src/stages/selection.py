# src/stages/selection.py
import json
import logging
import os
from typing import Dict, List, Tuple

import hydra
import torch
from datasets import concatenate_datasets, load_dataset
from omegaconf import DictConfig
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, OlmoeForCausalLM

from ..modeling import replace_moe_layers_with_trashcan


def get_model_and_tokenizer(
    cfg: DictConfig, device: torch.device
) -> Tuple[OlmoeForCausalLM, AutoTokenizer]:
    """
    Loads a pre-trained model and its tokenizer, adapts it to the TrashCanMoE architecture,
    and then applies the PEFT adapter for inference.
    """
    model_kwargs = {
        "low_cpu_mem_usage": True,
        "torch_dtype": torch.bfloat16,
    }
    # 步骤 1: 从原始路径加载基础模型
    base_model_name = cfg.selector_model.name
    model = OlmoeForCausalLM.from_pretrained(base_model_name, **model_kwargs)

    # 步骤 2: 适配模型架构以匹配训练时的结构
    replace_moe_layers_with_trashcan(model, model.config, cfg)

    # Manually update the model's config to reflect the new total number of experts
    original_num_experts = model.config.num_experts
    num_trash_experts = model.config.num_experts_per_tok
    model.config.num_experts = original_num_experts + num_trash_experts

    # 步骤 3: 加载 PEFT 适配器并为推理进行合并
    model = PeftModel.from_pretrained(model, cfg.model_checkpoint_path)
    model = model.merge_and_unload()
    model.to(device)

    # 步骤 4: 从基础模型的路径加载分词器
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
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


def select(cfg: DictConfig) -> None:
    """
    Main function for the data selection stage, managed by Hydra.
    """
    log = logging.getLogger(__name__)
    log.info("--- Starting Stage 2: Data Selection ---")

    # Determine the target device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # 1. Load Model and Tokenizer
    log.info(f"Loading and adapting model from checkpoint: {cfg.model_checkpoint_path}")
    model, tokenizer = get_model_and_tokenizer(cfg, device)
    num_hidden_layers = model.config.num_hidden_layers

    # 2. Define Expert Roles based on the TrashCanMoE architecture
    original_num_experts = model.config.num_experts
    num_trash_experts = (
        model.config.num_experts_per_tok
    )  # Trash experts are the k new ones
    total_experts = original_num_experts + num_trash_experts
    quality_expert_indices = list(range(original_num_experts))

    log.info(f"Total experts in router: {total_experts}")
    log.info(
        f"Original (quality) experts: {original_num_experts} (Indices 0 to {original_num_experts - 1})"
    )
    log.info(
        f"Added trash experts: {num_trash_experts} (Indices {original_num_experts} to {total_experts - 1})"
    )
    trash_expert_indices = list(range(original_num_experts, total_experts))

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
            k: v.to(device) if k != "input_ids" else v.to(device, dtype=torch.long)
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
            num_hidden_layers, batch_size, sequence_length, total_experts
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
            if i == 0 and j < 5:
                log.info(
                    f"--- Detailed Log for Sample {i * cfg.data_process.batch_size + j} ---"
                )
                log.info(f"  - Avg Logits Shape: {batch_logits[j].shape}")

                quality_logits_values = batch_logits[j][quality_expert_indices]
                log.info(
                    f"  - Quality Expert Logits (Indices {quality_expert_indices[0]}-{quality_expert_indices[-1]}): {quality_logits_values.tolist()}"
                )

                trash_logits_values = batch_logits[j][trash_expert_indices]
                log.info(
                    f"  - Trash Expert Logits (Indices {trash_expert_indices[0]}-{trash_expert_indices[-1]}): {trash_logits_values.tolist()}"
                )

                log.info(
                    f"  - Calculated Quality Score (Sum of Quality Logits): {quality_score}"
                )
                log.info(
                    "-" * (20 + len(str(i * cfg.data_process.batch_size + j)))
                )  # Separator
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
