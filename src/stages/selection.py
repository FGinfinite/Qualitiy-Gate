import json
import torch
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

def calculate_quality_score(router_logits: torch.Tensor, good_expert_indices: list[int]) -> float:
    """Calculates the quality score based on the logits of good experts."""
    good_expert_logits = router_logits[good_expert_indices]
    return torch.sum(good_expert_logits).item()

def select_data(cfg: DictConfig) -> None:
    """
    Selects data based on the quality score from a pre-trained selector model.
    """
    print("--- Starting Stage 2: Data Selection ---")

    # Load Model and Tokenizer
    print(f"Loading model from checkpoint: {cfg.selector_model.checkpoint_path}")
    model = AutoModelForCausalLM.from_pretrained(cfg.selector_model.checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg.selector_model.name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Define Expert Roles
    num_experts = model.config.num_experts
    num_trash_experts = int(num_experts * cfg.data_selection.trash_expert_ratio)
    good_expert_indices = list(range(num_trash_experts, num_experts))
    print(f"Total experts: {num_experts}, Good experts: {len(good_expert_indices)}, Trash experts: {num_trash_experts}")

    # Load Dataset
    print(f"Loading target dataset from: {cfg.data_selection.target_dataset_path}")
    # This is a placeholder for actual dataset loading.
    # We'll create a dummy dataset for now.
    dataset = load_dataset("text", data_files={"train": cfg.data_selection.target_dataset_path})['train']

    # Score Data
    scored_data = []
    for i, example in enumerate(tqdm(dataset, desc="Scoring data")):
        inputs = tokenizer(example['text'], return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_router_logits=True)
        
        # Assuming the router logits are the last element in the tuple
        router_logits = outputs.router_logits[-1].squeeze()
        
        quality_score = calculate_quality_score(router_logits, good_expert_indices)
        
        scored_data.append({
            "source_dataset": cfg.data_selection.target_dataset_path,
            "source_index": i,
            "text": example['text'],
            "quality_score": quality_score
        })

    # Sort and Select
    scored_data.sort(key=lambda x: x['quality_score'], reverse=True)
    num_to_select = int(len(scored_data) * cfg.data_selection.selection_percentage)
    selected_data = scored_data[:num_to_select]
    print(f"Selected {len(selected_data)} samples out of {len(scored_data)}")

    # Save Results
    print(f"Saving selected data to: {cfg.output.selected_data_path}")
    with open(cfg.output.selected_data_path, 'w') as f:
        for item in selected_data:
            f.write(json.dumps(item) + "\n")
            
    print("\n--- Stage 2: Data Selection Completed ---")