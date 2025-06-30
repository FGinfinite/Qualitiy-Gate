import argparse
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel
import sys
import os

# Add project root to the Python path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.modeling import replace_moe_layers_with_trashcan

def main():
    """
    Loads a trained model checkpoint and inspects its architecture and weights.
    """
    parser = argparse.ArgumentParser(description="Inspect a trained model checkpoint.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the model checkpoint directory (the adapter).",
    )
    args = parser.parse_args()

    # Define the base model name
    base_model_name = "allenai/OLMoE-1B-7B-0125"

    print(f"Loading base model: {base_model_name}...")
    # Load the base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16
    )

    print("Rebuilding model with TrashCanMoE architecture...")
    # Rebuild the model with the TrashCanMoE architecture
    replace_moe_layers_with_trashcan(model, model.config)

    print(f"Loading PEFT adapter from: {args.checkpoint_path}...")
    # Load the PEFT adapter
    model = PeftModel.from_pretrained(model, args.checkpoint_path)

    print("Merging LoRA weights...")
    # Merge the LoRA weights for easier inspection
    model = model.merge_and_unload()

    print("\nModel loading and merging complete.")
    print("==================================================")

    print("\nFinal Model Architecture:")
    print(model)
    print("==================================================")

    print("\nModel Parameters (Name, Shape, Requires Grad):")
    total_params = 0
    for name, param in model.named_parameters():
        print(f"- {name:<80} | Shape: {str(param.shape):<25} | Requires Grad: {param.requires_grad}")
        total_params += param.numel()
    
    print("==================================================")
    print(f"Total number of parameters: {total_params / 1e6:.2f}M")
    print("==================================================")


if __name__ == "__main__":
    main()