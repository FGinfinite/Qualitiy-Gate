# scripts/compare_router_weights.py

import os

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM


def load_model(model_path, model_kwargs, is_from_hub=False):
    """
    Loads a model from a given path.
    If is_from_hub is True, loads from Hugging Face Hub.
    Otherwise, loads from a local path and checks for existence.
    """
    if not is_from_hub and not os.path.exists(model_path):
        print(f"Error: Local model path not found at '{model_path}'")
        return None
    
    print(f"Loading model from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    return model

def get_router_weights(model):
    """Extracts router weights from the model."""
    router_weights = {}
    for name, param in model.named_parameters():
        if "router" in name or "gate" in name:
            router_weights[name] = param.detach().cpu().to(torch.float32).numpy()
    if not router_weights:
        print("Warning: No router weights found in the model.")
    return router_weights

def main():
    """
    Main function to load models, extract router weights,
    and compare their statistics.
    """
    # --- Configuration ---
    # IMPORTANT: This path needs to be updated with the new output path
    # after you run the training with the DDP config.
    trained_model_path = "outputs/2025-06-14/20-44-19/stage_1_pretrain"

    # Path to the configuration file for the base model
    config_path = "configs/stage_1_pretrain.yaml"

    # Model loading arguments
    model_kwargs = {
        "low_cpu_mem_usage": True,
        "torch_dtype": torch.bfloat16,
    }

    # --- Load Base Model Name from YAML ---
    print(f"Loading base model configuration from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        base_model_name = config['selector_model']['name']
        print(f"Base model name: {base_model_name}")
    except (FileNotFoundError, KeyError) as e:
        print(f"Error reading config file: {e}")
        return

    # --- Load Models ---
    # Load the base (untrained) model from Hugging Face Hub
    base_model = load_model(base_model_name, model_kwargs, is_from_hub=True)
    if base_model is None:
        return

    # Load the trained model from a local path
    print(f"\nAttempting to load trained model from: {trained_model_path}")
    if "YYYY-MM-DD" in trained_model_path:
        print("\nWARNING: Please update the 'trained_model_path' in this script with the actual path from your new training run.")
        return
        
    trained_model = load_model(trained_model_path, model_kwargs, is_from_hub=False)
    if trained_model is None:
        return

    # --- Extract Router Weights ---
    print("\nExtracting router weights...")
    base_router_weights = get_router_weights(base_model)
    trained_router_weights = get_router_weights(trained_model)

    if not base_router_weights or not trained_router_weights:
        print("Could not proceed with comparison due to missing router weights.")
        return

    # --- Compare Weights ---
    print("\n--- Router Weight Statistics ---")
    all_keys = sorted(base_router_weights.keys())

    for key in all_keys:
        if key in trained_router_weights:
            base_w = base_router_weights[key]
            trained_w = trained_router_weights[key]

            print(f"\nLayer: {key}")
            print(f"  Shape: {base_w.shape}")
            
            # Base Model Stats
            base_mean = np.mean(base_w)
            base_var = np.var(base_w)
            print(f"  Base Model    -> Mean: {base_mean:.6f}, Variance: {base_var:.6f}")

            # Trained Model Stats
            trained_mean = np.mean(trained_w)
            trained_var = np.var(trained_w)
            print(f"  Trained Model -> Mean: {trained_mean:.6f}, Variance: {trained_var:.6f}")
        else:
            print(f"\nLayer: {key} not found in trained model.")

    print("\n--- Comparison Complete ---")

if __name__ == "__main__":
    main()