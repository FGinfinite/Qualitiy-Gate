#!/usr/bin/env python3
# Copyright 2024 Select-MoE Project. All rights reserved.
# Licensed under the Apache License, Version 2.0

"""
Script to convert OLMoE pretrained models to Select-MoE format and save them.
This creates a standard HuggingFace model that can be loaded with from_pretrained().
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
from transformers.models.olmoe.modeling_olmoe import OlmoeForCausalLM

# 获取当前脚本所在目录的父目录（即项目根目录）
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)  # 将项目根目录添加到sys.path的最前面

from src.models.select_moe import (
    SelectMoeConfig,
    SelectMoeForCausalLM,
    register_select_moe,
)


def set_random_seed(seed):
    """
    设置所有随机数生成器的种子以确保可重现性。

    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 确保CUDA操作的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def convert_and_save_model(
    model_name="allenai/OLMoE-1B-7B-0125",
    save_path=None,
    device="cpu",
    quality_gate_init_mean=0.0,
    quality_gate_init_std=0.02,
    quality_loss_weight=0.01,
    trash_expert_mode="zero",
    enable_load_balancing=False,
    freeze_non_routing=False,  # New parameter
    push_to_hub=False,
    hub_repo_name=None,
    seed=42,
):
    """
    Convert OLMoE pretrained model to Select-MoE and save it.

    Args:
        model_name: HuggingFace model name or local path
        save_path: Local path to save the converted model
        device: Device to use for conversion
        quality_gate_init_mean: Mean for quality gate initialization
        quality_gate_init_std: Std for quality gate initialization
        quality_loss_weight: Weight for quality classification loss
        trash_expert_mode: Trash expert mode ("zero", "noise", "custom")
        enable_load_balancing: Whether to enable MoE load balancing
        freeze_non_routing: Whether to freeze non-routing weights after conversion
        push_to_hub: Whether to push to HuggingFace Hub
        hub_repo_name: Repository name for Hub upload
        seed: Random seed for reproducibility (default: 42)
    """

    # 设置随机种子以确保可重现性
    set_random_seed(seed)

    print("=" * 80)
    print("Converting OLMoE Model to Select-MoE")
    print(f"Source: {model_name}")
    print(f"Device: {device}")
    print(f"Random seed: {seed}")
    print("=" * 80)

    # Register Select-MoE
    register_select_moe()
    print("✓ Select-MoE registered with transformers")

    try:
        # 1. Load original model
        print(f"\n1. Loading original model: {model_name}")
        print("   This may take a few minutes...")

        original_model = OlmoeForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",  # Use original model's dtype
            device_map=device if device != "cpu" else None,
        )
        original_config = original_model.config

        print("✓ Original model loaded successfully!")
        print(f"  - Model type: {type(original_model).__name__}")
        print(f"  - Number of experts: {original_config.num_experts}")
        print(f"  - Experts per token: {original_config.num_experts_per_tok}")
        print(f"  - Number of layers: {original_config.num_hidden_layers}")
        print(f"  - Hidden size: {original_config.hidden_size}")
        print(f"  - Total parameters: {sum(p.numel() for p in original_model.parameters())}")

        # 2. Create Select-MoE config
        print("\n2. Creating Select-MoE configuration...")
        select_moe_config_dict = original_config.to_dict()
        select_moe_config_dict.update(
            {
                "model_type": "select_moe",
                "quality_gate_init_mean": quality_gate_init_mean,
                "quality_gate_init_std": quality_gate_init_std,
                "quality_loss_weight": quality_loss_weight,
                "trash_expert_mode": trash_expert_mode,
                "enable_load_balancing": enable_load_balancing,
            }
        )

        select_moe_config = SelectMoeConfig(**select_moe_config_dict)

        print("✓ Select-MoE config created!")
        print(f"  - Two-tier routing: Quality Gate + MoE + Trash Expert")
        print(f"  - Quality gate init: mean={quality_gate_init_mean}, std={quality_gate_init_std}")
        print(f"  - Quality loss weight: {quality_loss_weight}")
        print(f"  - Trash expert mode: {trash_expert_mode}")
        print(f"  - Load balancing: {enable_load_balancing}")

        # 3. Convert the model using new two-tier architecture
        print("\n3. Converting to two-tier routing architecture...")

        # Create new Select-MoE model with two-tier routing
        select_moe_model = SelectMoeForCausalLM(select_moe_config)

        # Copy compatible weights from original model
        original_state_dict = original_model.state_dict()
        select_moe_state_dict = select_moe_model.state_dict()

        print("Mapping original MLP weights to normal_moe...")
        copied_count = 0

        # Copy all weights from original model to new model
        # Special handling for mlp -> normal_moe mapping
        for key in original_state_dict:
            if key in select_moe_state_dict:
                # Direct mapping (non-MLP weights)
                select_moe_state_dict[key].copy_(original_state_dict[key])
                copied_count += 1
            elif ".mlp." in key:
                # Map mlp weights to normal_moe
                new_key = key.replace(".mlp.", ".normal_moe.")
                if new_key in select_moe_state_dict:
                    select_moe_state_dict[new_key].copy_(original_state_dict[key])
                    copied_count += 1
                    print(f"  Mapped: {key} -> {new_key}")
                else:
                    print(f"  Warning: Could not map {key} to {new_key}")

        print(f"Total weights copied: {copied_count}/{len(original_state_dict)}")

        # Verify quality gates are initialized (they should be random)
        quality_gate_count = 0
        for key in select_moe_state_dict:
            if "quality_gate" in key:
                quality_gate_count += 1
        print(f"Quality gates found: {quality_gate_count}")

        print("✓ Weight copying completed!")
        print("✓ Two-tier routing architecture conversion completed!")

        # Move to device if specified
        if device != "cpu":
            select_moe_model = select_moe_model.to(device)

        # 4. Verify the conversion
        print("\n4. Verifying conversion...")

        # Test forward pass
        batch_size, seq_len = 1, 8
        test_input = torch.randint(0, select_moe_config.vocab_size, (batch_size, seq_len))
        if device != "cpu":
            test_input = test_input.to(device)

        with torch.no_grad():
            outputs = select_moe_model(test_input, output_router_logits=True)

        print("✓ Forward pass successful!")
        print(f"  - Output shape: {outputs.logits.shape}")
        print(f"  - Router logits layers: {len(outputs.router_logits)}")

        # Verify new two-tier router output format
        first_layer_router = outputs.router_logits[0]
        if isinstance(first_layer_router, dict):
            print("✓ Two-tier routing output format verified!")
            print(f"  - Quality logits shape: {first_layer_router['quality_logits'].shape}")
            print(f"  - MoE logits shape: {first_layer_router['moe_logits'].shape}")

            # Verify dimensions
            expected_quality_dims = (batch_size, seq_len, 2)  # [good, bad]
            expected_moe_dims = (batch_size * seq_len, select_moe_config.num_experts)

            actual_quality_dims = first_layer_router["quality_logits"].shape
            actual_moe_dims = first_layer_router["moe_logits"].shape

            assert actual_quality_dims == expected_quality_dims, f"Quality logits shape mismatch: {actual_quality_dims} != {expected_quality_dims}"
            assert actual_moe_dims == expected_moe_dims, f"MoE logits shape mismatch: {actual_moe_dims} != {expected_moe_dims}"

            print("✓ Router logits dimensions verification passed!")
        else:
            raise ValueError("Expected dictionary format for two-tier router logits, got:", type(first_layer_router))

        # Optional: Freeze non-routing weights if specified
        if freeze_non_routing:
            print("\n4.5. Freezing non-routing weights...")
            trainable_params, frozen_params = select_moe_model.freeze_non_routing_weights()
            print(f"✓ Non-routing weights frozen. Trainable: {trainable_params:,}, Frozen: {frozen_params:,}")

        # 5. Save the model
        print("\n5. Saving converted model...")

        if save_path is None:
            save_path = f"./converted_models/select_moe_converted_{model_name.split('/')[-1]}"

        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)

        # Save model and config
        select_moe_model.save_pretrained(save_path)

        print(f"✓ Model saved to: {save_path}")
        print("✓ Config saved successfully!")

        # 6. Test loading the saved model
        print("\n6. Testing saved model loading...")

        # Load from saved path
        test_loaded_model = SelectMoeForCausalLM.from_pretrained(save_path)
        print("✓ Model loaded successfully from saved path!")
        print(f"  - Model type: {type(test_loaded_model).__name__}")
        print(f"  - Config type: {type(test_loaded_model.config).__name__}")

        # Quick functionality test
        if device != "cpu":
            test_loaded_model = test_loaded_model.to(device)

        with torch.no_grad():
            test_outputs = test_loaded_model(test_input, output_router_logits=True)

        print("✓ Loaded model forward pass successful!")
        print(f"  - Output shape: {test_outputs.logits.shape}")

        # Verify loaded model has correct two-tier routing format
        loaded_first_layer_router = test_outputs.router_logits[0]
        if isinstance(loaded_first_layer_router, dict):
            print("✓ Loaded model two-tier routing format verified!")
            print(f"  - Quality logits shape: {loaded_first_layer_router['quality_logits'].shape}")
            print(f"  - MoE logits shape: {loaded_first_layer_router['moe_logits'].shape}")
        else:
            raise ValueError("Loaded model router output format incorrect")

        # 7. Push to Hub (optional)
        if push_to_hub:
            print("\n7. Pushing to HuggingFace Hub...")

            if hub_repo_name is None:
                hub_repo_name = f"select_moe_{model_name.split('/')[-1]}"

            try:
                select_moe_model.push_to_hub(hub_repo_name, private=False)
                print(f"✓ Model pushed to Hub: {hub_repo_name}")
                print("   You can now load it with:")
                print(f"   model = SelectMoeForCausalLM.from_pretrained('{hub_repo_name}')")
            except Exception as hub_error:
                print(f"❌ Failed to push to Hub: {hub_error}")
                print("   Make sure you're logged in with: huggingface-cli login")

        # 8. Summary
        print("\n" + "=" * 60)
        print("🎉 CONVERSION COMPLETED SUCCESSFULLY!")
        print("=" * 60)

        print(f"\n📁 Local save path: {os.path.abspath(save_path)}")
        print("\n📝 Usage instructions:")
        print("   # Load the converted model directly")
        print("   from models.select_moe import SelectMoeForCausalLM, register_select_moe")
        print("   register_select_moe()")
        print(f"   model = SelectMoeForCausalLM.from_pretrained('{save_path}')")

        if push_to_hub and hub_repo_name:
            print("\n   # Or load from HuggingFace Hub")
            print(f"   model = SelectMoeForCausalLM.from_pretrained('{hub_repo_name}')")

        print("\n✅ Key benefits:")
        print("   ✓ No manual conversion needed")
        print("   ✓ Standard HuggingFace API compatibility")
        print("   ✓ Preserves all original pretrained weights")
        print("   ✓ Two-tier routing: Quality Gate + MoE + Trash Expert")
        print("   ✓ Ready for data selection and quality-aware training")

        return save_path

    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    """Main conversion script."""
    parser = argparse.ArgumentParser(description="Convert OLMoE pretrained models to Select-MoE format")

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="allenai/OLMoE-1B-7B-0125",
        help="Source model name or path",
    )
    parser.add_argument("--save-path", type=str, default=None, help="Local path to save converted model")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (cpu, cuda, cuda:0, etc.)",
    )

    # Select-MoE configuration
    parser.add_argument(
        "--quality-gate-init-mean",
        type=float,
        default=0.0,
        help="Mean for quality gate initialization",
    )
    parser.add_argument(
        "--quality-gate-init-std",
        type=float,
        default=0.02,
        help="Std for quality gate initialization",
    )
    parser.add_argument(
        "--quality-loss-weight",
        type=float,
        default=0.5,
        help="Weight for quality classification loss",
    )
    parser.add_argument(
        "--trash-expert-mode",
        type=str,
        default="zero",
        choices=["zero", "noise", "custom"],
        help="Trash expert mode",
    )
    parser.add_argument(
        "--enable-load-balancing",
        action="store_true",
        help="Enable MoE load balancing loss",
    )
    parser.add_argument(
        "--freeze-non-routing",
        action="store_true",
        help="Freeze non-routing weights after conversion (for routing-only training)",
    )

    # Random seed
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    # Hub upload
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push converted model to HuggingFace Hub",
    )
    parser.add_argument("--hub-repo-name", type=str, default=None, help="Repository name for Hub upload")

    args = parser.parse_args()

    # Validate device
    if args.device != "cpu" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        args.device = "cpu"
    elif args.device.startswith("cuda:"):
        device_id = int(args.device.split(":")[1])
        if device_id >= torch.cuda.device_count():
            print(f"⚠️  Device {args.device} not available, falling back to CPU")
            args.device = "cpu"

    print(f"Using device: {args.device}")
    if args.device != "cpu":
        print(f"GPU memory: {torch.cuda.get_device_properties(args.device).total_memory / 1024**3:.1f} GB")

    # Convert and save
    result_path = convert_and_save_model(
        model_name=args.model,
        save_path=args.save_path,
        device=args.device,
        quality_gate_init_mean=args.quality_gate_init_mean,
        quality_gate_init_std=args.quality_gate_init_std,
        quality_loss_weight=args.quality_loss_weight,
        trash_expert_mode=args.trash_expert_mode,
        enable_load_balancing=args.enable_load_balancing,
        freeze_non_routing=args.freeze_non_routing,  # New parameter
        push_to_hub=args.push_to_hub,
        hub_repo_name=args.hub_repo_name,
        seed=args.seed,
    )

    if result_path:
        print(f"\n✅ Success! Model saved to: {result_path}")
        exit(0)
    else:
        print("\n❌ Conversion failed!")
        exit(1)


if __name__ == "__main__":
    main()
