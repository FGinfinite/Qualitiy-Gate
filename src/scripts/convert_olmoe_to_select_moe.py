#!/usr/bin/env python3
# Copyright 2024 Select-MoE Project. All rights reserved.
# Licensed under the Apache License, Version 2.0

"""
Script to convert OLMoE pretrained models to Select-MoE format and save them.
This creates a standard HuggingFace model that can be loaded with from_pretrained().
"""

import argparse
import os
import sys
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.olmoe.modeling_olmoe import OlmoeForCausalLM

# Add project root to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.select_moe import (
    SelectMoeConfig,
    SelectMoeForCausalLM,
    replace_moe_layers_with_trashcan,
    register_select_moe
)


def convert_and_save_model(
    model_name="allenai/OLMoE-1B-7B-0125",
    save_path=None,
    device="cpu",
    trash_can_init_mean=0.0,
    trash_can_init_std=0.02,
    constraint_loss_weight=0.01,
    trash_can_loss_alpha=1.0,
    trash_can_loss_beta=2.0,
    push_to_hub=False,
    hub_repo_name=None
):
    """
    Convert OLMoE pretrained model to Select-MoE and save it.
    
    Args:
        model_name: HuggingFace model name or local path
        save_path: Local path to save the converted model
        device: Device to use for conversion
        trash_can_init_mean: Mean for trash can expert initialization
        trash_can_init_std: Std for trash can expert initialization
        constraint_loss_weight: Weight for constraint loss
        trash_can_loss_alpha: Alpha parameter for beta distribution loss
        trash_can_loss_beta: Beta parameter for beta distribution loss
        push_to_hub: Whether to push to HuggingFace Hub
        hub_repo_name: Repository name for Hub upload
    """
    
    print("=" * 80)
    print(f"Converting OLMoE Model to Select-MoE")
    print(f"Source: {model_name}")
    print(f"Device: {device}")
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
            torch_dtype=torch.float32,  # Use float32 for precise weight handling
            device_map=device if device != "cpu" else None,
        )
        original_config = original_model.config
        
        print(f"✓ Original model loaded successfully!")
        print(f"  - Model type: {type(original_model).__name__}")
        print(f"  - Number of experts: {original_config.num_experts}")
        print(f"  - Experts per token: {original_config.num_experts_per_tok}")
        print(f"  - Number of layers: {original_config.num_hidden_layers}")
        print(f"  - Hidden size: {original_config.hidden_size}")
        print(f"  - Total parameters: {sum(p.numel() for p in original_model.parameters()):,}")
        
        # 2. Create Select-MoE config
        print(f"\n2. Creating Select-MoE configuration...")
        select_moe_config_dict = original_config.to_dict()
        select_moe_config_dict.update({
            "model_type": "select_moe",
            "trash_can_init_mean": trash_can_init_mean,
            "trash_can_init_std": trash_can_init_std,
            "constraint_loss_weight": constraint_loss_weight,
            "trash_can_loss_alpha": trash_can_loss_alpha,
            "trash_can_loss_beta": trash_can_loss_beta,
        })
        
        select_moe_config = SelectMoeConfig(**select_moe_config_dict)
        
        print(f"✓ Select-MoE config created!")
        print(f"  - Original experts: {select_moe_config.num_experts}")
        print(f"  - Trash can experts: {select_moe_config.num_experts_per_tok}")
        print(f"  - Total experts per layer: {select_moe_config.num_experts + select_moe_config.num_experts_per_tok}")
        print(f"  - Trash can init: mean={trash_can_init_mean}, std={trash_can_init_std}")
        
        # 3. Convert the model
        print(f"\n3. Converting model architecture...")
        
        # Convert MoE layers in-place
        replace_moe_layers_with_trashcan(original_model, select_moe_config)
        
        # Create proper Select-MoE model instance
        select_moe_model = SelectMoeForCausalLM(select_moe_config)
        select_moe_model.load_state_dict(original_model.state_dict(), strict=False)
        
        # Move to device if specified
        if device != "cpu":
            select_moe_model = select_moe_model.to(device)
        
        print(f"✓ Model conversion completed!")
        
        # 4. Verify the conversion
        print(f"\n4. Verifying conversion...")
        
        # Test forward pass
        batch_size, seq_len = 1, 8
        test_input = torch.randint(0, select_moe_config.vocab_size, (batch_size, seq_len))
        if device != "cpu":
            test_input = test_input.to(device)
        
        with torch.no_grad():
            outputs = select_moe_model(test_input, output_router_logits=True)
            
        print(f"✓ Forward pass successful!")
        print(f"  - Output shape: {outputs.logits.shape}")
        print(f"  - Router logits layers: {len(outputs.router_logits)}")
        print(f"  - Router logits shape: {outputs.router_logits[0].shape}")
        
        # Verify router logits dimensions
        expected_experts = select_moe_config.num_experts + select_moe_config.num_experts_per_tok
        actual_experts = outputs.router_logits[0].shape[-1]
        assert actual_experts == expected_experts, f"Router logits mismatch: {actual_experts} != {expected_experts}"
        print(f"✓ Router logits verification passed!")
        
        # 5. Save the model
        print(f"\n5. Saving converted model...")
        
        if save_path is None:
            save_path = f"./converted_models/select_moe_converted_{model_name.split('/')[-1]}"
        
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Save model and config
        select_moe_model.save_pretrained(save_path)
        
        print(f"✓ Model saved to: {save_path}")
        print(f"✓ Config saved successfully!")
        
        # 6. Test loading the saved model
        print(f"\n6. Testing saved model loading...")
        
        # Load from saved path
        test_loaded_model = SelectMoeForCausalLM.from_pretrained(save_path)
        print(f"✓ Model loaded successfully from saved path!")
        print(f"  - Model type: {type(test_loaded_model).__name__}")
        print(f"  - Config type: {type(test_loaded_model.config).__name__}")
        
        # Quick functionality test
        if device != "cpu":
            test_loaded_model = test_loaded_model.to(device)
        
        with torch.no_grad():
            test_outputs = test_loaded_model(test_input, output_router_logits=True)
            
        print(f"✓ Loaded model forward pass successful!")
        print(f"  - Output shape: {test_outputs.logits.shape}")
        print(f"  - Router logits shape: {test_outputs.router_logits[0].shape}")
        
        # 7. Push to Hub (optional)
        if push_to_hub:
            print(f"\n7. Pushing to HuggingFace Hub...")
            
            if hub_repo_name is None:
                hub_repo_name = f"select_moe_{model_name.split('/')[-1]}"
            
            try:
                select_moe_model.push_to_hub(hub_repo_name, private=False)
                print(f"✓ Model pushed to Hub: {hub_repo_name}")
                print(f"   You can now load it with:")
                print(f"   model = SelectMoeForCausalLM.from_pretrained('{hub_repo_name}')")
            except Exception as hub_error:
                print(f"❌ Failed to push to Hub: {hub_error}")
                print(f"   Make sure you're logged in with: huggingface-cli login")
        
        # 8. Summary
        print(f"\n" + "="*60)
        print("🎉 CONVERSION COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print(f"\n📁 Local save path: {os.path.abspath(save_path)}")
        print(f"\n📝 Usage instructions:")
        print(f"   # Load the converted model directly")
        print(f"   from models.select_moe import SelectMoeForCausalLM, register_select_moe")
        print(f"   register_select_moe()")
        print(f"   model = SelectMoeForCausalLM.from_pretrained('{save_path}')")
        
        if push_to_hub and hub_repo_name:
            print(f"\n   # Or load from HuggingFace Hub")
            print(f"   model = SelectMoeForCausalLM.from_pretrained('{hub_repo_name}')")
        
        print(f"\n✅ Key benefits:")
        print(f"   ✓ No manual conversion needed")
        print(f"   ✓ Standard HuggingFace API compatibility")
        print(f"   ✓ Preserves all original pretrained weights")
        print(f"   ✓ Ready for fine-tuning with trash can experts")
        
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
    parser.add_argument("--model", type=str, default="allenai/OLMoE-1B-7B-0125",
                        help="Source model name or path")
    parser.add_argument("--save-path", type=str, default=None,
                        help="Local path to save converted model")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use (cpu, cuda, cuda:0, etc.)")
    
    # Select-MoE configuration
    parser.add_argument("--trash-can-init-mean", type=float, default=0.0,
                        help="Mean for trash can expert initialization")
    parser.add_argument("--trash-can-init-std", type=float, default=0.02,
                        help="Std for trash can expert initialization")
    parser.add_argument("--constraint-loss-weight", type=float, default=0.01,
                        help="Weight for constraint loss")
    parser.add_argument("--trash-can-loss-alpha", type=float, default=1.0,
                        help="Alpha parameter for beta distribution loss")
    parser.add_argument("--trash-can-loss-beta", type=float, default=2.0,
                        help="Beta parameter for beta distribution loss")
    
    # Hub upload
    parser.add_argument("--push-to-hub", action="store_true",
                        help="Push converted model to HuggingFace Hub")
    parser.add_argument("--hub-repo-name", type=str, default=None,
                        help="Repository name for Hub upload")
    
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
        trash_can_init_mean=args.trash_can_init_mean,
        trash_can_init_std=args.trash_can_init_std,
        constraint_loss_weight=args.constraint_loss_weight,
        trash_can_loss_alpha=args.trash_can_loss_alpha,
        trash_can_loss_beta=args.trash_can_loss_beta,
        push_to_hub=args.push_to_hub,
        hub_repo_name=args.hub_repo_name
    )
    
    if result_path:
        print(f"\n✅ Success! Model saved to: {result_path}")
        exit(0)
    else:
        print(f"\n❌ Conversion failed!")
        exit(1)


if __name__ == "__main__":
    main()