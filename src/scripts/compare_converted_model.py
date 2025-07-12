#!/usr/bin/env python3
# Copyright 2024 Select-MoE Project. All rights reserved.
# Licensed under the Apache License, Version 2.0

"""
Script to compare weights between original OLMoE model and already converted Select-MoE model.
This script loads both models separately and compares their weights.
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
    register_select_moe
)


def compare_state_dicts(original_dict, converted_dict, tolerance=1e-6):
    """
    Compare original and converted model state dictionaries.
    
    Args:
        original_dict: Original OLMoE model state dict
        converted_dict: Converted Select-MoE model state dict
        tolerance: Numerical tolerance for comparison
    
    Returns:
        dict: Detailed comparison results
    """
    results = {
        "identical_keys": [],
        "missing_in_converted": [],
        "missing_in_original": [],
        "shape_mismatch": [],
        "value_mismatch": [],
        "gate_analysis": []  # For gate weights
    }
    
    all_keys = set(original_dict.keys()) | set(converted_dict.keys())
    
    for key in sorted(all_keys):
        if key not in converted_dict:
            results["missing_in_converted"].append(key)
            continue
        elif key not in original_dict:
            results["missing_in_original"].append(key)
            continue
        
        original_tensor = original_dict[key]
        converted_tensor = converted_dict[key]
        
        # Special handling for gate weights
        if "gate.weight" in key and len(original_tensor.shape) == 2 and len(converted_tensor.shape) == 2:
            # Gate weights should be expanded from [64, hidden_size] to [72, hidden_size]
            if converted_tensor.shape[0] == original_tensor.shape[0] + 8 and converted_tensor.shape[1] == original_tensor.shape[1]:
                # Check if first 64 experts match
                first_64_experts = converted_tensor[:original_tensor.shape[0], :]
                matches = torch.allclose(original_tensor, first_64_experts, atol=tolerance)
                max_diff = torch.max(torch.abs(original_tensor - first_64_experts)).item() if original_tensor.shape == first_64_experts.shape else float('inf')
                
                # Analyze trash can experts (last 8)
                trash_can_experts = converted_tensor[original_tensor.shape[0]:, :]
                
                results["gate_analysis"].append({
                    "key": key,
                    "original_shape": original_tensor.shape,
                    "converted_shape": converted_tensor.shape,
                    "first_64_match": matches,
                    "max_diff_first_64": max_diff,
                    "trash_can_shape": trash_can_experts.shape,
                    "trash_can_mean": trash_can_experts.mean().item(),
                    "trash_can_std": trash_can_experts.std().item(),
                    "trash_can_min": trash_can_experts.min().item(),
                    "trash_can_max": trash_can_experts.max().item(),
                })
                continue
            else:
                results["shape_mismatch"].append({
                    "key": key,
                    "original_shape": original_tensor.shape,
                    "converted_shape": converted_tensor.shape
                })
                continue
        
        # Check shapes for non-gate weights
        if original_tensor.shape != converted_tensor.shape:
            results["shape_mismatch"].append({
                "key": key,
                "original_shape": original_tensor.shape,
                "converted_shape": converted_tensor.shape
            })
            continue
        
        # Check values for non-gate weights
        if torch.allclose(original_tensor, converted_tensor, atol=tolerance):
            results["identical_keys"].append(key)
        else:
            max_diff = torch.max(torch.abs(original_tensor - converted_tensor)).item()
            results["value_mismatch"].append({
                "key": key,
                "max_diff": max_diff,
                "shape": original_tensor.shape
            })
    
    return results


def compare_models(
    original_model_name="allenai/OLMoE-1B-7B-0125",
    converted_model_path="./converted_models/select_moe_converted_OLMoE-1B-7B-0125",
    device="cpu",
    dtype="bfloat16",
    memory_efficient=True
):
    """
    Compare original OLMoE model with converted Select-MoE model.
    
    Args:
        original_model_name: HuggingFace model name for original model
        converted_model_path: Local path to converted Select-MoE model
        device: Device to load models on
        dtype: Data type for model loading ('float32', 'bfloat16', 'float16')
        memory_efficient: If True, load models sequentially to save memory
    """
    
    print("=" * 80)
    print(f"Comparing Original vs Converted Select-MoE Models")
    print(f"Original: {original_model_name}")
    print(f"Converted: {converted_model_path}")
    print(f"Device: {device}")
    print(f"Data type: {dtype}")
    print(f"Memory efficient: {memory_efficient}")
    print("=" * 80)
    
    # Convert dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16
    }
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)
    
    # Register Select-MoE
    register_select_moe()
    print("✓ Select-MoE registered with transformers")
    
    # Check available memory
    if device != "cpu" and torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
        free_memory = torch.cuda.memory_reserved(device) / 1024**3
        print(f"GPU memory: {total_memory:.1f} GB total, {free_memory:.1f} GB reserved")
        
        # Estimate model memory usage
        model_memory_gb = 6.9 * (2 if dtype == "float32" else 1)  # Rough estimate
        if memory_efficient:
            print(f"Memory efficient mode: Loading models sequentially")
            print(f"Estimated memory per model: ~{model_memory_gb:.1f} GB")
        else:
            print(f"Standard mode: Loading both models simultaneously")
            print(f"Estimated total memory needed: ~{model_memory_gb * 2:.1f} GB")
            if model_memory_gb * 2 > total_memory * 0.8:
                print("⚠️  Warning: Estimated memory usage may exceed available GPU memory")
                print("   Consider using --memory-efficient flag or --dtype bfloat16")
    
    try:
        # 1. Load original model
        print(f"\n1. Loading original model: {original_model_name}")
        print("   This may take a few minutes...")
        
        original_model = OlmoeForCausalLM.from_pretrained(
            original_model_name,
            torch_dtype=torch_dtype,
            device_map=device if device != "cpu" else None,
            low_cpu_mem_usage=True,  # Enable memory efficient loading
        )
        original_config = original_model.config
        
        print(f"✓ Original model loaded successfully!")
        print(f"  - Model type: {type(original_model).__name__}")
        print(f"  - Number of experts: {original_config.num_experts}")
        print(f"  - Experts per token: {original_config.num_experts_per_tok}")
        print(f"  - Number of layers: {original_config.num_hidden_layers}")
        print(f"  - Hidden size: {original_config.hidden_size}")
        print(f"  - Data type: {torch_dtype}")
        print(f"  - Total parameters: {sum(p.numel() for p in original_model.parameters()):,}")
        
        # Show original gate weights info
        print(f"\nOriginal gate weights:")
        gate_count = 0
        for name, param in original_model.named_parameters():
            if "gate.weight" in name:
                print(f"  - {name}: {param.shape} ({param.dtype})")
                gate_count += 1
                if gate_count <= 2:  # Show first 2 layers only
                    pass
                elif gate_count == 3:
                    print(f"  - ... (total {original_config.num_hidden_layers} layers)")
                    break
        print(f"  Total gate layers: {original_config.num_hidden_layers}")
        
        # Get original state dict and optionally clear model from memory
        if memory_efficient:
            print(f"\n📊 Memory efficient mode: Extracting state dict...")
            original_state_dict = original_model.state_dict()
            # Move to CPU to free GPU memory
            original_model = original_model.cpu()
            if device != "cpu":
                torch.cuda.empty_cache()
                print(f"   ✓ Original model moved to CPU, GPU cache cleared")
        else:
            original_state_dict = original_model.state_dict()
        
        # 2. Load converted model
        print(f"\n2. Loading converted model: {converted_model_path}")
        
        try:
            converted_model = SelectMoeForCausalLM.from_pretrained(
                converted_model_path,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
            )
            if device != "cpu":
                converted_model = converted_model.to(device)
            
            converted_config = converted_model.config
            
            print(f"✓ Converted model loaded successfully!")
            print(f"  - Model type: {type(converted_model).__name__}")
            print(f"  - Number of experts: {converted_config.num_experts}")
            print(f"  - Trash can experts: {converted_config.num_experts_per_tok}")
            print(f"  - Total experts per layer: {converted_config.num_experts + converted_config.num_experts_per_tok}")
            print(f"  - Data type: {torch_dtype}")
            print(f"  - Total parameters: {sum(p.numel() for p in converted_model.parameters()):,}")
            
            # Show converted gate weights info
            print(f"\nConverted gate weights:")
            for name, param in converted_model.named_parameters():
                if "gate.weight" in name:
                    print(f"  - {name}: {param.shape} ({param.dtype})")
                    gate_count += 1
                    if gate_count <= 2:  # Show first 2 layers only
                        pass
                    elif gate_count == 3:
                        print(f"  - ... (total {converted_config.num_hidden_layers} layers)")
                        break
            
            converted_state_dict = converted_model.state_dict()
            
        except Exception as e:
            print(f"❌ Failed to load converted model: {e}")
            print(f"   Make sure the path exists and Select-MoE is registered")
            return False
        
        # 3. Compare state dictionaries
        print(f"\n" + "="*60)
        print("WEIGHT COMPARISON RESULTS")
        print("="*60)
        
        # Adjust tolerance based on dtype
        tolerance_map = {
            "float32": 1e-8,
            "bfloat16": 1e-6,
            "float16": 1e-5
        }
        tolerance = tolerance_map.get(dtype, 1e-6)
        print(f"Using tolerance: {tolerance} (appropriate for {dtype})")
        
        results = compare_state_dicts(original_state_dict, converted_state_dict, tolerance)
        
        print(f"✓ Identical weights: {len(results['identical_keys'])}")
        if len(results["identical_keys"]) <= 10:
            for key in results["identical_keys"]:
                print(f"    {key}")
        else:
            for key in results["identical_keys"][:5]:
                print(f"    {key}")
            print(f"    ... and {len(results['identical_keys']) - 5} more")
        
        if results["shape_mismatch"]:
            print(f"\n⚠️  Unexpected shape mismatches: {len(results['shape_mismatch'])}")
            for item in results["shape_mismatch"]:
                if "gate.weight" not in item["key"]:  # Only show non-gate mismatches
                    print(f"    {item['key']}: {item['original_shape']} -> {item['converted_shape']}")
        
        if results["value_mismatch"]:
            print(f"\n❌ Value mismatches: {len(results['value_mismatch'])}")
            for item in results["value_mismatch"][:5]:  # Show first 5
                print(f"    {item['key']}: max_diff={item['max_diff']:.2e}")
            if len(results["value_mismatch"]) > 5:
                print(f"    ... and {len(results['value_mismatch']) - 5} more")
        
        if results["missing_in_converted"]:
            print(f"\n⚠️  Missing in converted model: {len(results['missing_in_converted'])}")
            for key in results["missing_in_converted"][:5]:
                print(f"    {key}")
        
        if results["missing_in_original"]:
            print(f"\n⚠️  Extra in converted model: {len(results['missing_in_original'])}")
            for key in results["missing_in_original"][:5]:
                print(f"    {key}")
        
        # 4. Gate weight analysis
        print(f"\n" + "="*60)
        print("GATE WEIGHT ANALYSIS")
        print("="*60)
        
        if results["gate_analysis"]:
            print(f"Gate weights analyzed: {len(results['gate_analysis'])}")
            
            all_match = True
            for i, item in enumerate(results["gate_analysis"]):
                key = item["key"]
                original_shape = item["original_shape"]
                converted_shape = item["converted_shape"]
                first_64_match = item["first_64_match"]
                max_diff = item["max_diff_first_64"]
                
                if i < 2 or not first_64_match:  # Show first 2 or any mismatches
                    print(f"\n  {key}:")
                    print(f"    Original shape: {original_shape}")
                    print(f"    Converted shape: {converted_shape}")
                    print(f"    First 64 experts match: {'✓' if first_64_match else '❌'}")
                    print(f"    Max difference in first 64 experts: {max_diff:.2e}")
                    
                    # Trash can expert analysis
                    trash_shape = item["trash_can_shape"]
                    trash_mean = item["trash_can_mean"]
                    trash_std = item["trash_can_std"]
                    trash_min = item["trash_can_min"]
                    trash_max = item["trash_can_max"]
                    
                    print(f"    Trash can experts shape: {trash_shape}")
                    print(f"    Trash can weights mean: {trash_mean:.6f}")
                    print(f"    Trash can weights std: {trash_std:.6f}")
                    print(f"    Trash can weights range: [{trash_min:.6f}, {trash_max:.6f}]")
                    
                    # Check if initialization is reasonable
                    expected_std = 0.02
                    std_ratio = trash_std / expected_std
                    print(f"    Std ratio (actual/expected): {std_ratio:.3f}")
                    
                    if abs(trash_mean) > 0.01:
                        print(f"    ⚠️  Mean is not close to 0")
                    if not (0.8 <= std_ratio <= 1.2):
                        print(f"    ⚠️  Std deviation is not close to expected value")
                
                if not first_64_match:
                    all_match = False
            
            if len(results["gate_analysis"]) > 2 and all_match:
                print(f"\n  ... and {len(results['gate_analysis']) - 2} more gate layers (all matching)")
        
        # 5. Functionality test
        print(f"\n" + "="*60)
        print("FUNCTIONALITY TEST")
        print("="*60)
        
        print("Creating test input...")
        batch_size, seq_len = 1, 8
        input_ids = torch.randint(0, original_config.vocab_size, (batch_size, seq_len))
        if device != "cpu":
            input_ids = input_ids.to(device)
        
        # Test original model (reload to GPU if in memory efficient mode)
        if memory_efficient and device != "cpu":
            print("Moving original model back to GPU for testing...")
            original_model = original_model.to(device)
        
        print("Testing original model forward pass...")
        with torch.no_grad():
            original_outputs = original_model(input_ids, output_router_logits=True)
            print(f"✓ Original model output shape: {original_outputs.logits.shape}")
            print(f"  Router logits shape: {original_outputs.router_logits[0].shape}")
        
        print("Testing converted model forward pass...")
        with torch.no_grad():
            converted_outputs = converted_model(input_ids, output_router_logits=True)
            print(f"✓ Converted model output shape: {converted_outputs.logits.shape}")
            print(f"  Router logits shape: {converted_outputs.router_logits[0].shape}")
            
            # Check router logits dimensions
            expected_experts = converted_config.num_experts + converted_config.num_experts_per_tok
            actual_experts = converted_outputs.router_logits[0].shape[-1]
            print(f"  Expected experts: {expected_experts}")
            print(f"  Actual experts in router logits: {actual_experts}")
            print(f"  Router logits match expected: {'✓' if actual_experts == expected_experts else '❌'}")
        
        # Clean up memory
        if memory_efficient and device != "cpu":
            print("\nCleaning up GPU memory...")
            del original_model, converted_model
            torch.cuda.empty_cache()
            print("✓ Models removed from GPU, cache cleared")
        
        # 6. Summary
        print(f"\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        total_original_weights = len(original_state_dict)
        total_converted_weights = len(converted_state_dict)
        identical_weights = len(results["identical_keys"])
        gate_weights = len(results["gate_analysis"])
        unexpected_mismatches = len(results["shape_mismatch"]) + len(results["value_mismatch"])
        
        print(f"Original model weights: {total_original_weights}")
        print(f"Converted model weights: {total_converted_weights}")
        print(f"Identical weights: {identical_weights}")
        print(f"Gate weights (expanded): {gate_weights}")
        print(f"Unexpected mismatches: {unexpected_mismatches}")
        print(f"Missing weights: {len(results['missing_in_converted']) + len(results['missing_in_original'])}")
        
        # Success criteria
        success = (
            len(results["value_mismatch"]) == 0 and  # No unexpected value mismatches
            len(results["missing_in_converted"]) == 0 and  # No missing weights
            len(results["missing_in_original"]) == 0 and  # No extra weights (except expected)
            len([item for item in results["shape_mismatch"] if "gate.weight" not in item["key"]]) == 0 and  # No unexpected shape mismatches
            all(item["first_64_match"] for item in results["gate_analysis"])  # All gate weights preserved correctly
        )
        
        if success:
            print(f"\n🎉 SUCCESS: Model comparison passed all tests!")
            print(f"   ✓ All non-gate weights are identical")
            print(f"   ✓ Gate weights preserve first 64 experts perfectly")
            print(f"   ✓ Trash can experts properly initialized")
            print(f"   ✓ Model functionality verified")
            print(f"   ✓ Converted model is ready for use!")
        else:
            print(f"\n❌ FAILURE: Some tests failed")
            if results["value_mismatch"]:
                print(f"   - {len(results['value_mismatch'])} unexpected value mismatches")
            if results["shape_mismatch"]:
                print(f"   - {len([item for item in results['shape_mismatch'] if 'gate.weight' not in item['key']])} unexpected shape mismatches")
            if any(not item["first_64_match"] for item in results["gate_analysis"]):
                print(f"   - Gate weight preservation failed")
            if results["missing_in_converted"] or results["missing_in_original"]:
                print(f"   - Missing or extra weights detected")
        
        return success
        
    except Exception as e:
        print(f"\n❌ Comparison failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the model comparison."""
    parser = argparse.ArgumentParser(description="Compare original OLMoE model with converted Select-MoE model")
    
    parser.add_argument("--original-model", type=str, default="allenai/OLMoE-1B-7B-0125",
                        help="Original model name or path")
    parser.add_argument("--converted-model", type=str, default="./converted_models/select_moe_converted_OLMoE-1B-7B-0125",
                        help="Path to converted Select-MoE model")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use (cpu, cuda, cuda:0, cuda:1, etc.)")
    parser.add_argument("--dtype", type=str, default="bfloat16", 
                        choices=["float32", "bfloat16", "float16"],
                        help="Data type for model loading")
    parser.add_argument("--memory-efficient", action="store_true",
                        help="Use memory efficient mode (load models sequentially)")
    
    args = parser.parse_args()
    
    # Validate device
    if args.device != "cpu" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        args.device = "cpu"
    elif args.device.startswith("cuda:"):
        try:
            device_id = int(args.device.split(":")[1])
            if device_id >= torch.cuda.device_count():
                print(f"⚠️  Device {args.device} not available, falling back to CPU")
                args.device = "cpu"
        except (ValueError, IndexError):
            print(f"⚠️  Invalid device format {args.device}, falling back to CPU")
            args.device = "cpu"
    
    # Validate dtype for device
    if args.device != "cpu" and args.dtype == "float16":
        print("⚠️  Warning: float16 may cause numerical issues on some GPUs")
        print("   Consider using bfloat16 instead for better stability")
    
    print(f"Configuration:")
    print(f"  Device: {args.device}")
    print(f"  Data type: {args.dtype}")
    print(f"  Memory efficient: {args.memory_efficient}")
    
    if args.device != "cpu":
        total_memory = torch.cuda.get_device_properties(args.device).total_memory / 1024**3
        print(f"  GPU memory: {total_memory:.1f} GB")
        
        # Recommend memory efficient mode for limited memory
        if total_memory < 15 and not args.memory_efficient:
            print(f"💡 Recommendation: Consider using --memory-efficient for GPU with <15GB memory")
    
    try:
        success = compare_models(
            original_model_name=args.original_model,
            converted_model_path=args.converted_model,
            device=args.device,
            dtype=args.dtype,
            memory_efficient=args.memory_efficient
        )
        print(f"\nComparison {'PASSED' if success else 'FAILED'}")
        return success
    except KeyboardInterrupt:
        print(f"\n⚠️  Comparison interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Comparison failed with unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)