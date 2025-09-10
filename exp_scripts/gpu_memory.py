#!/usr/bin/env python3
"""
Simple script to show GPU memory usage with CUDA_VISIBLE_DEVICES mapping.
"""

import os
import torch
from typing import Dict, List


def get_gpu_info() -> List[Dict]:
    """Get GPU information using torch."""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return []
    
    gpus = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        memory_allocated = torch.cuda.memory_allocated(i) // (1024 * 1024)  # MB
        memory_reserved = torch.cuda.memory_reserved(i) // (1024 * 1024)   # MB
        memory_total = props.total_memory // (1024 * 1024)                 # MB
        memory_free = memory_total - memory_reserved
        
        gpus.append({
            'index': i,
            'name': props.name,
            'memory_allocated': memory_allocated,
            'memory_reserved': memory_reserved,
            'memory_total': memory_total,
            'memory_free': memory_free
        })
    return gpus


def parse_cuda_visible_devices() -> List[int]:
    """Parse CUDA_VISIBLE_DEVICES environment variable."""
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    
    if not cuda_visible:
        return []
    
    try:
        return [int(x.strip()) for x in cuda_visible.split(',') if x.strip()]
    except ValueError:
        print(f"Invalid CUDA_VISIBLE_DEVICES format: {cuda_visible}")
        return []


def format_memory(mb: int) -> str:
    """Format memory in MB to human readable format."""
    if mb >= 1024:
        return f"{mb/1024:.1f}GB"
    return f"{mb}MB"


def main():
    print("=" * 80)
    print("GPU Memory Usage Report")
    print("=" * 80)
    
    # Get all GPU info
    all_gpus = get_gpu_info()
    if not all_gpus:
        print("No GPUs found or nvidia-smi not available")
        return
    
    # Get CUDA_VISIBLE_DEVICES mapping
    visible_gpus = parse_cuda_visible_devices()
    
    print(f"\nCUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    
    if visible_gpus:
        print(f"Visible GPU indices: {visible_gpus}")
        print("\nCUDA_VISIBLE_DEVICES Mapping:")
        print("-" * 70)
        for cuda_idx, real_idx in enumerate(visible_gpus):
            if real_idx < len(all_gpus):
                gpu = all_gpus[real_idx]
                used_pct = (gpu['memory_reserved'] / gpu['memory_total']) * 100
                print(f"CUDA:{cuda_idx} -> GPU:{real_idx} | {gpu['name'][:20]:<20} | "
                      f"Free: {format_memory(gpu['memory_free']):<8} | "
                      f"Reserved: {format_memory(gpu['memory_reserved']):<8} ({used_pct:.1f}%) | "
                      f"Total: {format_memory(gpu['memory_total'])}")
            else:
                print(f"CUDA:{cuda_idx} -> GPU:{real_idx} | ERROR: GPU index out of range")
    
    print("\nAll Available GPUs:")
    print("-" * 90)
    for gpu in all_gpus:
        used_pct = (gpu['memory_reserved'] / gpu['memory_total']) * 100
        status = "VISIBLE" if gpu['index'] in visible_gpus else "HIDDEN"
        print(f"GPU:{gpu['index']} | {gpu['name'][:25]:<25} | "
              f"Free: {format_memory(gpu['memory_free']):<8} | "
              f"Reserved: {format_memory(gpu['memory_reserved']):<8} ({used_pct:.1f}%) | "
              f"Allocated: {format_memory(gpu['memory_allocated']):<8} | "
              f"Total: {format_memory(gpu['memory_total']):<8} | {status}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()