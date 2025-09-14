"""
GPU Memory Occupation Script

This script creates tensors of specified size on each GPU specified by CUDA_VISIBLE_DEVICES
and holds them in memory until the user inputs 'stop' or manually terminates the script.
Optionally keeps GPUs busy with computation to maintain utilization.

Usage:
    # Occupy 10GB on each GPU
    CUDA_VISIBLE_DEVICES=0,1,2,3 python exp_scripts/occupy_gpu_memory.py --memory-size 10

    # Occupy 8GB and keep GPUs busy with computation
    CUDA_VISIBLE_DEVICES=0,1,2,3 python exp_scripts/occupy_gpu_memory.py --memory-size 8 --busy-work

    # Use default 10GB without busy work
    CUDA_VISIBLE_DEVICES=0,1,2,3 python exp_scripts/occupy_gpu_memory.py
"""

import argparse
import os
import threading
import time
from typing import List

import torch


def occupy_gpu(device_id: int, tensors: List[torch.Tensor], stop_event: threading.Event, memory_gb: float = 10.0, enable_busy_work: bool = False):
    """Occupy GPU memory by creating a tensor of specified size on the specified device.

    Args:
        device_id: GPU device ID
        tensors: List to store created tensors
        stop_event: Threading event to signal when to stop
        memory_gb: Memory size in GB to occupy
        enable_busy_work: Whether to keep GPU busy with computation
    """
    try:
        device = torch.device(f"cuda:{device_id}")

        # Calculate available memory for main tensor
        # If busy work is enabled, reserve memory for computation tensors
        available_memory_gb = memory_gb
        if enable_busy_work:
            # Reserve memory for busy work tensors:
            # batch_size=32, matrix_size=2048: 32*2048*2048*4 bytes ≈ 0.5GB per tensor
            # We need 4 tensors (a, b, c, d), so ~2GB total
            busy_work_memory_gb = 2.5  # Conservative estimate with buffer
            if memory_gb <= busy_work_memory_gb:
                print(f"Warning: Memory size {memory_gb}GB too small for busy work. Using minimal busy work.")
                available_memory_gb = memory_gb * 0.3  # Use only 30% for main tensor
                use_minimal_busy_work = True
            else:
                available_memory_gb = memory_gb - busy_work_memory_gb
                use_minimal_busy_work = False
        else:
            use_minimal_busy_work = False

        # Calculate tensor size for main memory occupation
        # float32 takes 4 bytes per element
        tensor_size = int((available_memory_gb * 1024**3) // 4)

        print(f"Creating {available_memory_gb:.1f}GB main tensor on GPU {device_id}...")

        # Create the main tensor and move it to GPU
        tensor = torch.randn(tensor_size, dtype=torch.float32, device=device)
        tensors.append(tensor)

        print(f"✓ Successfully occupied ~{memory_gb}GB on GPU {device_id}")

        # Keep the tensor alive until stop event is set
        if enable_busy_work:
            print(f"Starting intensive computation on GPU {device_id}")

            if use_minimal_busy_work:
                # Use smaller tensors for low memory situations
                batch_size = 8
                matrix_size = 512
            else:
                # Use larger tensors for high-intensity computation
                batch_size = 32
                matrix_size = 2048

            a = torch.randn(batch_size, matrix_size, matrix_size, device=device, dtype=torch.float32)
            b = torch.randn(batch_size, matrix_size, matrix_size, device=device, dtype=torch.float32)

            # Additional tensors for more operations
            c = torch.randn(batch_size, matrix_size, device=device, dtype=torch.float32)
            d = torch.randn(batch_size, matrix_size, device=device, dtype=torch.float32)

            iteration = 0
            while not stop_event.is_set():
                # High-intensity matrix operations
                # Batch matrix multiplication
                result1 = torch.bmm(a, b)

                # Element-wise operations
                result2 = torch.sin(result1) + torch.cos(result1)
                result3 = torch.exp(torch.clamp(result2 * 0.01, -10, 10))  # Clamp to prevent overflow

                # More matrix operations
                result4 = torch.sum(result3, dim=-1)
                result5 = torch.matmul(result4, c.unsqueeze(-1)).squeeze(-1)

                # Use result to prevent optimization
                _ = torch.matmul(result5.unsqueeze(1), d.unsqueeze(0))

                # Update tensors periodically to prevent optimization
                if iteration % 50 == 0:
                    noise_scale = 0.001
                    a = a + torch.randn_like(a) * noise_scale
                    b = b + torch.randn_like(b) * noise_scale
                    c = c + torch.randn_like(c) * noise_scale
                    d = d + torch.randn_like(d) * noise_scale

                iteration += 1
        else:
            while not stop_event.is_set():
                time.sleep(1)

    except Exception as e:
        print(f"✗ Error occupying GPU {device_id}: {e}")
    finally:
        print(f"Releasing memory on GPU {device_id}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="GPU Memory Occupation Script")
    parser.add_argument("--memory-size", type=float, default=34.5, help="Memory size in GB to occupy per GPU (default: 10.0)")
    parser.add_argument("--busy-work", action="store_true", help="Keep GPUs busy with computation to maintain utilization")
    return parser.parse_args()


def main():
    """Main function to manage GPU memory occupation."""
    args = parse_args()
    if not torch.cuda.is_available():
        print("✗ CUDA is not available!")
        return

    # Use all visible GPUs (CUDA_VISIBLE_DEVICES already filters them)
    available_gpus = torch.cuda.device_count()
    gpu_ids = list(range(available_gpus))

    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
    print(f"Visible GPUs: {gpu_ids}")
    print(f"Total GPUs to occupy: {len(gpu_ids)}")
    print(f"Memory per GPU: ~{args.memory_size}GB")
    print(f"Total memory to occupy: ~{len(gpu_ids) * args.memory_size}GB")
    print(f"Busy work enabled: {args.busy_work}")
    print("-" * 50)

    tensors = []
    threads = []
    stop_event = threading.Event()

    try:
        # Create threads to occupy each GPU
        for gpu_id in gpu_ids:
            thread = threading.Thread(target=occupy_gpu, args=(gpu_id, tensors, stop_event, args.memory_size, False))
            thread.start()
            threads.append(thread)

        print("\n" + "=" * 50)
        print("GPU memory occupation started!")
        print("Type 'stop' and press Enter to release memory and exit")
        print("Or use Ctrl+C to force quit")
        print("=" * 50)

        # Wait for user input or KeyboardInterrupt
        while True:
            try:
                user_input = input().strip().lower()
                if user_input == "stop":
                    print("\nReceived stop command. Releasing memory...")
                    break
            except EOFError:
                # Handle case where input is not available (e.g., non-interactive mode)
                print("\nNon-interactive mode detected. Use Ctrl+C to stop.")
                while True:
                    time.sleep(1)

    except KeyboardInterrupt:
        print("\n\nReceived interrupt signal. Releasing memory...")

    finally:
        # Signal all threads to stop
        stop_event.set()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Clear tensors explicitly
        tensors.clear()

        # Force garbage collection and clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("\n✓ All GPU memory released successfully!")
        print("Script terminated.")


if __name__ == "__main__":
    main()
