"""
GPU Memory Occupation Script

This script creates 10GB tensors on each GPU specified by CUDA_VISIBLE_DEVICES
and holds them in memory until the user inputs 'stop' or manually terminates the script.

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python exp_scripts/occupy_gpu_memory.py
"""

import os
import threading
import time
from typing import List

import torch


def occupy_gpu(device_id: int, tensors: List[torch.Tensor], stop_event: threading.Event):
    """Occupy GPU memory by creating a 10GB tensor on the specified device."""
    try:
        device = torch.device(f"cuda:{device_id}")

        # Calculate tensor size for approximately 10GB
        # float32 takes 4 bytes per element
        # 10GB = 10 * 1024^3 bytes / 4 bytes per element
        tensor_size = (6 * 1024**3) // 4

        print(f"Creating 10GB tensor on GPU {device_id}...")

        # Create the tensor and move it to GPU
        tensor = torch.randn(tensor_size, dtype=torch.float32, device=device)
        tensors.append(tensor)

        print(f"✓ Successfully occupied ~10GB on GPU {device_id}")

        # Keep the tensor alive until stop event is set
        while not stop_event.is_set():
            time.sleep(1)

    except Exception as e:
        print(f"✗ Error occupying GPU {device_id}: {e}")
    finally:
        print(f"Releasing memory on GPU {device_id}")


def main():
    """Main function to manage GPU memory occupation."""
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
    print(f"Memory per GPU: ~10GB")
    print(f"Total memory to occupy: ~{len(gpu_ids) * 10}GB")
    print("-" * 50)

    tensors = []
    threads = []
    stop_event = threading.Event()

    try:
        # Create threads to occupy each GPU
        for gpu_id in gpu_ids:
            thread = threading.Thread(target=occupy_gpu, args=(gpu_id, tensors, stop_event))
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
