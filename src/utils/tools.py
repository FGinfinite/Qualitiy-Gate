import time
from datetime import datetime

import torch
from accelerate import Accelerator


def grab_gpu(memory_need: float, accelerator: Accelerator | None = None, over_grab: bool = False):
    """
    通过创建并返回一个持久的“占位符”张量来抢占GPU显存。

    该函数会持续等待，直到有足够的连续显存来创建一个指定大小的张量。
    然后它会返回这个张量。调用者有责任持有这个张量，并在需要使用
    这部分显存之前手动释放它 (`del placeholder; torch.cuda.empty_cache()`)。

    - 在 `accelerate` 模式下，每个进程抢占并返回其对应GPU的一个占位符。
    - 在单进程模式下，函数会为每个可见的GPU创建一个占位符，并以列表形式返回。

    Args:
        memory_need (float): 需要为每个GPU抢占的显存量（单位: GB）。
        accelerator (Accelerator | None, optional): accelerate的Accelerator对象。
            如果提供，则启用多进程模式。默认为 `None`。

    Returns:
        torch.Tensor | list[torch.Tensor] | None:
        - 在 accelerate 模式下，返回单个占位符张量。
        - 在单进程模式下，返回一个占位符张量列表。
        - 如果CUDA不可用或发生错误，返回 None。
    """
    if not torch.cuda.is_available():
        print("CUDA 不可用。无法抢占GPU。")
        return None

    memory_need_in_bytes = int(memory_need * 1024 * 1024 * 1024)
    num_gpus = torch.cuda.device_count()

    def _grab_single_gpu(device_index):
        """辅助函数，用于在单个GPU上抢占显存并返回占位符。"""
        print(f"进程 {accelerator.process_index if accelerator else 'main'}: 开始在 GPU {device_index} 上等待并抢占 {memory_need} GB 显存...")
        last_log_time = time.time()
        while True:
            try:
                free, _ = torch.cuda.mem_get_info(device_index)

                # 根据 over_grab 标志决定分配策略
                if over_grab:
                    safety_margin = 200 * 1024 * 1024  # 200MB 安全边际
                    size_to_allocate = free - safety_margin
                else:
                    size_to_allocate = memory_need_in_bytes

                if free >= size_to_allocate > 0:
                    placeholder = torch.empty(
                        size_to_allocate // 4,  # float32 is 4 bytes
                        dtype=torch.float32,
                        device=f"cuda:{device_index}",
                    )
                    allocated_gb = placeholder.nbytes / (1024**3)
                    print(f"进程 {accelerator.process_index if accelerator else 'main'}: 成功在 GPU {device_index} 上分配了 {allocated_gb:.2f} GB 的占位符。")
                    return placeholder
                else:
                    current_time = time.time()
                    if current_time - last_log_time >= 10:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        free_gb = free / (1024**3)
                        print(
                            f"[{timestamp}] 进程 {accelerator.process_index if accelerator else 'main'}: "
                            f"GPU {device_index}: 连续空间不足 (可用: {free_gb:.2f}GB / "
                            f"需要: {memory_need:.2f}GB)。正在等待..."
                        )
                        last_log_time = current_time
            except Exception as e:
                print(f"进程 {accelerator.process_index if accelerator else 'main'}: 尝试在 GPU {device_index} 上分配占位符时发生错误: {e}")
            time.sleep(1)

    if accelerator is not None:
        # Accelerate 多进程模式
        device_index = accelerator.process_index
        if device_index >= num_gpus:
            print(f"进程 {device_index}: 目标 GPU 索引超出可用范围 ({num_gpus} 个)。跳过抢占。")
            # 仍然需要等待，以防其他进程正在抢占
            accelerator.wait_for_everyone()
            return None

        placeholder = _grab_single_gpu(device_index)

        print(f"进程 {device_index}: 等待所有其他进程完成显存抢占...")
        accelerator.wait_for_everyone()
        return placeholder
    else:
        # 单进程模式
        placeholders = []
        for device_index in range(num_gpus):
            placeholder = _grab_single_gpu(device_index)
            if placeholder:
                placeholders.append(placeholder)
        print(f"所有 {len(placeholders)} 个GPU均已成功分配占位符。")
        return placeholders
