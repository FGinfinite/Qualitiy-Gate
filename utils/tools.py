import time
import torch
from datetime import datetime

def grab_gpu(memory_need: float):
    """
    监控并抢占所有可用GPU上的指定显存量。

    该函数持续检查可用的GPU，并尝试分配一个指定大小（单位为GB）的张量。
    它利用PyTorch的内存缓存机制为当前进程预留显存。张量在创建后会立即被删除，
    但其占用的显存块会保持缓存状态。

    Args:
        memory_need (float): 需要抢占的GPU显存量（单位: GB）。
    """
    if not torch.cuda.is_available():
        print("CUDA 不可用。无法抢占GPU。")
        return

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("未找到任何GPU。")
        return
        
    grabbed_gpus = set()
    # 将GB转换为字节
    memory_need_bytes = int(memory_need * 1024 * 1024 * 1024)

    print(f"开始在 {num_gpus} 个可用GPU上抢占 {memory_need} GB 显存...")
    
    last_log_time = time.time()

    while len(grabbed_gpus) < num_gpus:
        for i in range(num_gpus):
            if i in grabbed_gpus:
                continue

            try:
                free_memory, _ = torch.cuda.mem_get_info(i)
                if free_memory > memory_need_bytes:
                    print(f"GPU {i}: 可用显存 ({free_memory / (1024**3):.2f} GB) > 所需 ({memory_need} GB)。正在抢占...")
                    # 创建一个张量来占用显存。
                    # 大小根据字节计算。一个float32张量每个元素占用4个字节。
                    tensor_size = memory_need_bytes // 4
                    tensor = torch.empty(tensor_size, dtype=torch.float32, device=f'cuda:{i}')
                    
                    # 此刻，显存已被PyTorch为此进程缓存。
                    # 我们可以删除张量变量，但显存仍分配给该进程。
                    del tensor
                    
                    grabbed_gpus.add(i)
                    print(f"成功在 GPU {i} 上抢占 {memory_need} GB 显存。")
            except Exception as e:
                print(f"尝试在 GPU {i} 上抢占显存时发生错误: {e}")

        if len(grabbed_gpus) < num_gpus:
            current_time = time.time()
            if current_time - last_log_time >= 300: # 每5分钟报告一次
                waiting_for_gpus = [i for i in range(num_gpus) if i not in grabbed_gpus]
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"[{timestamp}] 状态报告: 已抢占 {len(grabbed_gpus)}/{num_gpus} 个GPU。仍在等待GPU: {waiting_for_gpus}。")
                last_log_time = current_time
            time.sleep(1) # 降低检查频率

    print(f"成功在所有 {num_gpus} 个可用GPU上抢占了 {memory_need} GB 显存。")
