"""
多GPU并行K-Means轮廓系数计算实现

使用多进程并行计算不同k值的轮廓系数，支持：
- 多GPU设备分配
- 进程池管理
- 内存友好的数据共享
- 容错和监控
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.multiprocessing as mp

from .gpu_metrics import gpu_silhouette_score_cosine


class ParallelKMeansSelector:
    """多GPU并行K-means选择器"""

    def __init__(self, random_state: int = 42, debug_print: bool = False):
        """
        初始化并行K-means选择器

        Args:
            random_state: 随机种子
            debug_print: 是否启用调试输出
        """
        self.random_state = random_state
        self.debug_print = debug_print
        self.logger = logging.getLogger(__name__)

    def get_available_devices(self) -> List[str]:
        """获取可用的GPU设备列表"""
        if not torch.cuda.is_available():
            self.logger.warning("CUDA不可用，无法使用GPU并行")
            return []

        # 检查CUDA_VISIBLE_DEVICES环境变量
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if visible_devices is not None:
            if visible_devices == "":
                return []
            device_ids = visible_devices.split(",")
            devices = [f"cuda:{i}" for i in range(len(device_ids))]
        else:
            num_gpus = torch.cuda.device_count()
            devices = [f"cuda:{i}" for i in range(num_gpus)]

        self.logger.info(f"检测到可用GPU设备: {devices}")
        return devices

    def allocate_processes_to_devices(self, num_processes: int, devices: List[str], strategy: str = "round_robin") -> Dict[int, str]:
        """
        将进程分配给GPU设备

        Args:
            num_processes: 总进程数
            devices: 可用设备列表
            strategy: 分配策略 ('round_robin' 或 'balanced')

        Returns:
            进程ID到设备的映射字典
        """
        if not devices:
            raise ValueError("没有可用的GPU设备")

        process_device_map = {}

        if strategy == "round_robin":
            # 轮询分配
            for i in range(num_processes):
                device = devices[i % len(devices)]
                process_device_map[i] = device
        elif strategy == "balanced":
            # 平衡分配
            processes_per_device = num_processes // len(devices)
            remainder = num_processes % len(devices)

            process_id = 0
            for device_idx, device in enumerate(devices):
                # 基础进程数
                device_processes = processes_per_device
                # 余数分配给前面的设备
                if device_idx < remainder:
                    device_processes += 1

                for _ in range(device_processes):
                    process_device_map[process_id] = device
                    process_id += 1
        else:
            raise ValueError(f"未知的分配策略: {strategy}")

        self.logger.info(f"进程-设备分配映射: {process_device_map}")
        return process_device_map

    def split_k_values(self, k_values: List[int], num_processes: int) -> List[List[int]]:
        """
        将k值列表分配给多个进程

        Args:
            k_values: k值列表
            num_processes: 进程数

        Returns:
            每个进程分配到的k值列表
        """
        if num_processes <= 0:
            raise ValueError("进程数必须大于0")

        if num_processes >= len(k_values):
            # 进程数大于等于k值数量，每个进程最多一个k值
            k_splits = [[k] for k in k_values]
            # 填充空列表到进程数
            while len(k_splits) < num_processes:
                k_splits.append([])
        else:
            # 平均分配k值
            k_per_process = len(k_values) // num_processes
            remainder = len(k_values) % num_processes

            k_splits = []
            start_idx = 0

            for i in range(num_processes):
                # 基础k值数
                current_k_count = k_per_process
                # 余数分配给前面的进程
                if i < remainder:
                    current_k_count += 1

                end_idx = start_idx + current_k_count
                k_splits.append(k_values[start_idx:end_idx])
                start_idx = end_idx

        self.logger.info(f"k值分配: {[(i, len(ks)) for i, ks in enumerate(k_splits)]}")
        return k_splits

    def find_optimal_k_elbow_parallel(
        self,
        data: torch.Tensor,
        k_range: Optional[Tuple[int, int]] = None,
        max_iters: int = 300,
        n_runs: int = 30,
        parallel_processes: int = 4,
        gpu_allocation_strategy: str = "round_robin",
    ) -> Dict:
        """
        使用多GPU并行计算Elbow Method选择最优k值

        Args:
            data: 输入数据 [N, D]
            k_range: k值搜索范围 (min_k, max_k)
            max_iters: K-Means最大迭代次数
            n_runs: 每个k值运行次数
            parallel_processes: 并行进程数
            gpu_allocation_strategy: GPU分配策略

        Returns:
            包含最优k值和相关指标的字典
        """
        self.logger.info(f"开始多GPU并行Elbow Method，进程数: {parallel_processes}")

        # 1. 准备k值列表
        n_samples = data.shape[0]
        if k_range is None:
            min_k = max(2, int(n_samples**0.5 / 10))
            max_k = min(int(n_samples**0.5), 200)
            k_range = (min_k, max_k)

        min_k, max_k = k_range
        k_values = list(range(min_k, max_k + 1, max(1, (max_k - min_k) // 20)))

        self.logger.info(f"k值搜索范围: [{min_k}, {max_k}]")
        self.logger.info(f"候选k值: {k_values}")

        # 2. 获取可用设备并分配进程
        available_devices = self.get_available_devices()
        if not available_devices:
            raise RuntimeError("没有可用的GPU设备进行并行计算")

        # 限制进程数不超过k值数量
        effective_processes = min(parallel_processes, len(k_values))
        self.logger.info(f"有效并行进程数: {effective_processes}")

        # 进程-设备分配
        process_device_map = self.allocate_processes_to_devices(effective_processes, available_devices, gpu_allocation_strategy)

        # k值分配
        k_value_splits = self.split_k_values(k_values, effective_processes)

        # 3. 准备共享数据
        # 将数据移到CPU并共享内存
        shared_data = data.cpu().share_memory_()

        # 4. 启动并行计算
        self.logger.info("启动并行工作进程...")

        # 使用spawn方法启动进程（CUDA要求）
        mp.set_start_method("spawn", force=True)

        with mp.Pool(effective_processes) as pool:
            # 准备参数列表
            args_list = []
            for process_id in range(effective_processes):
                args = (
                    process_id,
                    shared_data,
                    k_value_splits[process_id],
                    process_device_map[process_id],
                    max_iters,
                    n_runs,
                    self.random_state + process_id,  # 每个进程使用不同的随机种子
                    self.debug_print,
                )
                args_list.append(args)

            # 启动异步计算
            async_results = [pool.apply_async(_worker_process, args) for args in args_list]

            # 监控进度
            if self.debug_print:
                self.logger.info("等待工作进程完成...")
                for i, async_result in enumerate(async_results):
                    try:
                        async_result.wait(timeout=10)  # 10秒超时检查
                        if async_result.ready():
                            self.logger.info(f"进程 {i} 已完成")
                    except mp.TimeoutError:
                        self.logger.info(f"进程 {i} 仍在运行中...")

            # 收集结果
            results = []
            for i, async_result in enumerate(async_results):
                try:
                    result = async_result.get(timeout=3600)  # 1小时超时
                    results.append(result)
                    self.logger.info(f"成功收集进程 {i} 的结果")
                except Exception as e:
                    self.logger.error(f"进程 {i} 执行失败: {e}")
                    raise

        # 5. 汇总结果
        self.logger.info("汇总并行计算结果...")
        return self._aggregate_results(results, k_values)

    def _aggregate_results(self, results: List[Dict], all_k_values: List[int]) -> Dict:
        """汇总多进程计算结果"""
        # 合并所有结果
        all_inertias = {}
        all_silhouette_scores = {}

        for result in results:
            for k, inertia, silhouette in zip(result["k_values"], result["inertias"], result["silhouette_scores"], strict=True):
                all_inertias[k] = inertia
                all_silhouette_scores[k] = silhouette

        # 按k值排序
        sorted_k_values = sorted(all_inertias.keys())
        inertias = [all_inertias[k] for k in sorted_k_values]
        silhouette_scores = [all_silhouette_scores[k] for k in sorted_k_values]

        self.logger.info(f"汇总结果: 计算了 {len(sorted_k_values)} 个k值")

        # 寻找Elbow点
        optimal_k = self._find_elbow_point(sorted_k_values, inertias)

        # 找到最佳轮廓系数对应的k
        if silhouette_scores:
            best_silhouette_idx = max(range(len(silhouette_scores)), key=lambda i: silhouette_scores[i])
            best_silhouette_k = sorted_k_values[best_silhouette_idx]
        else:
            best_silhouette_k = optimal_k

        self.logger.info(f"并行Elbow方法推荐k={optimal_k}")
        self.logger.info(f"并行轮廓系数最高k={best_silhouette_k}")

        return {
            "optimal_k_elbow": optimal_k,
            "optimal_k_silhouette": best_silhouette_k,
            "k_values": sorted_k_values,
            "inertias": inertias,
            "silhouette_scores": silhouette_scores,
            "recommended_k": optimal_k,
        }

    def _find_elbow_point(self, k_values: List[int], inertias: List[float]) -> int:
        """寻找Elbow点（与串行版本相同的逻辑）"""
        if len(inertias) < 3:
            return k_values[0]

        # 计算二阶差分
        inertias_tensor = torch.tensor(inertias, dtype=torch.float32)

        # 标准化惯性值到[0,1]范围
        inertias_norm = (inertias_tensor - inertias_tensor.min()) / (inertias_tensor.max() - inertias_tensor.min() + 1e-8)

        # 计算一阶和二阶差分
        first_diff = inertias_norm[1:] - inertias_norm[:-1]
        second_diff = first_diff[1:] - first_diff[:-1]

        # 寻找二阶差分最大的点（最大曲率）
        if len(second_diff) > 0:
            elbow_idx = second_diff.argmax().item() + 1  # +1因为二阶差分索引偏移
            if elbow_idx < len(k_values):
                return k_values[elbow_idx]

        # 如果找不到明显的肘部，返回中间值
        return k_values[len(k_values) // 2]


def _worker_process(
    process_id: int,
    shared_data: torch.Tensor,
    k_values: List[int],
    device: str,
    max_iters: int,
    n_runs: int,
    random_state: int,
    debug_print: bool,
) -> Dict:
    """
    工作进程函数：计算分配的k值的轮廓系数

    Args:
        process_id: 进程ID
        shared_data: 共享的数据张量
        k_values: 分配给此进程的k值列表
        device: 分配的GPU设备
        max_iters: K-means最大迭代次数
        n_runs: 每个k值运行次数
        random_state: 随机种子
        debug_print: 是否启用调试输出

    Returns:
        包含计算结果的字典
    """
    # 设置进程级日志
    logger = logging.getLogger(f"worker_{process_id}")

    try:
        if debug_print:
            logger.info(f"进程 {process_id} 开始，设备: {device}, k值: {k_values}")

        # 设置设备
        torch.cuda.set_device(device)
        device_obj = torch.device(device)

        # 将数据移到指定设备
        data = shared_data.to(device_obj, dtype=torch.float32)

        # 导入K-means类（避免循环导入）
        from .kmeans_clustering import GPUKMeansClustering

        # 创建K-means聚类器
        kmeans = GPUKMeansClustering(device_obj, random_state, debug_print)

        # 计算分配的k值
        inertias = []
        silhouette_scores = []

        for k in k_values:
            if debug_print:
                logger.info(f"进程 {process_id} 计算 k={k}")

            # 多次运行选择最佳结果
            best_inertia = float("inf")
            best_labels = None

            for _run in range(n_runs):
                centers, labels = kmeans.kmeans_cosine(data, k, max_iters=max_iters)
                inertia = kmeans.compute_inertia_cosine(data, centers, labels)

                if inertia < best_inertia:
                    best_inertia = inertia
                    best_labels = labels

            inertias.append(best_inertia)

            # 计算轮廓系数
            if k > 1 and len(torch.unique(best_labels)) > 1:
                silhouette_score = gpu_silhouette_score_cosine(data, best_labels)
                silhouette_scores.append(silhouette_score)
            else:
                silhouette_scores.append(0.0)

            if debug_print:
                logger.info(f"进程 {process_id} k={k}: 惯性={best_inertia:.3f}, 轮廓系数={silhouette_scores[-1]:.3f}")

        result = {"k_values": k_values, "inertias": inertias, "silhouette_scores": silhouette_scores}

        if debug_print:
            logger.info(f"进程 {process_id} 完成，返回 {len(k_values)} 个结果")

        return result

    except Exception as e:
        logger.error(f"进程 {process_id} 发生错误: {e}")
        raise
