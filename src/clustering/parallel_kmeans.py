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
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.multiprocessing as mp

from .gpu_metrics import gpu_silhouette_score_cosine


class ParallelKMeansSelector:
    """多GPU并行K-means选择器"""

    def __init__(self, random_state: int = 42, debug_print: bool = False, output_dir: Optional[str] = None):
        """
        初始化并行K-means选择器

        Args:
            random_state: 随机种子
            debug_print: 是否启用调试输出
            output_dir: 输出目录，用于创建子进程日志文件
        """
        self.random_state = random_state
        self.debug_print = debug_print
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)

        # 创建子进程日志目录
        if output_dir:
            self.sub_logs_dir = Path(output_dir) / "sub_logs"
            self.sub_logs_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"子进程日志目录: {self.sub_logs_dir}")
        else:
            self.sub_logs_dir = None

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

    def split_k_values(self, k_values: List[int], num_processes: int, strategy: str = "interleaved") -> List[List[int]]:
        """
        将k值列表分配给多个进程

        Args:
            k_values: k值列表
            num_processes: 进程数
            strategy: 分配策略 ('sequential' 或 'interleaved')

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
            if strategy == "sequential":
                # 原有的连续分配策略
                k_splits = self._split_k_values_sequential(k_values, num_processes)
            elif strategy == "interleaved":
                # 新的交替分配策略，平衡计算负载
                k_splits = self._split_k_values_interleaved(k_values, num_processes)
            else:
                raise ValueError(f"未知的分配策略: {strategy}")

        # 计算负载均衡度（基于k值的平方，因为轮廓系数计算复杂度约为O(n²k)）
        loads = [sum(k**2 for k in ks) for ks in k_splits]
        if loads:
            max_load = max(loads)
            min_load = min(loads)
            balance_ratio = min_load / max_load if max_load > 0 else 1.0
        else:
            balance_ratio = 1.0

        self.logger.info(f"k值分配策略: {strategy}")
        self.logger.info(f"k值分配: {[(i, len(ks)) for i, ks in enumerate(k_splits)]}")
        self.logger.info(f"负载均衡度: {balance_ratio:.3f} (越接近1.0越均衡)")
        if self.debug_print:
            for i, (ks, load) in enumerate(zip(k_splits, loads, strict=True)):
                self.logger.debug(f"进程 {i}: k值={ks}, 估计负载={load}")

        return k_splits

    def _split_k_values_sequential(self, k_values: List[int], num_processes: int) -> List[List[int]]:
        """原有的连续分配策略"""
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

        return k_splits

    def _split_k_values_interleaved(self, k_values: List[int], num_processes: int) -> List[List[int]]:
        """
        交替分配策略：使用蛇形模式分配k值，平衡计算负载

        例如4个进程分配10个k值:
        进程0: [k0, k7, k8]     # 第1轮取k0，第2轮倒序取k7，第3轮正序取k8
        进程1: [k1, k6, k9]     # 第1轮取k1，第2轮倒序取k6，第3轮正序取k9
        进程2: [k2, k5]         # 第1轮取k2，第2轮倒序取k5
        进程3: [k3, k4]         # 第1轮取k3，第2轮倒序取k4
        """
        k_splits = [[] for _ in range(num_processes)]

        idx = 0
        round_num = 0

        while idx < len(k_values):
            if round_num % 2 == 0:
                # 偶数轮：正序分配 (0, 1, 2, 3)
                for process_id in range(num_processes):
                    if idx < len(k_values):
                        k_splits[process_id].append(k_values[idx])
                        idx += 1
            else:
                # 奇数轮：倒序分配 (3, 2, 1, 0)
                for process_id in range(num_processes - 1, -1, -1):
                    if idx < len(k_values):
                        k_splits[process_id].append(k_values[idx])
                        idx += 1

            round_num += 1

        return k_splits

    def find_optimal_k_elbow_parallel(
        self,
        data: torch.Tensor,
        k_range: Optional[Tuple[int, int]] = None,
        max_iters: int = 300,
        n_runs: int = 30,
        parallel_processes: int = 4,
        gpu_allocation_strategy: str = "round_robin",
        k_distribution_strategy: str = "interleaved",
        base_timeout_hours: float = 2.0,
        per_k_timeout_hours: float = 4.0,
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
            k_distribution_strategy: k值分配策略 ('sequential' 或 'interleaved')
            base_timeout_hours: 基础超时时间（小时）
            per_k_timeout_hours: 每个k值额外超时时间（小时）

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
        # k_values = list(range(min_k, max_k + 1, max(1, (max_k - min_k) // 20)))
        k_values = list(range(min_k, max_k + 1))  # 更细粒度地调优k值

        # 计算动态超时时间
        total_timeout_seconds = int((base_timeout_hours + len(k_values) * per_k_timeout_hours / parallel_processes) * 3600)

        self.logger.info(f"k值搜索范围: [{min_k}, {max_k}]")
        self.logger.info(f"候选k值: {k_values}")
        self.logger.info(f"估计总超时时间: {total_timeout_seconds / 3600:.1f} 小时")

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
        k_value_splits = self.split_k_values(k_values, effective_processes, k_distribution_strategy)

        # 3. 准备共享数据和时间戳
        # 将数据移到CPU并共享内存
        shared_data = data.cpu().share_memory_()

        # 生成时间戳用于日志文件命名
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # 4. 启动并行计算
        self.logger.info("启动并行工作进程...")

        # 使用spawn方法启动进程（CUDA要求）
        mp.set_start_method("spawn", force=True)

        with mp.Pool(effective_processes) as pool:
            # 准备参数列表
            args_list = []
            for process_id in range(effective_processes):
                # 为每个进程准备日志文件路径
                log_file_path = None
                if self.sub_logs_dir:
                    log_file_path = str(self.sub_logs_dir / f"worker_{process_id}_{timestamp}.log")

                args = (
                    process_id,
                    shared_data,
                    k_value_splits[process_id],
                    process_device_map[process_id],
                    max_iters,
                    n_runs,
                    self.random_state + process_id,  # 每个进程使用不同的随机种子
                    self.debug_print,
                    log_file_path,  # 子进程日志文件路径
                )
                args_list.append(args)

            # 启动异步计算
            async_results = [pool.apply_async(_worker_process, args) for args in args_list]

            # 增强进度监控
            self.logger.info("等待工作进程完成...")

            # 定期检查进程状态
            check_interval = 30  # 30秒检查一次
            start_time = time.time()

            while True:
                completed_count = sum(1 for result in async_results if result.ready())
                elapsed_time = time.time() - start_time

                self.logger.info(f"进度: {completed_count}/{effective_processes} 进程完成, 已用时: {elapsed_time / 3600:.2f} 小时")

                if completed_count == effective_processes:
                    break

                # 检查是否接近超时
                if elapsed_time > total_timeout_seconds * 0.8:
                    self.logger.warning(
                        f"已达到80%超时时间 ({total_timeout_seconds / 3600:.1f}小时), 还有 {effective_processes - completed_count} 个进程未完成"
                    )

                # 等待一段时间后再次检查
                try:
                    # 短暂等待，避免无限循环占用CPU
                    time.sleep(min(check_interval, total_timeout_seconds - elapsed_time))
                except KeyboardInterrupt:
                    self.logger.warning("收到中断信号，尝试终止所有进程...")
                    pool.terminate()
                    break

                if elapsed_time > total_timeout_seconds:
                    self.logger.error(f"超过总超时时间 {total_timeout_seconds / 3600:.1f} 小时，终止剩余进程")
                    break

            # 收集结果
            results = []
            for i, async_result in enumerate(async_results):
                try:
                    # 使用剩余时间作为超时
                    remaining_timeout = max(60, total_timeout_seconds - (time.time() - start_time))
                    result = async_result.get(timeout=remaining_timeout)
                    results.append(result)
                    self.logger.info(f"成功收集进程 {i} 的结果")
                except mp.TimeoutError:
                    self.logger.error(f"进程 {i} 超时，请检查子进程日志文件")
                    if self.sub_logs_dir:
                        log_file = self.sub_logs_dir / f"worker_{i}_{timestamp}.log"
                        if log_file.exists():
                            self.logger.error(f"进程 {i} 日志文件: {log_file}")
                    # 继续处理其他进程的结果，不抛出异常
                    continue
                except Exception as e:
                    self.logger.error(f"进程 {i} 执行失败: {e}")
                    if self.sub_logs_dir:
                        log_file = self.sub_logs_dir / f"worker_{i}_{timestamp}.log"
                        if log_file.exists():
                            self.logger.error(f"进程 {i} 日志文件: {log_file}")
                            # 尝试读取日志文件的最后几行以获取错误信息
                            try:
                                with open(log_file, "r", encoding="utf-8") as f:
                                    lines = f.readlines()
                                    if lines:
                                        self.logger.error(f"进程 {i} 最后的日志输出:")
                                        for line in lines[-10:]:  # 显示最后10行
                                            self.logger.error(f"  {line.strip()}")
                            except Exception:
                                pass
                    # 继续处理其他进程的结果，不抛出异常
                    continue

            # 检查是否有任何成功的结果
            if not results:
                self.logger.error("所有子进程都失败了，无法继续")
                if self.sub_logs_dir:
                    self.logger.error(f"请检查子进程日志文件在: {self.sub_logs_dir}")
                raise RuntimeError("所有并行计算进程都失败了")

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

        # 打印出所有k值对应的轮廓系数
        if silhouette_scores:
            self.logger.info("------ 各k值的轮廓系数详情 ------")
            for k, score in zip(sorted_k_values, silhouette_scores):
                # 使用 f-string 格式化输出，保留4位小数使结果更易读
                self.logger.info(f"  k = {k:<2} | 轮廓系数 (Silhouette Score) = {score:.4f}")
            self.logger.info("------------------------------------")

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
    log_file_path: Optional[str] = None,
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
        log_file_path: 子进程日志文件路径

    Returns:
        包含计算结果的字典
    """
    # 设置进程级日志
    logger = logging.getLogger(f"worker_{process_id}")

    # 如果提供了日志文件路径，设置文件日志处理器
    file_handler = None
    if log_file_path:
        try:
            # 创建文件处理器
            file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)

            # 设置详细的日志格式
            formatter = logging.Formatter("[%(asctime)s] [PID %(process)d] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
            file_handler.setFormatter(formatter)

            # 清除之前的处理器并添加新的
            logger.handlers.clear()
            logger.addHandler(file_handler)
            logger.setLevel(logging.DEBUG)

            # 防止日志消息传播到父日志器
            logger.propagate = False

            # 记录进程开始信息
            logger.info(f"====== 子进程 {process_id} 开始 ======")
            logger.info(f"分配的GPU设备: {device}")
            logger.info(f"分配的k值: {k_values}")
            logger.info(f"随机种子: {random_state}")
            logger.info(f"Python版本: {sys.version}")
            logger.info(f"PyTorch版本: {torch.__version__}")

        except Exception as e:
            # 如果日志文件创建失败，回退到默认日志
            print(f"[Worker {process_id}] 无法创建日志文件 {log_file_path}: {e}")
            logger = logging.getLogger(f"worker_{process_id}")

    try:
        start_time = time.time()

        if debug_print or log_file_path:
            logger.info(f"进程 {process_id} 开始工作")
            logger.info(f"CUDA可用性: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logger.info(f"CUDA设备数量: {torch.cuda.device_count()}")

        # 设置设备
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
            device_obj = torch.device(device)
            if debug_print or log_file_path:
                logger.info(f"设置当前CUDA设备: {device}")
                logger.info(f"GPU内存信息: {torch.cuda.get_device_properties(device)}")
        else:
            device_obj = torch.device("cpu")
            logger.warning("未检测到CUDA，使用CPU计算")

        # 将数据移到指定设备
        data = shared_data.to(device_obj, dtype=torch.float32)
        if debug_print or log_file_path:
            logger.info(f"数据形状: {data.shape}，设备: {data.device}")

        # 导入K-means类（避免循环导入）
        from .kmeans_clustering import GPUKMeansClustering

        # 创建K-means聚类器
        kmeans = GPUKMeansClustering(device_obj, random_state, debug_print or bool(log_file_path))

        # 计算分配的k值
        inertias = []
        silhouette_scores = []

        if debug_print or log_file_path:
            logger.info(f"开始计算 {len(k_values)} 个k值")

        for i, k in enumerate(k_values):
            k_start_time = time.time()

            if debug_print or log_file_path:
                logger.info(f"==== 计算 k={k} ({i + 1}/{len(k_values)}) ====")

            # 多次运行选择最佳结果
            best_inertia = float("inf")
            best_labels = None

            if debug_print or log_file_path:
                logger.info(f"开始 {n_runs} 次运行")

            for run in range(n_runs):
                run_start_time = time.time()

                try:
                    centers, labels = kmeans.kmeans_cosine(data, k, max_iters=max_iters)
                    inertia = kmeans.compute_inertia_cosine(data, centers, labels)

                    if inertia < best_inertia:
                        best_inertia = inertia
                        best_labels = labels

                    run_time = time.time() - run_start_time
                    if debug_print or log_file_path:
                        logger.debug(f"  运行 {run + 1}: 惯性={inertia:.6f}, 用时={run_time:.2f}s")

                except Exception as e:
                    logger.error(f"  运行 {run + 1} 失败: {e}")
                    logger.error(f"  错误详情: {traceback.format_exc()}")
                    continue

            if best_labels is None:
                logger.error(f"k={k} 的所有运行都失败了")
                inertias.append(float("inf"))
                silhouette_scores.append(0.0)
                continue

            inertias.append(best_inertia)

            # 计算轮廓系数
            if k > 1 and len(torch.unique(best_labels)) > 1:
                try:
                    silhouette_start_time = time.time()
                    silhouette_score = gpu_silhouette_score_cosine(data, best_labels)
                    silhouette_time = time.time() - silhouette_start_time
                    silhouette_scores.append(silhouette_score)

                    if debug_print or log_file_path:
                        logger.info(f"  轮廓系数: {silhouette_score:.6f}, 计算时间: {silhouette_time:.2f}s")

                except Exception as e:
                    logger.error(f"  计算轮廓系数失败: {e}")
                    logger.error(f"  错误详情: {traceback.format_exc()}")
                    silhouette_scores.append(0.0)
            else:
                silhouette_scores.append(0.0)
                if debug_print or log_file_path:
                    logger.info(f"  k={k} 太小或聚类无效，轮廓系数设为0")

            k_total_time = time.time() - k_start_time
            if debug_print or log_file_path:
                logger.info(f"k={k} 完成: 惯性={best_inertia:.6f}, 轮廓系数={silhouette_scores[-1]:.6f}, 总时间={k_total_time:.2f}s")

        total_time = time.time() - start_time
        result = {"k_values": k_values, "inertias": inertias, "silhouette_scores": silhouette_scores}

        if debug_print or log_file_path:
            logger.info(f"进程 {process_id} 完成，返回 {len(k_values)} 个结果，总用时: {total_time:.2f}s")
            logger.info(f"====== 子进程 {process_id} 成功结束 ======")

        return result

    except Exception as e:
        error_msg = f"进程 {process_id} 发生严重错误: {e}"
        logger.error(error_msg)
        logger.error(f"完整错误信息: {traceback.format_exc()}")

        # 记录系统状态
        try:
            if torch.cuda.is_available():
                logger.error(f"GPU内存使用情况: {torch.cuda.memory_summary(device)}")
        except Exception:
            pass

        logger.error(f"====== 子进程 {process_id} 异常结束 ======")
        raise

    finally:
        # 清理日志处理器
        if file_handler:
            try:
                file_handler.close()
                logger.removeHandler(file_handler)
            except Exception:
                pass
