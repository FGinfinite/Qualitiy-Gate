# src/data/dataset_loader.py
import contextlib
import logging
import os
from functools import partial
from typing import Dict, List, Optional

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoTokenizer


@contextlib.contextmanager
def temp_seed(seed):
    """临时设置随机种子的上下文管理器"""
    state = np.random.get_state()
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def load_local_datasets(
    data_dir: str,
    dataset_names: Optional[List[str]] = None,
    sample_percentage: float = 1.0,
    seed: int = 0,
) -> Dataset:
    """
    从本地目录加载多个数据集并合并

    Args:
        data_dir: 数据集根目录 (例如: data/train/processed)
        dataset_names: 要加载的数据集名称列表，如果为None则加载所有可用数据集
        sample_percentage: 每个数据集的采样比例
        seed: 随机种子

    Returns:
        合并后的数据集
    """
    # 如果是相对路径，转换为绝对路径
    if not os.path.isabs(data_dir):
        # 从项目根目录开始的相对路径
        project_root = os.getcwd()
        data_dir = os.path.join(project_root, data_dir)

    if not os.path.exists(data_dir):
        raise ValueError(f"数据目录不存在: {data_dir}")

    # 如果没有指定数据集名称，自动发现所有可用数据集
    if dataset_names is None:
        dataset_names = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    if not dataset_names:
        raise ValueError(f"在目录 {data_dir} 中没有找到任何数据集")

    all_datasets = []

    for dataset_name in dataset_names:
        dataset_path = os.path.join(data_dir, dataset_name)
        if not os.path.exists(dataset_path):
            print(f"警告: 数据集目录不存在，跳过: {dataset_path}")
            continue

        # 查找数据文件
        data_files = []
        for file in os.listdir(dataset_path):
            if file.endswith((".jsonl", ".json")):
                data_files.append(os.path.join(dataset_path, file))

        if not data_files:
            print(f"警告: 在 {dataset_path} 中没有找到数据文件，跳过")
            continue

        # 加载数据集
        try:
            dataset = load_dataset("json", data_files=data_files)["train"]

            # 采样
            if sample_percentage < 1.0:
                sample_size = int(len(dataset) * sample_percentage)
                with temp_seed(seed):
                    indices = np.random.permutation(len(dataset))[:sample_size]
                dataset = dataset.select(indices)

            all_datasets.append(dataset)
            print(f"已加载数据集 '{dataset_name}': {len(dataset)} 个样本")

        except Exception as e:
            print(f"加载数据集 '{dataset_name}' 失败: {e}")
            continue

    if not all_datasets:
        raise ValueError("没有成功加载任何数据集")

    # 合并所有数据集
    combined_dataset = concatenate_datasets(all_datasets)
    print(f"已合并 {len(dataset_names)} 个数据集，总共 {len(combined_dataset)} 个样本")

    return combined_dataset


def convert_openhermes_format(example: Dict, dataset_name: str = "openhermes", example_index: int = None) -> Dict:
    """
    转换OpenHermes-2.5格式到项目标准格式

    关键处理：
    - 完全过滤空的system消息，避免生成无意义的特殊token
    - 角色映射：human→user, gpt→assistant, system→system
    - 字段映射：conversations→messages, from→role, value→content

    Args:
        example: OpenHermes-2.5格式的数据样本
        dataset_name: 数据集名称，用于生成dataset字段
        example_index: 样本在数据集中的索引，用于生成唯一ID

    Returns:
        转换后的标准格式数据
    """
    if "conversations" not in example:
        raise ValueError("OpenHermes数据必须包含'conversations'字段")

    messages = []
    for msg in example["conversations"]:
        # 角色映射
        role_map = {"human": "user", "gpt": "assistant", "system": "system"}

        if msg["from"] not in role_map:
            continue  # 跳过不支持的角色

        role = role_map[msg["from"]]
        content = msg["value"]

        # 关键：完全过滤空的system消息
        if role == "system" and not content.strip():
            continue  # 跳过空system消息，不添加到messages中

        messages.append({"role": role, "content": content.strip() if content else ""})

    if not messages:
        raise ValueError("转换后没有有效的消息")

    # 生成简单而可靠的ID
    original_id = example.get("id")
    if original_id is not None and str(original_id).strip() != "":
        # 如果有有效的原始ID，使用它
        example_id = f"{dataset_name}_{original_id}"
    else:
        # 使用数据集索引生成ID（推荐方案，简单且可靠）
        example_id = f"{dataset_name}_{example_index:06d}"
    return {"dataset": dataset_name, "id": example_id, "messages": messages}


def convert_gsm8k_format(example: Dict, dataset_name: str = "gsm8k", example_index: int = None) -> Dict:
    """
    转换GSM8K格式到项目标准格式

    GSM8K数据集包含数学问题和答案：
    - question: 数学问题文本
    - answer: 包含推理步骤和最终答案的文本

    Args:
        example: GSM8K格式的数据样本
        dataset_name: 数据集名称，用于生成dataset字段
        example_index: 样本在数据集中的索引，用于生成唯一ID

    Returns:
        转换后的标准格式数据
    """
    if "question" not in example or "answer" not in example:
        raise ValueError("GSM8K数据必须包含'question'和'answer'字段")

    # 构建标准messages格式
    messages = [
        {"role": "user", "content": example["question"].strip()},
        {"role": "assistant", "content": example["answer"].strip()},
    ]

    # 生成唯一ID
    example_id = f"{dataset_name}_{example_index:06d}"

    return {"dataset": dataset_name, "id": example_id, "messages": messages}


def convert_hendrycks_math_format(example: Dict, dataset_name: str = "hendrycks_math", example_index: int = None) -> Dict:
    """
    转换HENDRYCKS_MATH格式到项目标准格式

    HENDRYCKS_MATH数据集包含高等数学问题和详细解答：
    - problem: 数学问题文本
    - solution: 详细解答过程
    - level: 难度级别（如 "Level 5"）
    - type: 数学类型（如 "Algebra"）

    Args:
        example: HENDRYCKS_MATH格式的数据样本
        dataset_name: 数据集名称，用于生成dataset字段
        example_index: 样本在数据集中的索引，用于生成唯一ID

    Returns:
        转换后的标准格式数据
    """
    if "problem" not in example or "solution" not in example:
        raise ValueError("HENDRYCKS_MATH数据必须包含'problem'和'solution'字段")

    # 构建标准messages格式
    messages = [
        {"role": "user", "content": example["problem"].strip()},
        {"role": "assistant", "content": example["solution"].strip()},
    ]

    # 生成唯一ID
    example_id = f"{dataset_name}_{example_index:06d}"

    return {"dataset": dataset_name, "id": example_id, "messages": messages}


def load_hf_datasets(
    hf_config: Dict,
    sample_percentage: float = 1.0,
    seed: int = 0,
    use_shared_memory: bool = False,
    shared_memory_config: Optional[Dict] = None,
) -> Dataset:
    """
    从HuggingFace加载数据集并转换格式

    Args:
        hf_config: HF数据集配置，包含datasets列表
        sample_percentage: 采样比例
        seed: 随机种子
        use_shared_memory: 是否使用共享内存加速
        shared_memory_config: 共享内存配置

    Returns:
        转换后的合并数据集
    """
    if not hf_config or "datasets" not in hf_config:
        raise ValueError("HF配置必须包含'datasets'字段")

    log = logging.getLogger(__name__)

    # 尝试共享内存加载（可选）
    if use_shared_memory:
        try:
            from share_dataset import LoadResult, SharedDatasetClient

            client = SharedDatasetClient()
            for dataset_config in hf_config["datasets"]:
                dataset_name = dataset_config["name"]

                # 只支持OpenHermes数据集的共享内存加载
                if "openhermes" in dataset_name.lower():
                    log.info(f"尝试从共享内存加载数据集: {dataset_name}")

                    result = client.load_shared_dataset(dataset_config)
                    if result.status == LoadResult.SUCCESS:
                        log.info(f"共享内存加载成功! 耗时: {result.load_time:.4f}秒, {result.message}")

                        dataset = result.dataset

                        # 应用采样（如果需要）
                        if sample_percentage < 1.0:
                            original_size = len(dataset)
                            sample_size = int(original_size * sample_percentage)
                            with temp_seed(seed):
                                indices = np.random.permutation(original_size)[:sample_size]
                            dataset = dataset.select(indices)
                            log.info(f"采样后数据集大小: {len(dataset)}")

                        # 由于从共享内存加载的数据已经是转换后的格式，直接返回
                        log.info(f"从共享内存成功加载完整数据集，共 {len(dataset)} 个样本")
                        return dataset
                    else:
                        log.info(f"共享内存加载失败: {result.message}，回退到传统加载方式")

        except ImportError:
            log.debug("共享内存模块不可用，回退到传统加载方式")
        except Exception as e:
            log.warning(f"共享内存加载出错: {e}，回退到传统加载方式")

    all_datasets = []

    for dataset_config in hf_config["datasets"]:
        dataset_name = dataset_config["name"]
        internal_name = dataset_config.get("dataset_name", dataset_name.split("/")[-1])
        subset = dataset_config.get("subset", None)
        split = dataset_config.get("split", "train")

        # 检查是否需要加载所有子集
        if subset == "__full_subset__":
            try:
                from datasets import get_dataset_config_names

                # 定义已知数据集的子集列表（用于离线模式的 fallback）
                KNOWN_DATASET_SUBSETS = {
                    "EleutherAI/hendrycks_math": [
                        "algebra",
                        "counting_and_probability",
                        "geometry",
                        "intermediate_algebra",
                        "number_theory",
                        "prealgebra",
                        "precalculus",
                    ],
                }

                # 尝试从 HuggingFace Hub 获取子集列表
                try:
                    all_subsets = get_dataset_config_names(dataset_name)

                    # 检查是否返回了 'default'（这通常意味着离线模式或无法获取配置）
                    if all_subsets == ["default"] and dataset_name in KNOWN_DATASET_SUBSETS:
                        log.info("检测到离线模式或无法获取子集列表，使用预定义的子集列表")
                        all_subsets = KNOWN_DATASET_SUBSETS[dataset_name]

                except Exception as e:
                    # 如果获取失败，使用预定义的子集列表
                    if dataset_name in KNOWN_DATASET_SUBSETS:
                        log.info(f"无法从 Hub 获取子集列表 ({e})，使用预定义的子集列表")
                        all_subsets = KNOWN_DATASET_SUBSETS[dataset_name]
                    else:
                        raise ValueError(f"无法获取数据集 {dataset_name} 的子集列表，且没有预定义的 fallback 列表") from e

                print(f"检测到 __full_subset__ 标记，将加载 {dataset_name} 的所有 {len(all_subsets)} 个子集")
                print(f"子集列表: {all_subsets}")

                # 遍历每个子集
                for subset_name in all_subsets:
                    try:
                        print(f"  正在加载子集 '{subset_name}'...")
                        dataset = load_dataset(dataset_name, subset_name, split=split)

                        # 采样
                        if sample_percentage < 1.0:
                            sample_size = int(len(dataset) * sample_percentage)
                            with temp_seed(seed):
                                indices = np.random.permutation(len(dataset))[:sample_size]
                            dataset = dataset.select(indices)

                        # 转换格式（使用子集名称作为 dataset_name）
                        if "hendrycks_math" in dataset_name.lower() or "math" in dataset_name.lower():
                            columns_to_remove = [col for col in dataset.column_names if col not in ["dataset", "id"]]
                            dataset = dataset.map(
                                lambda example, idx, name=subset_name: convert_hendrycks_math_format(example, dataset_name=name, example_index=idx),
                                with_indices=True,
                                desc=f"转换数据集格式 - {subset_name}",
                                load_from_cache_file=True,
                                remove_columns=columns_to_remove,
                            )
                        else:
                            raise ValueError(f"__full_subset__ 仅支持 HENDRYCKS_MATH 等多子集数据集，不支持: {dataset_name}")

                        all_datasets.append(dataset)
                        print(f"  ✓ 已加载子集 '{subset_name}': {len(dataset)} 个样本")

                    except Exception as e:
                        print(f"  ✗ 加载子集 '{subset_name}' 失败: {e}")
                        continue

            except Exception as e:
                print(f"加载 __full_subset__ 失败: {e}")
                continue

        else:
            # 原有的单子集加载逻辑
            try:
                # 加载HF数据集
                print(f"正在从HuggingFace加载数据集 '{dataset_name}'...")
                if subset:
                    dataset = load_dataset(dataset_name, subset, split=split)
                else:
                    dataset = load_dataset(dataset_name, split=split)

                # 采样
                if sample_percentage < 1.0:
                    sample_size = int(len(dataset) * sample_percentage)
                    with temp_seed(seed):
                        indices = np.random.permutation(len(dataset))[:sample_size]
                    dataset = dataset.select(indices)

                # 转换格式
                if "openhermes" in dataset_name.lower():
                    # 获取需要移除的列（保留 dataset 和 id）
                    columns_to_remove = [col for col in dataset.column_names if col not in ["dataset", "id"]]

                    # OpenHermes-2.5 格式转换，使用lambda避免闭包变量问题
                    dataset = dataset.map(
                        lambda example, idx, name=internal_name: convert_openhermes_format(example, dataset_name=name, example_index=idx),
                        with_indices=True,  # 传递索引
                        desc=f"转换数据集格式 - {internal_name}",
                        load_from_cache_file=True,  # 使用缓存提升性能
                        remove_columns=columns_to_remove,  # 移除所有原始字段
                    )
                elif "gsm8k" in dataset_name.lower():
                    # 获取需要移除的列（保留 dataset 和 id）
                    columns_to_remove = [col for col in dataset.column_names if col not in ["dataset", "id"]]

                    # GSM8K 格式转换
                    dataset = dataset.map(
                        lambda example, idx, name=internal_name: convert_gsm8k_format(example, dataset_name=name, example_index=idx),
                        with_indices=True,  # 传递索引
                        desc=f"转换数据集格式 - {internal_name}",
                        load_from_cache_file=True,  # 使用缓存提升性能
                        remove_columns=columns_to_remove,  # 移除所有原始字段
                    )
                elif "hendrycks_math" in dataset_name.lower() or "math" in dataset_name.lower():
                    # 获取需要移除的列（保留 dataset 和 id）
                    columns_to_remove = [col for col in dataset.column_names if col not in ["dataset", "id"]]

                    # HENDRYCKS_MATH 格式转换
                    dataset = dataset.map(
                        lambda example, idx, name=internal_name: convert_hendrycks_math_format(example, dataset_name=name, example_index=idx),
                        with_indices=True,  # 传递索引
                        desc=f"转换数据集格式 - {internal_name}",
                        load_from_cache_file=True,  # 使用缓存提升性能
                        remove_columns=columns_to_remove,  # 移除所有原始字段
                    )
                else:
                    # 其他HF数据集可以在这里添加转换逻辑
                    raise ValueError(f"暂不支持数据集格式: {dataset_name}")

                all_datasets.append(dataset)
                print(f"已加载HF数据集 '{internal_name}': {len(dataset)} 个样本")

            except Exception as e:
                print(f"加载HF数据集 '{dataset_name}' 失败: {e}")
                continue

    if not all_datasets:
        raise ValueError("没有成功加载任何HF数据集")

    # 合并所有数据集
    combined_dataset = concatenate_datasets(all_datasets)
    print(f"已合并 {len(hf_config['datasets'])} 个HF数据集，总共 {len(combined_dataset)} 个样本")

    return combined_dataset


def encode_with_messages_format(example: Dict, tokenizer: AutoTokenizer, max_seq_length: int) -> Dict:
    """
    使用messages格式编码数据，基于LESS库的实现

    Args:
        example: 包含messages字段的数据样本
        tokenizer: 分词器
        max_seq_length: 最大序列长度

    Returns:
        编码后的数据
    """
    messages = example["messages"]
    if len(messages) == 0:
        raise ValueError("messages字段为空")

    example_text = concat_messages(messages, tokenizer)
    tokenized_example = tokenizer(example_text, return_tensors="pt", max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    # 掩码非assistant部分，避免计算损失
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    concat_messages(messages[:message_idx], tokenizer),
                    return_tensors="pt",
                    max_length=max_seq_length,
                    truncation=True,
                ).input_ids.shape[1]

            if message_idx < len(messages) - 1 and messages[message_idx + 1]["role"] == "assistant":
                # 这里也忽略assistant的role部分
                messages_so_far = concat_messages(messages[: message_idx + 1], tokenizer) + "<|assistant|>\n"
            else:
                messages_so_far = concat_messages(messages[: message_idx + 1], tokenizer)

            message_end_idx = tokenizer(
                messages_so_far,
                return_tensors="pt",
                max_length=max_seq_length,
                truncation=True,
            ).input_ids.shape[1]
            labels[:, message_start_idx:message_end_idx] = -100

            if message_end_idx >= max_seq_length:
                break

    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
    }


def concat_messages(messages, tokenizer):
    """拼接消息，与LESS库保持一致"""
    message_text = ""
    for message in messages:
        if message["role"] == "system":
            message_text += "<|system|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "user":
            message_text += "<|user|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "assistant":
            message_text += "<|assistant|>\n" + message["content"].strip() + tokenizer.eos_token + "\n"
        else:
            raise ValueError(f"不支持的role: {message['role']}")
    return message_text


def encode_with_flexible_format(
    example: Dict, tokenizer: AutoTokenizer, max_seq_length: int, full_sequence_prediction: bool = False, mask_special_tokens: bool = False
) -> Dict:
    """
    灵活的编码函数，支持多种掩码模式

    Args:
        example: 包含messages字段的数据样本
        tokenizer: 分词器
        max_seq_length: 最大序列长度
        full_sequence_prediction: 是否进行全序列预测
        mask_special_tokens: 是否掩码特殊format token

    Returns:
        编码后的数据

    掩码模式说明：
    1. full_sequence_prediction=True, mask_special_tokens=True (推荐)：
       掩码特殊token，学习预测问题内容和答案内容
    2. full_sequence_prediction=True, mask_special_tokens=False：
       学习预测整个序列，包括特殊token
    3. full_sequence_prediction=False, mask_special_tokens=True：
       只学习预测assistant回复，掩码特殊token
    4. full_sequence_prediction=False, mask_special_tokens=False：
       传统SFT模式，学习预测<|assistant|>+答案内容
    """
    messages = example["messages"]
    if len(messages) == 0:
        raise ValueError("messages字段为空")

    example_text = concat_messages(messages, tokenizer)
    tokenized_example = tokenizer(example_text, return_tensors="pt", max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()
    attention_mask = torch.ones_like(input_ids)

    # 特殊token列表
    special_tokens = ["<|user|>", "<|assistant|>", "<|system|>"]

    if full_sequence_prediction:
        # 全序列预测模式
        if mask_special_tokens:
            # 模式1: 掩码特殊token，学习预测内容
            # 需要找到所有特殊token的位置并掩码
            text_tokens = tokenizer.tokenize(example_text)
            current_pos = 0

            for _ in text_tokens:
                if current_pos >= labels.shape[1]:
                    break

                # 检查是否是特殊token的开始
                for special_token in special_tokens:
                    special_token_tokens = tokenizer.tokenize(special_token)
                    if (
                        current_pos + len(special_token_tokens) <= labels.shape[1]
                        and text_tokens[current_pos : current_pos + len(special_token_tokens)] == special_token_tokens
                    ):
                        # 掩码这个特殊token的所有子token
                        labels[:, current_pos : current_pos + len(special_token_tokens)] = -100
                        current_pos += len(special_token_tokens)
                        break
                else:
                    current_pos += 1
        # else: 模式2，不需要额外掩码，保持所有token的labels
    else:
        # 传统SFT模式，只学习assistant部分
        for message_idx, message in enumerate(messages):
            if message["role"] != "assistant":
                if message_idx == 0:
                    message_start_idx = 0
                else:
                    message_start_idx = tokenizer(
                        concat_messages(messages[:message_idx], tokenizer),
                        return_tensors="pt",
                        max_length=max_seq_length,
                        truncation=True,
                    ).input_ids.shape[1]

                if message_idx < len(messages) - 1 and messages[message_idx + 1]["role"] == "assistant":
                    if mask_special_tokens:
                        # 模式3: 不包含<|assistant|>
                        messages_so_far = concat_messages(messages[: message_idx + 1], tokenizer)
                    else:
                        # 模式4: 包含<|assistant|>
                        messages_so_far = concat_messages(messages[: message_idx + 1], tokenizer) + "<|assistant|>\n"
                else:
                    messages_so_far = concat_messages(messages[: message_idx + 1], tokenizer)

                message_end_idx = tokenizer(
                    messages_so_far,
                    return_tensors="pt",
                    max_length=max_seq_length,
                    truncation=True,
                ).input_ids.shape[1]

                labels[:, message_start_idx:message_end_idx] = -100

                if message_end_idx >= max_seq_length:
                    break

    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
    }


def encode_data(
    raw_datasets: Dataset,
    tokenizer: AutoTokenizer,
    max_seq_length: int,
    processing_num_workers: int = 10,
    overwrite_cache: bool = False,
    full_sequence_prediction: bool = False,
    mask_special_tokens: bool = False,
) -> Dataset:
    """
    编码数据集

    Args:
        raw_datasets: 原始数据集
        tokenizer: 分词器
        max_seq_length: 最大序列长度
        processing_num_workers: 处理工作进程数
        overwrite_cache: 是否覆盖缓存
        full_sequence_prediction: 是否进行全序列预测
        mask_special_tokens: 是否掩码特殊格式token

    Returns:
        编码后的数据集
    """
    # 如果已经编码过，直接返回
    if "input_ids" in raw_datasets.features:
        return raw_datasets

    # 检查数据格式并选择编码函数
    if "messages" not in raw_datasets.column_names:
        raise ValueError("数据集必须包含'messages'字段")

    # 根据配置选择编码函数
    if full_sequence_prediction or mask_special_tokens:
        # 使用新的灵活编码函数
        encode_function = partial(
            encode_with_flexible_format,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            full_sequence_prediction=full_sequence_prediction,
            mask_special_tokens=mask_special_tokens,
        )
        mode_desc = f"灵活编码(全序列={full_sequence_prediction}, 掩码特殊token={mask_special_tokens})"
    else:
        # 使用传统编码函数（向后兼容）
        encode_function = partial(
            encode_with_messages_format,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        )
        mode_desc = "传统SFT编码"

    # 使用多进程处理，移除原始列
    lm_datasets = raw_datasets.map(
        encode_function,
        batched=False,
        num_proc=processing_num_workers,
        load_from_cache_file=not overwrite_cache,
        desc=f"对数据进行分词和格式化 - {mode_desc}",
        remove_columns=raw_datasets.column_names,  # 移除所有原始列
    )

    # 显式设置为torch格式
    lm_datasets.set_format(type="torch")

    return lm_datasets


def load_and_prepare_dataset(cfg: DictConfig) -> Dataset:
    """
    加载和准备数据集的主要函数，支持本地和HF数据源

    Args:
        cfg: 配置对象，使用dataset_from参数控制数据源选择
             dataset_from="local": 使用本地数据集
             dataset_from="hf": 使用HuggingFace数据集

    Returns:
        准备好的数据集
    """
    # 从配置中获取参数
    sample_percentage = getattr(cfg.dataset, "subset_ratio", 1.0)
    seed = getattr(cfg, "seed", 42)  # 使用全局种子
    shuffle = getattr(cfg.dataset, "shuffle", True)
    dataset_from = getattr(cfg.dataset, "dataset_from", "local")

    # 序列长度排序相关配置
    sort_by_length = getattr(cfg.dataset, "sort_by_length", False)

    # 共享内存相关配置
    use_shared_memory = getattr(cfg.dataset, "use_shared_memory", False)
    shared_memory_config = getattr(cfg.dataset, "shared_memory", None)

    log = logging.getLogger(__name__)

    # 根据dataset_from参数选择数据源
    if dataset_from == "local":
        # 使用本地数据集
        local_config = cfg.dataset.local
        log.info(f"使用本地数据集，从 {local_config.data_dir} 加载数据集...")
        dataset = load_local_datasets(
            data_dir=local_config.data_dir,
            dataset_names=local_config.dataset_names,
            sample_percentage=sample_percentage,
            seed=seed,
        )

    elif dataset_from == "hf":
        # 使用HuggingFace数据集
        hf_config = {"datasets": cfg.dataset.hf.datasets}
        log.info("使用HuggingFace数据集...")
        dataset = load_hf_datasets(
            hf_config=hf_config,
            sample_percentage=sample_percentage,
            seed=seed,
            use_shared_memory=use_shared_memory,
            shared_memory_config=shared_memory_config,
        )

    else:
        raise ValueError(f"不支持的数据源类型: {dataset_from}，支持的类型: 'local', 'hf'")

    # 序列长度排序处理
    if sort_by_length:
        log.info("启用字符串长度排序功能...")

        # 如果同时启用排序和shuffle，优先使用排序并给出提示
        if shuffle:
            log.info("检测到同时启用排序和shuffle，优先使用长度排序，跳过shuffle")
            shuffle = False

        try:
            # 使用字符串长度快速排序
            dataset = sort_dataset_by_string_length(dataset, descending=True)
            log.info("✓ 数据集已按字符串长度降序排列")

        except Exception as e:
            log.error(f"字符串长度排序失败: {e}")
            log.info("回退到原始数据集顺序")

    # 打乱数据集（如果排序未启用）
    if shuffle:
        dataset = dataset.shuffle(seed=seed)
        log.info(f"已打乱数据集，使用种子: {seed}")

    log.info(f"数据集准备完成，总共 {len(dataset)} 个样本")
    return dataset


def load_selected_data(data_path: str) -> Dataset:
    """
    加载选择后的数据文件（JSONL格式）

    Args:
        data_path: 选择后的数据文件路径，格式为包含dataset、id、scores、messages字段的JSONL

    Returns:
        加载的数据集
    """
    # 如果是相对路径，转换为绝对路径
    if not os.path.isabs(data_path):
        project_root = os.getcwd()
        data_path = os.path.join(project_root, data_path)

    if not os.path.exists(data_path):
        raise ValueError(f"数据文件不存在: {data_path}")

    # 加载JSONL文件
    try:
        dataset = load_dataset("json", data_files=data_path)["train"]
        print(f"已加载选择后的数据: {len(dataset)} 个样本")

        # 验证数据格式
        if "messages" not in dataset.column_names:
            raise ValueError("数据集必须包含'messages'字段")

        # 打印一些统计信息
        if len(dataset) > 0:
            sample = dataset[0]
            print("数据格式验证:")
            print(f"  - 包含字段: {list(sample.keys())}")
            if "scores" in sample:
                scores = [item["scores"] for item in dataset]
                print(f"  - 分数范围: {min(scores):.4f} - {max(scores):.4f}")
            if "dataset" in sample:
                dataset_counts = {}
                for item in dataset:
                    ds_name = item["dataset"]
                    dataset_counts[ds_name] = dataset_counts.get(ds_name, 0) + 1
                print(f"  - 数据集分布: {dataset_counts}")

        return dataset

    except Exception as e:
        raise ValueError(f"加载选择后的数据失败: {e}") from e


def compute_string_lengths(dataset: Dataset) -> List[int]:
    """
    快速计算数据集中每个样本的字符串总长度（使用numpy向量化优化）

    Args:
        dataset: 包含messages字段的数据集

    Returns:
        每个样本的字符串总长度列表
    """
    log = logging.getLogger(__name__)
    log.info(f"正在计算 {len(dataset)} 个样本的字符串长度...")

    all_texts = []
    batch_size = 3000

    # 批量访问数据集，避免逐个访问
    for start_idx in tqdm(range(0, len(dataset), batch_size), desc="提取文本"):
        end_idx = min(start_idx + batch_size, len(dataset))
        batch = dataset[start_idx:end_idx]

        # 处理批次中的每个样本
        if isinstance(batch["messages"], list) and len(batch["messages"]) > 0:
            if isinstance(batch["messages"][0], list):
                # 批处理模式
                for messages in batch["messages"]:
                    text_parts = []
                    if isinstance(messages, list):
                        for msg in messages:
                            if isinstance(msg, dict):
                                text_parts.extend([str(msg.get("role", "")), str(msg.get("content", ""))])
                    all_texts.append("".join(text_parts))
            else:
                # 单个样本模式
                messages = batch["messages"]
                text_parts = []
                if isinstance(messages, list):
                    for msg in messages:
                        if isinstance(msg, dict):
                            text_parts.extend([str(msg.get("role", "")), str(msg.get("content", ""))])
                all_texts.append("".join(text_parts))
        else:
            all_texts.extend([""] * (end_idx - start_idx))

    # 使用numpy向量化计算长度
    text_array = np.array(all_texts, dtype=object)
    lengths = np.array([len(text) for text in text_array])

    log.info(f"字符串长度计算完成，平均长度: {lengths.mean():.2f}")
    log.info(f"长度范围: {lengths.min()} - {lengths.max()}")

    return lengths.tolist()


def sort_dataset_by_string_length(dataset: Dataset, descending: bool = True) -> Dataset:
    """
    根据字符串总长度对数据集进行快速排序

    Args:
        dataset: 要排序的数据集
        descending: 是否降序排列（True为降序，False为升序）

    Returns:
        按字符串长度排序的数据集
    """
    log = logging.getLogger(__name__)

    # 计算字符串长度
    lengths = compute_string_lengths(dataset)

    # 创建索引和长度的配对，然后排序
    indexed_lengths = list(enumerate(lengths))
    indexed_lengths.sort(key=lambda x: x[1], reverse=descending)

    # 提取排序后的索引
    sorted_indices = [idx for idx, _ in indexed_lengths]

    log.info(f"按字符串长度{'降序' if descending else '升序'}排列数据集")
    log.info(f"排序前长度范围: {min(lengths)} - {max(lengths)}")

    # 根据排序后的索引重新排列数据集
    sorted_dataset = dataset.select(sorted_indices)

    # 验证排序结果
    if len(sorted_indices) > 10:
        first_lengths = [lengths[i] for i in sorted_indices[:5]]
        last_lengths = [lengths[i] for i in sorted_indices[-5:]]
        log.info(f"排序后前5个长度: {first_lengths}")
        log.info(f"排序后后5个长度: {last_lengths}")

    return sorted_dataset


def get_data_statistics(lm_datasets: Dataset) -> None:
    """
    获取数据集统计信息

    Args:
        lm_datasets: 编码后的数据集
    """

    def get_length(examples):
        lengths = [len(ids) for ids in examples["input_ids"]]
        completion_lens = []
        for labels in examples["labels"]:
            # 修复tensor创建警告
            if isinstance(labels, torch.Tensor):
                com_len = (labels > -1).sum().item()
            else:
                com_len = (torch.tensor(labels) > -1).sum().item()
            completion_lens.append(com_len)
        return {"length": lengths, "c_length": completion_lens}

    data_size = len(lm_datasets)
    dataset_with_lengths = lm_datasets.map(get_length, batched=True)
    lengths = dataset_with_lengths["length"]
    avg_length = sum(lengths) / len(lengths)
    c_lengths = dataset_with_lengths["c_length"]
    avg_c_length = sum(c_lengths) / len(c_lengths)

    log = logging.getLogger(__name__)

    log.info("数据集统计信息:")
    log.info(f"  样本数: {data_size}")
    log.info(f"  平均token数: {avg_length:.2f}")
    log.info(f"  平均完成token数: {avg_c_length:.2f}")
