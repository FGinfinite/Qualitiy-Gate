# src/training/full_rank_finetuning.py
"""
全秩微调相关的工具函数模块

本模块提供了与全秩微调相关的功能，包括：
- 模块名称匹配（兼容PEFT库的逻辑）
- 全秩微调的设置和参数管理
- 权重的保存和加载
"""

import logging
import os
import re
from typing import Dict, List, Set

import torch


def get_target_modules_names(model: torch.nn.Module, target_patterns: List[str]) -> Set[str]:
    """
    获取与目标模式匹配的模块名称集合，使用PEFT库的匹配逻辑。

    Args:
        model: PyTorch模型
        target_patterns: 目标模块名称模式列表

    Returns:
        匹配的模块名称集合
    """
    target_modules = set()

    for name, _ in model.named_modules():
        for pattern in target_patterns:
            # 检查完全匹配
            if name == pattern:
                target_modules.add(name)
            # 检查结尾匹配（PEFT库的常见做法）
            elif name.endswith(pattern):
                target_modules.add(name)
            # 检查模式匹配（支持正则表达式）
            elif re.search(pattern, name):
                target_modules.add(name)

    return target_modules


def setup_full_rank_training(model: torch.nn.Module, target_patterns: List[str], mode: str = "module") -> Dict[str, torch.nn.Parameter]:
    """
    设置全秩微调，冻结除目标模块外的所有参数。

    Args:
        model: PyTorch模型
        target_patterns: 目标模块名称模式列表，或者参数名模式列表（routing模式）
        mode: 匹配模式，"module"（模块匹配）或"parameter"（参数名匹配）

    Returns:
        可训练参数的字典
    """
    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False

    trainable_params = {}

    if mode == "module":
        # 原有的模块匹配逻辑
        target_module_names = get_target_modules_names(model, target_patterns)

        # 解冻目标模块的参数
        for name, module in model.named_modules():
            if name in target_module_names:
                for param_name, param in module.named_parameters():
                    param.requires_grad = True
                    full_param_name = f"{name}.{param_name}" if name else param_name
                    trainable_params[full_param_name] = param

        log = logging.getLogger(__name__)
        log.info(f"设置全秩微调，目标模块: {target_module_names}")

    elif mode == "parameter":
        # 新的参数名匹配逻辑（用于routing_only模式）
        for name, param in model.named_parameters():
            # 检查参数名是否匹配任何路由相关的模式
            if any(pattern in name for pattern in target_patterns):
                param.requires_grad = True
                trainable_params[name] = param

        log = logging.getLogger(__name__)
        log.info(f"设置参数级微调，目标模式: {target_patterns}")

    else:
        raise ValueError(f"不支持的模式: {mode}。支持的模式: ['module', 'parameter']")

    trainable_count = sum(p.numel() for p in trainable_params.values())
    total_count = sum(p.numel() for p in model.parameters())
    log.info(f"可训练参数: {trainable_count:,} / 总参数: {total_count:,} ({100 * trainable_count / total_count:.2f}%)")

    return trainable_params


def save_full_rank_weights(model: torch.nn.Module, target_patterns: List[str], save_path: str, mode: str = "module", tokenizer=None) -> None:
    """
    保存全秩微调的模块权重和分词器。

    Args:
        model: PyTorch模型
        target_patterns: 目标模块名称模式列表，或者参数名模式列表
        save_path: 保存路径
        mode: 匹配模式，"module"（模块匹配）或"parameter"（参数名匹配）
        tokenizer: 可选的分词器，如果提供则一并保存
    """
    # 收集目标参数的状态字典
    full_rank_state_dict = {}

    if mode == "module":
        target_module_names = get_target_modules_names(model, target_patterns)

        for name, module in model.named_modules():
            if name in target_module_names:
                for param_name, param in module.named_parameters():
                    full_param_name = f"{name}.{param_name}" if name else param_name
                    full_rank_state_dict[full_param_name] = param.data.clone()

        metadata = {"target_module_names": list(target_module_names)}

    elif mode == "parameter":
        # 参数名模式匹配
        for name, param in model.named_parameters():
            if any(pattern in name for pattern in target_patterns):
                full_rank_state_dict[name] = param.data.clone()

        metadata = {"target_patterns": target_patterns}

    else:
        raise ValueError(f"不支持的模式: {mode}。支持的模式: ['module', 'parameter']")

    # 保存权重和元数据
    checkpoint = {
        "full_rank_weights": full_rank_state_dict,
        "mode": mode,
        "target_patterns": target_patterns,
        **metadata,
    }

    # 保存分词器（如果提供）
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)

    if tokenizer is not None:
        tokenizer.save_pretrained(save_dir)
        checkpoint["tokenizer_saved"] = True
        log = logging.getLogger(__name__)
        log.info(f"分词器已保存到 {save_dir}")
    else:
        checkpoint["tokenizer_saved"] = False

    torch.save(checkpoint, save_path)

    log = logging.getLogger(__name__)
    log.info(f"权重已保存到: {save_path} (模式: {mode}, 参数数量: {len(full_rank_state_dict)})")


def load_full_rank_weights(model: torch.nn.Module, checkpoint_path: str):
    """
    加载全秩微调的权重到预训练模型，并返回分词器（如果存在）。

    Args:
        model: 预训练的PyTorch模型
        checkpoint_path: 权重检查点路径

    Returns:
        tokenizer: 如果checkpoint中保存了分词器则返回，否则返回None
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点文件未找到: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    full_rank_weights = checkpoint["full_rank_weights"]
    mode = checkpoint.get("mode", "module")  # 向后兼容

    # 加载权重
    model_state_dict = model.state_dict()
    loaded_count = 0
    missing_params = []

    for param_name, weight in full_rank_weights.items():
        if param_name in model_state_dict:
            # 检查形状是否匹配
            if model_state_dict[param_name].shape == weight.shape:
                model_state_dict[param_name].copy_(weight)
                loaded_count += 1
            else:
                log = logging.getLogger(__name__)
                log.warning(f"参数 {param_name} 形状不匹配 - 模型: {model_state_dict[param_name].shape}, 检查点: {weight.shape}")
        else:
            missing_params.append(param_name)

    if missing_params:
        log = logging.getLogger(__name__)
        log.warning(f"以下 {len(missing_params)} 个参数在模型中未找到:")
        for param in missing_params[:5]:  # 只显示前5个
            log.info(f"  - {param}")
        if len(missing_params) > 5:
            log.info(f"  ... 还有 {len(missing_params) - 5} 个参数")

    log = logging.getLogger(__name__)
    log.info(f"成功加载 {loaded_count}/{len(full_rank_weights)} 个权重参数 (模式: {mode})")

    # 尝试加载分词器
    tokenizer = None
    if checkpoint.get("tokenizer_saved", False):
        checkpoint_dir = os.path.dirname(checkpoint_path)
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            log.info(f"成功从 {checkpoint_dir} 加载分词器")
        except Exception as e:
            log.warning(f"无法从 {checkpoint_dir} 加载分词器: {e}")

    return tokenizer


def print_trainable_parameters(model: torch.nn.Module) -> None:
    """
    打印模型的可训练参数统计信息（类似PEFT的print_trainable_parameters）。

    Args:
        model: PyTorch模型
    """
    log = logging.getLogger(__name__)

    trainable_params = 0
    all_param = 0

    for param in model.parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    log.info(f"trainable params: {trainable_params:,} || all params: {all_param:,} || trainable%: {100 * trainable_params / all_param:.4f}")


def load_trainer_checkpoint_full_rank(model: torch.nn.Module, checkpoint_dir: str, target_patterns: List[str] = None) -> None:
    """
    从Trainer保存的checkpoint加载全秩微调权重到预训练模型。

    Args:
        model: 预训练的PyTorch模型
        checkpoint_dir: Trainer checkpoint目录路径
        target_patterns: 可选，如果提供则只加载匹配的模块权重
    """
    from src.models.select_moe import SelectMoeForCausalLM

    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint目录未找到: {checkpoint_dir}")

    # 使用transformers加载完整的checkpoint
    log = logging.getLogger(__name__)
    log.info(f"正在从checkpoint加载模型: {checkpoint_dir}")
    checkpoint_model = SelectMoeForCausalLM.from_pretrained(checkpoint_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)

    if target_patterns is None:
        # 如果没有指定目标模块，加载所有权重
        log.info("加载所有权重...")
        model.load_state_dict(checkpoint_model.state_dict())
        log.info("成功加载所有权重")
    else:
        # 只加载指定的目标模块权重
        target_module_names = get_target_modules_names(model, target_patterns)
        log.info(f"只加载目标模块权重: {target_module_names}")

        model_state_dict = model.state_dict()
        checkpoint_state_dict = checkpoint_model.state_dict()
        loaded_count = 0

        for param_name, param_value in checkpoint_state_dict.items():
            # 检查是否属于目标模块
            module_name = ".".join(param_name.split(".")[:-1])  # 去掉参数名，得到模块名
            if module_name in target_module_names:
                if param_name in model_state_dict:
                    model_state_dict[param_name].copy_(param_value)
                    loaded_count += 1
                    log.info(f"已加载: {param_name}")
                else:
                    log.warning(f"参数 {param_name} 在目标模型中未找到")

        log.info(f"成功加载 {loaded_count} 个目标模块参数")

    # 清理checkpoint模型以释放内存
    del checkpoint_model
    torch.cuda.empty_cache()


def load_trainer_checkpoint_full_model(checkpoint_dir: str) -> torch.nn.Module:
    """
    直接从Trainer checkpoint加载完整的模型。

    Args:
        checkpoint_dir: Trainer checkpoint目录路径

    Returns:
        加载的完整模型
    """
    from src.models.select_moe import SelectMoeForCausalLM, register_select_moe

    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint目录未找到: {checkpoint_dir}")

    # 注册模型类型
    register_select_moe()

    # 加载完整模型
    log = logging.getLogger(__name__)
    log.info(f"正在从checkpoint加载完整模型: {checkpoint_dir}")
    model = SelectMoeForCausalLM.from_pretrained(checkpoint_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)

    log.info("成功加载完整模型")
    return model
