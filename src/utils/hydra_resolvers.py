"""
Custom Hydra resolvers for automatic configuration extraction.
"""

import os
import re

from omegaconf import OmegaConf


def extract_config_from_path(checkpoint_path: str) -> str:
    """
    Extract configuration information from model checkpoint path.

    Parses paths like:
    outputs/stage_1_pretrain/2025-08-10/03-42-54-batch=8_lr=0.001_loss=beta_moment_matching_tag=none/full_rank_weights.pt

    Returns a formatted string like:
    batch=8_lr=0.001_loss=beta_moment_matching_tag=none

    Args:
        checkpoint_path: Path to the model checkpoint

    Returns:
        Extracted configuration string, or "unknown" if parsing fails
    """
    try:
        # Extract the directory name that contains the configuration parameters
        # This is the parent directory of the checkpoint file
        checkpoint_dir = os.path.dirname(checkpoint_path)
        dir_name = os.path.basename(checkpoint_dir)

        # Pattern to match the timestamp-config format:
        # HH-MM-SS-batch=X_lr=Y_loss=Z_tag=W
        pattern = r"^\d{2}-\d{2}-\d{2}-(.*)"
        match = re.match(pattern, dir_name)

        if match:
            # Extract the config part after the timestamp
            config_part = match.group(1)
            return config_part
        else:
            # Fallback: try to extract any config-like patterns from the directory name
            # Look for patterns like key=value connected by underscores
            config_patterns = re.findall(r"[a-zA-Z_]+=[^_]+", dir_name)
            if config_patterns:
                return "_".join(config_patterns)
            else:
                return "unknown"

    except Exception as e:
        print(f"Warning: Failed to extract config from path '{checkpoint_path}': {e}")
        return "unknown"


def extract_data_timestamp(data_path: str) -> str:
    """
    Extract timestamp from dataset data path.

    Example: outputs/stage_2_selection/2025-08-11/03-52-40-batch=8_lr=0.001_loss=beta_moment_matching_tag=none/selected_data.jsonl
    Returns: 03-52-40
    """
    try:
        # Split by '/' and find the timestamp part (format: HH-MM-SS)
        parts = data_path.split("/")
        for part in parts:
            # Look for pattern HH-MM-SS at the beginning
            if re.match(r"^\d{2}-\d{2}-\d{2}", part):
                timestamp_match = re.match(r"^(\d{2}-\d{2}-\d{2})", part)
                if timestamp_match:
                    return timestamp_match.group(1)
        return "unknown"
    except Exception:
        return "unknown"


def extract_data_config(data_path: str) -> str:
    """
    Extract configuration parameters from dataset data path.

    Example: outputs/stage_2_selection/2025-08-11/03-52-40-batch=8_lr=0.001_loss=beta_moment_matching_tag=none/selected_data.jsonl
    Returns: 03-52-40-batch=8_lr=0.001_loss=beta_moment_matching_tag=none
    """
    try:
        # Split by '/' and find the directory with timestamp and config
        parts = data_path.split("/")
        for part in parts:
            # Look for pattern HH-MM-SS-config_params
            if re.match(r"^\d{2}-\d{2}-\d{2}-", part):
                return part
        return "unknown"
    except Exception:
        return "unknown"


def extract_model_config(batch_size: str, learning_rate: str, tag: str, model_name: str = "") -> str:
    """
    Create model configuration string from training parameters.

    Args:
        batch_size: Training batch size
        learning_rate: Learning rate
        tag: Training tag
        model_name: Model name (optional, for future use)

    Returns:
        Formatted string like: batch=128_lr=2e-05_tag=SE_qwen
    """
    try:
        # Convert learning rate to scientific notation if it's a float
        try:
            lr_float = float(learning_rate)
            if lr_float < 0.001:
                lr_str = f"{lr_float:.0e}"
            else:
                lr_str = str(learning_rate)
        except (ValueError, TypeError):
            lr_str = str(learning_rate)

        return f"batch={batch_size}_lr={lr_str}_tag={tag}"
    except Exception:
        return f"batch={batch_size}_lr={learning_rate}_tag={tag}"


def register_custom_resolvers():
    """
    Register all custom Hydra resolvers.
    This function should be called before @hydra.main decorators.
    """

    # Register the config extraction resolver
    OmegaConf.register_new_resolver("extract_config", extract_config_from_path, use_cache=True)

    # Register additional utility resolvers that might be useful

    # Extract just the batch size from checkpoint path
    def extract_batch_from_path(checkpoint_path: str) -> str:
        try:
            config_str = extract_config_from_path(checkpoint_path)
            batch_match = re.search(r"batch=([^_]+)", config_str)
            return batch_match.group(1) if batch_match else "unknown"
        except Exception:
            return "unknown"

    OmegaConf.register_new_resolver("extract_batch", extract_batch_from_path, use_cache=True)

    # Extract just the learning rate from checkpoint path
    def extract_lr_from_path(checkpoint_path: str) -> str:
        try:
            config_str = extract_config_from_path(checkpoint_path)
            lr_match = re.search(r"lr=([^_]+)", config_str)
            return lr_match.group(1) if lr_match else "unknown"
        except Exception:
            return "unknown"

    OmegaConf.register_new_resolver("extract_lr", extract_lr_from_path, use_cache=True)

    # Extract just the loss type from checkpoint path
    def extract_loss_from_path(checkpoint_path: str) -> str:
        try:
            config_str = extract_config_from_path(checkpoint_path)
            # Match loss= followed by anything until _tag= or end of string
            loss_match = re.search(r"loss=([^_]+(?:_[^=]+)*?)(?=_tag=|$)", config_str)
            return loss_match.group(1) if loss_match else "unknown"
        except Exception:
            return "unknown"

    OmegaConf.register_new_resolver("extract_loss", extract_loss_from_path, use_cache=True)

    # Extract just the tag from checkpoint path
    def extract_tag_from_path(checkpoint_path: str) -> str:
        try:
            config_str = extract_config_from_path(checkpoint_path)
            tag_match = re.search(r"tag=([^_]+)", config_str)
            return tag_match.group(1) if tag_match else "none"
        except Exception:
            return "none"

    OmegaConf.register_new_resolver("extract_tag", extract_tag_from_path, use_cache=True)

    # Register new data path resolvers
    OmegaConf.register_new_resolver("extract_data_timestamp", extract_data_timestamp, use_cache=True)
    OmegaConf.register_new_resolver("extract_data_config", extract_data_config, use_cache=True)
    OmegaConf.register_new_resolver("extract_model_config", extract_model_config, use_cache=True)


if __name__ == "__main__":
    # Test the resolver functions
    test_path = "outputs/stage_1_pretrain/2025-08-10/03-42-54-batch=8_lr=0.001_loss=beta_moment_matching_tag=none/full_rank_weights.pt"
    test_data_path = "outputs/stage_2_selection/2025-08-11/03-52-40-batch=8_lr=0.001_loss=beta_moment_matching_tag=none/selected_data.jsonl"

    print("Testing resolver functions:")
    print(f"Full config: {extract_config_from_path(test_path)}")
    print(f"Data timestamp: {extract_data_timestamp(test_data_path)}")
    print(f"Data config: {extract_data_config(test_data_path)}")
    print(f"Model config: {extract_model_config('128', '2e-05', 'SE_qwen')}")

    # Register resolvers for testing
    register_custom_resolvers()

    # Test individual extractors
    from omegaconf import OmegaConf

    config = OmegaConf.create(
        {
            "test_path": test_path,
            "test_data_path": test_data_path,
            "batch": "${extract_batch:${test_path}}",
            "lr": "${extract_lr:${test_path}}",
            "loss": "${extract_loss:${test_path}}",
            "tag": "${extract_tag:${test_path}}",
            "full_config": "${extract_config:${test_path}}",
            "data_timestamp": "${extract_data_timestamp:${test_data_path}}",
            "data_config": "${extract_data_config:${test_data_path}}",
            "model_config": "${extract_model_config:128,2e-05,SE_qwen}",
            "full_format": "MODEL=<|${extract_model_config:128,2e-05,SE_qwen}|>-DATA=<|${extract_data_config:${test_data_path}}|>",
        }
    )

    print("OmegaConf interpolation results:")
    print(OmegaConf.to_yaml(config))
