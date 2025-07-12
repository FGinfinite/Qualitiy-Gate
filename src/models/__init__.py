# Copyright 2024 Select-MoE Project. All rights reserved.
# Licensed under the Apache License, Version 2.0

"""Custom models for Select-MoE project with transformers registration support."""

from .select_moe import (
    SelectMoeConfig,
    SelectMoeModel,
    SelectMoeForCausalLM,
    TrashCanSparseMoeBlock,
    replace_moe_layers_with_trashcan,
    register_select_moe,
)


def register_custom_models():
    """Register all custom models for AutoConfig and AutoModel."""
    register_select_moe()
    print("Select-MoE model registered successfully!")
    print("Available model types:")
    print("  - select_moe: SelectMoeConfig, SelectMoeModel, SelectMoeForCausalLM")
    print("Features:")
    print("  - Inherits pretrained OLMoE parameters")
    print("  - Dynamically adds trash can experts (count = top_k)")
    print("  - Custom constraint loss for data quality selection")


__all__ = [
    "SelectMoeConfig",
    "SelectMoeModel",
    "SelectMoeForCausalLM",
    "TrashCanSparseMoeBlock",
    "replace_moe_layers_with_trashcan",
    "register_custom_models",
    "register_select_moe",
]