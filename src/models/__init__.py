# Copyright 2024 Select-MoE Project. All rights reserved.
# Licensed under the Apache License, Version 2.0

"""Custom models for Select-MoE project with transformers registration support."""

from .select_moe import (
    SelectMoeConfig,
    SelectMoeForCausalLM,
    SelectMoeModel,
    register_select_moe,
)


def register_custom_models():
    """Register all custom models for AutoConfig and AutoModel."""
    register_select_moe()
    print("Select-MoE model registered successfully!")
    print("Available model types:")
    print("  - select_moe: SelectMoeConfig, SelectMoeModel, SelectMoeForCausalLM")
    print("Features:")
    print("  - Two-tier routing architecture with quality gate")
    print("  - MoE experts + trash expert for low-quality data")
    print("  - Quality classification loss for data selection")


__all__ = [
    "SelectMoeConfig",
    "SelectMoeModel",
    "SelectMoeForCausalLM",
    "register_custom_models",
    "register_select_moe",
]
