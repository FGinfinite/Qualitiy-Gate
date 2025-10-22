# Copyright 2024 Quality-Gate Project. All rights reserved.
# Licensed under the Apache License, Version 2.0

"""Custom models for Quality-Gate project with transformers registration support."""

from .quality_gate_model import (
    QualityGateConfig,
    QualityGateForCausalLM,
    QualityGateModel,
    register_quality_gate,
)


def register_custom_models():
    """Register all custom models for AutoConfig and AutoModel."""
    register_quality_gate()
    print("Quality-Gate model registered successfully!")
    print("Available model types:")
    print("  - quality_gate: QualityGateConfig, QualityGateModel, QualityGateForCausalLM")
    print("Features:")
    print("  - Quality gate architecture for data quality assessment")
    print("  - Gate inserted before FFN layer in each transformer block")
    print("  - Quality classification loss for data selection")


__all__ = [
    "QualityGateConfig",
    "QualityGateModel",
    "QualityGateForCausalLM",
    "register_custom_models",
    "register_quality_gate",
]
