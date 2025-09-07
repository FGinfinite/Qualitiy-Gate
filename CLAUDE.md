# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Select-MoE is a data selection framework using Mixture-of-Experts (MoE) models. The project implements a four-stage pipeline:

1. **Stage 1 (Warmup)**: Train a Select-MoE router to learn data quality discrimination
2. **Stage 2 (Selection)**: Use the trained router to score and filter training data
3. **Stage 3 (Finetune)**: Fine-tune target models (Llama-2-7B, Qwen2.5-1.5B) with selected data using LoRA
4. **Stage 4 (Evaluate)**: Evaluate model performance using lm-eval

## Key Architecture Components

### Select-MoE Model (`src/models/select_moe.py`)
- **Two-tier Routing Architecture**: Implements quality gate + MoE + trash expert parallel processing
- **Quality Gate**: Single-score quality assessment
  - Outputs single raw score instead of 2-class logits
  - `good_ratio = sigmoid(quality_score)`, `bad_ratio = 1 - good_ratio`
  - Simpler architecture with better gradient flow
- **MoE Integration**: Uses standard OlmoeSparseMoeBlock for expert routing
- **Trash Expert**: Configurable output modes (zero, noise, custom) for low-quality data
- **Quality Classification Loss**: 
  - Extensible loss function framework supporting multiple loss types
  - Proper padding token handling with attention_mask
  - Custom loss function support for experimentation
  - Direct sigmoid on raw score for better optimization
- **Router Output Format**: Returns dictionary with `quality_score` and `moe_logits` for data selection

### Core Pipeline (`src/main.py`)
- Hydra-based configuration management with stage-specific configs
- Four distinct execution stages controlled by config selection
- Integrated with accelerate for distributed training

### Data Processing (`src/data/`)
- Dataset loaders for training and evaluation data
- Support for multiple datasets: CoT, Dolly, FLAN-v2, OASST1
- **Flexible Data Encoding**: Support for multiple encoding modes
  - **Full Sequence Prediction**: Learn to predict both user questions and assistant responses
  - **Assistant-Only SFT**: Traditional supervised fine-tuning on assistant responses only
  - **Special Token Masking**: Option to mask format tokens like `<|user|>`, `<|assistant|>`
  - **Sample-wise Averaging**: Prevent bias toward longer sequences in loss computation

### Data Selection Module (`src/selection/`)
- **Decoupled Selection Logic**: Clean separation between router computation and data selection algorithms
- **Consolidated Functions**: Single source of truth for all data selection utilities
- **Clustering Integration**: Seamless integration with GPU-accelerated clustering algorithms
- **Router Data Processing**: Utilities for loading and processing router inference outputs
- **Multi-Script Support**: Shared functions used by both standalone and batch selection scripts

### Clustering-Based Selection (`src/clustering/`)
- **GPU K-Means with Elbow Method**: Automatic k-value selection using GPU acceleration and multi-GPU parallel processing
- **GPU Silhouette Score**: High-performance GPU-accelerated silhouette coefficient calculation for cosine distance
- **Round-Robin Selection**: Selects high-quality data from each cluster using quality scores
- **Cosine Distance Metric**: Uses cosine similarity for better semantic clustering of MoE logits
- **ClusterBasedSelection**: Main selection class with extensible interface for future clustering algorithms
- **Multi-GPU Parallel Processing**: Supports parallel k-value search across multiple GPUs for faster clustering

## Code Quality Instructions

**IMPORTANT**: When checking syntax or formatting code, Claude must ALWAYS use:
```bash
# Check code syntax and style issues
ruff check

# Format code automatically
ruff format
```
Never rely on manual inspection for code formatting - always use these tools.

## Development Commands

### Environment Setup
```bash
# Install dependencies using uv
uv sync

# Activate virtual environment
source .venv/bin/activate

# Install tools (optional)
./tools/install.sh
```

### Code Quality Checks
```bash
# Check code syntax and style issues
ruff check

# Auto-format all code
ruff format

# Check specific file
ruff check src/models/select_moe.py

# Format specific file
ruff format src/models/select_moe.py
```

### Model Conversion
```bash
# Convert OLMoE to Select-MoE format
python scripts/convert_olmoe_to_select_moe.py --save-path ./converted_models/select_moe_converted_OLMoE-1B-7B-0125

# Verify conversion
python scripts/compare_converted_model.py --converted-model ./converted_models/select_moe_converted_OLMoE-1B-7B-0125 --dtype bfloat16
```

### Training Pipeline
```bash
# Stage 1: Router warmup training (set CUDA_VISIBLE_DEVICES first)
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_stage_1.sh

# Stage 2: Router data computation (inference only)
CUDA_VISIBLE_DEVICES=0 bash scripts/run_stage_2.sh model_checkpoint_path=outputs/stage_1_pretrain/YYYY-MM-DD/HH-MM-SS/full_rank_weights.pt

# Stage 2b: Data selection using standalone scripts (after router data computation)
CUDA_VISIBLE_DEVICES=0 uv run scripts/continue_selection.py router_data_dir=outputs/stage_2_selection/YYYY-MM-DD/HH-MM-SS/router_data

# Stage 2c: Batch data selection across multiple experiments
CUDA_VISIBLE_DEVICES=0 uv run scripts/batch_selection.py root_dir=outputs/stage_2_selection

# Stage 3: Target model fine-tuning
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_stage_3.sh dataset.data_path=outputs/stage_2_selection/YYYY-MM-DD/HH-MM-SS/selected_data.jsonl

# Stage 4: Model evaluation
bash scripts/eval.sh
```

### Configuration Override
All stages support Hydra parameter overrides:
```bash
# Override learning rate and batch size
bash scripts/run_stage_1.sh training.learning_rate=5e-5 training.batch_size=8

# Override quality loss configuration
bash scripts/run_stage_1.sh training.quality_loss_type=beta_moment_matching training.quality_loss_debug=true

# Override custom loss parameters
bash scripts/run_stage_1.sh training.quality_loss_type=mean_variance_regularization training.quality_loss_params.lambda_var=0.2

# Override selection percentage
bash scripts/run_stage_2.sh selection_percentage=0.1

# Override K-Means parameters
bash scripts/run_stage_2.sh clustering_method=kmeans clustering_params.k=50 clustering_params.auto_k=false

# Enable K-Means multi-GPU parallel computation
bash scripts/run_stage_2.sh clustering_method=kmeans clustering_params.enable_parallel_kmeans=true clustering_params.parallel_processes=8

# Override K-Means parallel parameters with custom GPU allocation
bash scripts/run_stage_2.sh clustering_method=kmeans clustering_params.enable_parallel_kmeans=true clustering_params.parallel_processes=12 clustering_params.gpu_allocation_strategy=balanced

# Override LoRA parameters
bash scripts/run_stage_3.sh training.lora.r=64 training.lora.lora_alpha=128

# Enable clustering debug output
bash scripts/run_stage_2.sh debug_print=true

# Stage 2b: Standalone data selection with debug output
CUDA_VISIBLE_DEVICES=1 uv run scripts/continue_selection.py router_data_dir=outputs/stage_2_selection/YYYY-MM-DD/HH-MM-SS/router_data debug_print=true

# Stage 2c: Multi-GPU parallel K-means selection
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run scripts/continue_selection.py router_data_dir=outputs/stage_2_selection/YYYY-MM-DD/HH-MM-SS/router_data clustering_method=kmeans clustering_params.enable_parallel_kmeans=true clustering_params.parallel_processes=8

# Batch processing multiple experiments
CUDA_VISIBLE_DEVICES=0 uv run scripts/batch_selection.py root_dir=outputs/stage_2_selection selection_percentage=0.1 clustering_method=kmeans
```

### Validation and Testing
```bash
# Validate GPU silhouette coefficient implementation
python scripts/validate_gpu_silhouette.py

# This will run accuracy tests comparing GPU vs CPU silhouette scores
# on tiny datasets (10 samples, 3 features, 2 clusters) to ensure
# numerical precision within 1e-6 tolerance
```

### Evaluation Commands
```bash
# MMLU evaluation using lm-eval
accelerate launch -m lm_eval --model hf \
    --model_args "pretrained=meta-llama/Llama-2-7b-hf,peft=outputs/stage_3_finetune/YYYY-MM-DD/HH-MM-SS/checkpoint-XXXX" \
    --tasks mmlu \
    --batch_size auto \
    --output_path outputs/stage_4_eval

# Multi-task evaluation
accelerate launch -m lm_eval --model hf \
    --model_args "pretrained=meta-llama/Llama-2-7b-hf,peft=outputs/stage_3_finetune/YYYY-MM-DD/HH-MM-SS/checkpoint-XXXX" \
    --tasks mmlu,hellaswag,arc_easy,arc_challenge \
    --batch_size auto \
    --output_path outputs/stage_4_eval
```

## Important Configuration Files

- `configs/stage_1_warmup.yaml` - Router warmup training configuration
- `configs/stage_2_selection.yaml` - Data selection configuration  
- `configs/stage_3_finetune.yaml` - Target model fine-tuning configuration
- `configs/stage_4_evaluate.yaml` - Model evaluation configuration
- `configs/accelerate_config_*.yaml` - Distributed training configurations

## Key Parameters

### Stage 1 (Warmup)
- `training.peft_mode`: Training mode (`lora` or `full_rank`)
  - `lora`: Low-rank adaptation using LoRA adapters for specified modules
  - `full_rank`: Full-rank training of router parameters (quality gate and MoE gate only)
- `training.learning_rate`: Learning rate (default: 1e-4)
- `dataset.subset_ratio`: Training data proportion (default: 0.05)
- `training.quality_loss_weight`: Weight for quality classification loss (default: 0.01)
- `training.quality_gate_init_mean/std`: Quality gate initialization parameters
- `training.trash_expert_mode`: Trash expert behavior ("zero", "noise", "custom")
- `training.enable_load_balancing`: Enable MoE load balancing loss

#### Quality Loss Configuration
- `training.quality_loss_type`: Loss function type (default: "sigmoid")
  - `"sigmoid"`: Direct sigmoid loss
  - `"beta_moment_matching"`: Moment matching with target Beta distribution
  - `"mean_variance_regularization"`: Mean centering with variance regularization
  - `"mse"`: Mean squared error loss
- `training.quality_loss_debug`: Enable debug output (default: false)
- `training.quality_loss_params`: Parameters for custom loss functions
  - **Beta Moment Matching**:
    - `beta_target_mean`: Target mean (default: 0.5)
    - `beta_target_var`: Target variance (default: 0.05)  
    - `w_mean`: Weight for mean loss (default: 1.0)
    - `w_var`: Weight for variance loss (default: 1.0)
  - **Mean-Variance Regularization**:
    - `lambda_var`: Variance regularization weight (default: 0.1)
  - **Data Encoding Configuration**:
    - `full_sequence_prediction`: Enable full sequence prediction vs assistant-only SFT (default: true)
    - `mask_special_tokens`: Mask special format tokens like <|user|>, <|assistant|> (default: true)
    - `sample_wise_averaging`: Use sample-wise vs token-wise loss averaging (default: true)

### Stage 2 (Router Data Computation)
- `model_checkpoint_path`: Path to Stage 1 output weights
- Router inference parameters for data quality scoring
- Outputs router data files for subsequent selection algorithms

### Data Selection (Standalone Scripts)
- `selection_percentage`: Data selection ratio (default: 0.05)
- `router_data_dir`: Path to Stage 2 router data output directory
- `clustering_method`: Clustering algorithm (currently only 'kmeans' supported)
- `debug_print`: Enable detailed clustering debug output (default: false)
- `clustering_params`: Clustering-specific parameters
  - **K-Means Parameters**:
    - `auto_k`: Auto k-value selection using Elbow Method (default: true)
    - `k`: Manual k-value (only when auto_k=false)
    - `k_range`: K-value search range for Elbow Method (default: [10, 100])
    - `max_iters`: Maximum iterations (default: 300)
    - `enable_parallel_kmeans`: Enable multi-GPU parallel computation for k-value search (default: false)
    - `parallel_processes`: Number of parallel processes for k-value computation (default: 4)
    - `gpu_allocation_strategy`: GPU allocation strategy - "round_robin" or "balanced" (default: "round_robin")

### Stage 3 (Finetune)
- `training.lora.r`: LoRA rank (default: 128)
- `training.learning_rate`: Learning rate (default: 2e-5)
- `dataset.data_path`: Path to Stage 2 selected data

## Memory Requirements

- **Stage 1 (Warmup)**: 
  - `lora` mode: LoRA fine-tuning requires ~8GB GPU memory
  - `full_rank` mode: Full-rank router training requires ~16GB GPU memory
  - Supports both Qwen2.5-1.5B and larger models with FSDP
- **Stage 2**: 
  - Serial mode: K-means clustering with GPU acceleration requires moderate GPU memory (~4-8GB depending on dataset size)
  - Parallel mode: Multi-GPU parallel K-means requires ~2-4GB per GPU (distributed across multiple devices)
  - Parallel processes can be allocated flexibly across available GPUs for optimal resource utilization
- **Stage 3**: Llama-2-7B LoRA training requires ~24GB GPU memory
- **Multi-GPU**: Use FSDP configurations for distributed training

## Path Dependencies

Ensure correct path dependencies between stages:
- Stage 2 requires Stage 1's `full_rank_weights.pt`
- **Data Selection** requires Stage 2's router data files (`*_router_data.pt`)
- Stage 3 requires selected data (`selected_data.jsonl` from standalone selection scripts)
- Stage 4 requires Stage 3's LoRA checkpoint

Always verify output paths before proceeding to the next stage.

## Pipeline Architecture Changes

### Decoupled Stage 2 Architecture
**Stage 2** has been split into two distinct phases for better modularity:

1. **Router Data Computation** (`src/stages/selection.py`):
   - Pure router inference on training data
   - Saves quality scores and MoE logits to `router_data/` directory
   - No data selection logic - focused solely on computation
   - Memory efficient with proper GPU cleanup

2. **Data Selection Algorithms** (`src/selection/`):
   - Standalone scripts for various selection strategies
   - Load router data from Stage 2 output
   - Apply clustering-based selection algorithms
   - Output final `selected_data.jsonl` for Stage 3

### Benefits of Decoupled Architecture:
- **Experimental Flexibility**: Test different selection algorithms on same router data
- **Resource Efficiency**: Separate GPU memory requirements for inference vs. clustering
- **Debugging**: Isolate router computation issues from selection algorithm issues
- **Scalability**: Run selection algorithms across multiple experiments in batch mode

## Architecture Overview

### Core Components
- **Two-tier Routing**: Quality gate assessment followed by MoE expert routing
- **Quality Gate**: Outputs single quality score with sigmoid transformation to good/bad ratios
- **Trash Expert**: Handles low-quality data with configurable output modes
- **Loss Framework**: Extensible loss functions with proper padding token handling
- **Clustering-Based Selection**: GPU-accelerated clustering algorithms for data diversity
  - **K-Means + Elbow Method**: Automatic optimal k-value selection with multi-GPU parallel processing
  - **GPU Silhouette Score**: Precise GPU computation for clustering evaluation
  - **Round-Robin Selection**: Ensures balanced selection from all clusters
  - **Cosine Distance**: Semantic clustering using MoE logits as features
  - **Extensible Interface**: Clean architecture for future clustering algorithm additions

### Router Output Format
```python
# Current format
quality_score = layer_output["quality_score"]    # Shape: [batch, seq_len, 1]
moe_logits = layer_output["moe_logits"]          # Shape: [batch*seq_len, num_experts]

# Compute ratios
good_ratio = torch.sigmoid(quality_score)        # Shape: [batch, seq_len, 1]
bad_ratio = 1.0 - good_ratio                     # Shape: [batch, seq_len, 1]
```

## Programmatic Usage

```python
from src.models.select_moe import SelectMoeForCausalLM, register_select_moe

# Register Select-MoE (required before loading)
register_select_moe()

# Load converted model
model = SelectMoeForCausalLM.from_pretrained("./converted_models/select_moe_converted_OLMoE-1B-7B-0125")

# Training mode (enable router logits)
outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_router_logits=True)

# Process router logits
for layer_output in outputs.router_logits:
    quality_score = layer_output["quality_score"]   # Shape: [batch, seq_len, 1]
    moe_logits = layer_output["moe_logits"]         # Shape: [batch*seq_len, num_experts]
    # Compute good/bad ratios
    good_ratio = torch.sigmoid(quality_score)       # Shape: [batch, seq_len, 1]
    bad_ratio = 1.0 - good_ratio                    # Shape: [batch, seq_len, 1]

# Configure quality loss type
model.config.quality_loss_type = "beta_moment_matching"
model.config.quality_loss_debug = True

# Configure loss parameters
model.config.beta_target_mean = 0.5
model.config.beta_target_var = 0.05
model.config.w_mean = 1.0
model.config.w_var = 1.0
model.config.lambda_var = 0.1
```

## Code Structure Map

### 📂 Root Directory
- `pyproject.toml` - Python project configuration with dependencies and build settings
- `uv.lock` - UV lock file for reproducible dependency management
- `TODO` - Development task tracking and progress notes
- `CLAUDE.md` - This file: comprehensive project documentation and AI guidance
- `README.md` - Project overview and quick start guide
- `笔记.md` - Chinese development notes and documentation

### 📂 configs/
Configuration files for different pipeline stages using Hydra framework:
- `stage_1_pretrain.yaml` - Router training configuration (PEFT mode, learning rates, quality loss)
- `stage_2_selection.yaml` - Data selection configuration (clustering methods, selection ratios)
- `stage_3_finetune.yaml` - Target model fine-tuning configuration (LoRA parameters)
- `batch_selection.yaml` - Batch data selection configuration for processing multiple experiments
- `continue_selection.yaml` - Standalone clustering selection script configuration
- `accelerate_config/` - Multi-GPU distributed training configurations:
  - `DDP.yaml` - Distributed Data Parallel configuration
  - `FSDP.yaml` - Fully Sharded Data Parallel configuration
  - `SINGLE.yaml` - Single GPU configuration
- `training/` - Model-specific training configurations:
  - `llama_2_7b.yaml` - Llama-2-7B training parameters
  - `qwen_2.5_1.5b.yaml` - Qwen2.5-1.5B training parameters
  - `qwen_3_1.7b.yaml` - Qwen-3-1.7B training parameters

### 📂 src/
Main source code organized by functionality:

#### 📂 src/models/
- `select_moe.py` - Core Select-MoE model implementation with two-tier routing architecture
- `__init__.py` - Model package initialization and exports

#### 📂 src/data/
- `dataset_loader.py` - Multi-dataset loading and preprocessing (CoT, Dolly, FLAN-v2, OASST1)
- `__init__.py` - Data package initialization and exports

#### 📂 src/clustering/
GPU-accelerated clustering algorithms for intelligent data selection:
- `cluster_selection.py` - Main clustering-based selection coordinator class with extensible interface
- `kmeans_clustering.py` - GPU K-means with automatic k-value selection via Elbow Method and multi-GPU parallel processing
- `parallel_kmeans.py` - Multi-GPU parallel K-means implementation for accelerated k-value search
- `gpu_metrics.py` - High-performance GPU silhouette coefficient computation for cosine distance
- `__init__.py` - Clustering package initialization and exports

#### 📂 src/selection/
Decoupled data selection module with consolidated selection algorithms:
- `data_selection.py` - Core data selection functions and utilities
  - `cluster_based_selection()` - Main clustering-based selection algorithm
  - `load_all_router_data()` - Load router data from multiple datasets
  - `load_original_dataset_mapping()` - Load original dataset message mappings
  - `rebuild_scored_data_with_messages()` - Rebuild scored data with complete messages
  - `rebuild_logits_data()` - Process MoE logits for clustering algorithms
  - `parse_clustering_params()` - Parse and validate clustering configuration
- `__init__.py` - Selection module exports for easy importing

#### 📂 src/stages/
Pipeline stage implementations following the four-stage architecture:
- `warmup.py` - Stage 1 implementation: Select-MoE router warmup training with quality loss
- `selection.py` - Stage 2 implementation: **router inference and data computation only** (selection logic moved to `src/selection/`)
- `finetune.py` - Stage 3 implementation: target model LoRA fine-tuning with selected data
- `__init__.py` - Stages package initialization and exports

#### 📂 src/training/
Training utilities and distributed computing support:
- `full_rank_finetuning.py` - Full-rank weight loading and saving utilities for router training

#### 📂 src/utils/
Utility functions for various system operations:
- `hydra_resolvers.py` - Custom Hydra resolvers for configuration management
- `logging_utils.py` - Enhanced logging utilities and configuration
- `tools.py` - General utility functions and tools
- `__init__.py` - Utilities package initialization and exports

### 📂 scripts/
Executable scripts for various operations:
- `run_stage_1.sh` - Stage 1 router warmup training execution script
- `run_stage_2.sh` - Stage 2 router data computation execution script (inference only)
- `run_stage_3.sh` - Stage 3 target model fine-tuning execution script
- `eval.sh` - Stage 4 model evaluation execution script
- `continue_selection.py` - **Standalone clustering-based data selection script** (Stage 2b: processes router data from Stage 2)
- `batch_selection.py` - **Batch data selection script** (Stage 2c: processes multiple experiments automatically)
- `convert_olmoe_to_select_moe.py` - Model conversion from OLMoE to Select-MoE format
- `compare_converted_model.py` - Model conversion verification and comparison
- `convert.sh` - Shell script wrapper for model conversion operations

### 📂 exp_scripts/
Experimental and analysis scripts:
- `make_result_table.py` - Generate evaluation result tables from lm-eval outputs
- `test_encoding_modes.py` - Test different data encoding modes and compare results

### 📂 docs/
Comprehensive project documentation:
- `README_continue_selection.md` - Detailed guide for standalone data selection scripts
- `custom_loss_fn.md` - Documentation for custom loss function implementations
- `docs.md` - General project documentation and guides
- `gpu_fps_fixes_summary.md` - GPU performance optimization and fixes summary
- `router_data_format.md` - Router data format specifications and examples

### 📂 examples/
Example scripts and analysis tools:
- `comprehensive_analysis.py` - Data analysis and visualization tools for router outputs
- `visualize_custom_loss.py` - Visualization tools for custom loss function analysis
- `README.md` - Examples documentation and usage instructions

### 📂 ref_model/
Reference model implementations for comparison and development:
- `olmoe/` - Original OLMoE model implementation and configuration:
  - `configuration_olmoe.py` - OLMoE configuration class
  - `modeling_olmoe.py` - OLMoE model implementation
  - `__init__.py` - OLMoE package initialization
- `moe_plus_plus/` - MoE++ model reference implementation:
  - `configuration_moe_plus_plus.py` - MoE++ configuration class
  - `modeling_moe_plus_plus.py` - MoE++ model implementation
  - `moe_plus_plus_layer.py` - MoE++ layer implementations

### 📂 dataset/
Training and evaluation datasets:
- `train/processed/` - Preprocessed training data:
  - `cot/cot_data.jsonl` - Chain-of-thought dataset
  - `dolly/dolly_data.jsonl` - Dolly instruction dataset
  - `flan_v2/flan_v2_data.jsonl` - FLAN-v2 dataset
  - `oasst1/` - OpenAssistant dataset with variants
- `eval/` - Evaluation datasets:
  - `bbh/` - Big Bench Hard evaluation data
  - `mmlu/` - MMLU evaluation data
  - `tydiqa/` - TyDiQA evaluation data
- `selected_data/` - Pre-selected data samples for various tasks and seeds

### 📂 converted_models/
Model conversion outputs:
- `select_moe_converted_OLMoE-1B-7B-0125/` - Converted Select-MoE model files

### 📂 outputs/
Pipeline execution outputs organized by stage and timestamp:
- `stage_1_pretrain/` - Router pretraining outputs with model weights and logs
- `stage_2_selection/` - Router data computation outputs with router data files
- `stage_3_finetune/` - Target model fine-tuning outputs with LoRA checkpoints
- `stage_4_eval/` - Model evaluation results
- `continue_selection/` - Standalone selection script outputs
- `batch_selection/` - Batch processing outputs with reports
- `visual_figs/` - Generated analysis figures and visualizations

### 📂 tools/
Development and system tools:
- `install.sh` - Development environment setup and tool installation script
- `chsrc` - Chinese source configuration tool

### 📂 LESS/
LESS (Low-rank Expert Selection System) reference implementation:
- Complete reference implementation with evaluation scripts
- Training utilities and data processing tools
- Evaluation benchmarks for multiple tasks

### Key Entry Points
- `src/main.py` - Main pipeline coordinator with Hydra configuration management
- **Data Selection Scripts**:
  - `scripts/continue_selection.py` - Single experiment data selection with clustering algorithms
  - `scripts/batch_selection.py` - Batch processing for multiple experiments
- `scripts/run_stage_*.sh` - Stage-specific execution scripts with parameter overrides

### Recommended Workflow
1. **Stage 1**: Train Select-MoE router using `scripts/run_stage_1.sh` (warmup training)
2. **Stage 2**: Compute router data using `scripts/run_stage_2.sh` 
3. **Stage 2b/2c**: Apply data selection algorithms using standalone scripts:
   - Single experiment: `scripts/continue_selection.py`
   - Multiple experiments: `scripts/batch_selection.py`
4. **Stage 3**: Fine-tune target model using `scripts/run_stage_3.sh`
5. **Stage 4**: Evaluate model performance using `scripts/eval.sh`
