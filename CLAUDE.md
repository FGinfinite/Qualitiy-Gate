# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Select-MoE is a data selection framework using Mixture-of-Experts (MoE) models. The project implements a four-stage pipeline:

1. **Stage 1 (Pretrain)**: Train a Select-MoE router to learn data quality discrimination
2. **Stage 2 (Selection)**: Use the trained router to score and filter training data
3. **Stage 3 (Finetune)**: Fine-tune target models (Llama-2-7B) with selected data using LoRA
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

### Clustering-Based Selection (`src/clustering/`)
- **GPU K-Means with Elbow Method**: Automatic k-value selection using GPU acceleration
- **GPU HDBSCAN**: Parameter-free clustering with RAPIDS cuML support and CPU fallback
- **GPU Silhouette Score**: High-performance GPU-accelerated silhouette coefficient calculation for cosine distance
- **Round-Robin Selection**: Selects high-quality data from each cluster using quality scores
- **Cosine Distance Metric**: Uses cosine similarity for better semantic clustering of MoE logits
- **ClusterBasedSelection**: Main selection class integrating both clustering algorithms

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
# Stage 1: Router pretraining (set CUDA_VISIBLE_DEVICES first)
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_stage_1.sh

# Stage 2: Clustering-based data selection
CUDA_VISIBLE_DEVICES=0 bash scripts/run_stage_2.sh model_checkpoint_path=outputs/stage_1_pretrain/YYYY-MM-DD/HH-MM-SS/full_rank_weights.pt

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

# Override selection percentage and clustering method
bash scripts/run_stage_2.sh selection_percentage=0.1 clustering_method=hdbscan

# Override K-Means parameters
bash scripts/run_stage_2.sh clustering_method=kmeans clustering_params.k=50 clustering_params.auto_k=false

# Enable K-Means multi-GPU parallel computation
bash scripts/run_stage_2.sh clustering_method=kmeans clustering_params.enable_parallel_kmeans=true clustering_params.parallel_processes=8

# Override K-Means parallel parameters with custom GPU allocation
bash scripts/run_stage_2.sh clustering_method=kmeans clustering_params.enable_parallel_kmeans=true clustering_params.parallel_processes=12 clustering_params.gpu_allocation_strategy=balanced

# Override HDBSCAN parameters
bash scripts/run_stage_2.sh clustering_method=hdbscan clustering_params.min_cluster_size=100 clustering_params.auto_tune=true

# Override LoRA parameters
bash scripts/run_stage_3.sh training.lora.r=64 training.lora.lora_alpha=128

# Enable clustering debug output
bash scripts/run_stage_2.sh debug_print=true

# Use standalone clustering selection with debug
CUDA_VISIBLE_DEVICES=1 uv run scripts/continue_selection.py debug_print=true

# Use standalone clustering selection with multi-GPU parallel K-means
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run scripts/continue_selection.py clustering_method=kmeans clustering_params.enable_parallel_kmeans=true clustering_params.parallel_processes=8
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

- `configs/stage_1_pretrain.yaml` - Router pretraining configuration
- `configs/stage_2_selection.yaml` - Data selection configuration  
- `configs/stage_3_finetune.yaml` - Target model fine-tuning configuration
- `configs/stage_4_evaluate.yaml` - Model evaluation configuration
- `configs/accelerate_config_*.yaml` - Distributed training configurations

## Key Parameters

### Stage 1 (Pretrain)
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

### Stage 2 (Selection)
- `selection_percentage`: Data selection ratio (default: 0.05)
- `model_checkpoint_path`: Path to Stage 1 output weights
- `clustering_method`: Clustering algorithm ('kmeans' or 'hdbscan', default: 'kmeans')
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
  - **HDBSCAN Parameters**:
    - `min_cluster_size`: Minimum cluster size (auto-estimated if not specified)
    - `min_samples`: Minimum samples (auto-set if not specified)  
    - `metric`: Distance metric (default: 'cosine')
    - `use_gpu`: Enable GPU acceleration (default: true)
    - `auto_tune`: Enable automatic parameter tuning (default: false)

### Stage 3 (Finetune)
- `training.lora.r`: LoRA rank (default: 128)
- `training.learning_rate`: Learning rate (default: 2e-5)
- `dataset.data_path`: Path to Stage 2 selected data

## Memory Requirements

- **Stage 1**: 
  - `lora` mode: LoRA fine-tuning requires ~8GB GPU memory
  - `full_rank` mode: Full-rank router training requires ~16GB GPU memory
- **Stage 2**: 
  - Serial mode: Clustering-based data selection with GPU acceleration requires moderate GPU memory (~4-8GB depending on dataset size)
  - Parallel mode: Multi-GPU parallel K-means requires ~2-4GB per GPU (distributed across multiple devices)
  - Parallel processes can be allocated flexibly across available GPUs for optimal resource utilization
- **Stage 3**: Llama-2-7B LoRA training requires ~24GB GPU memory
- **Multi-GPU**: Use FSDP configurations for distributed training

## Path Dependencies

Ensure correct path dependencies between stages:
- Stage 2 requires Stage 1's `full_rank_weights.pt`
- Stage 3 requires Stage 2's `selected_data.jsonl`
- Stage 4 requires Stage 3's LoRA checkpoint

Always verify output paths before proceeding to the next stage.

## Architecture Overview

### Core Components
- **Two-tier Routing**: Quality gate assessment followed by MoE expert routing
- **Quality Gate**: Outputs single quality score with sigmoid transformation to good/bad ratios
- **Trash Expert**: Handles low-quality data with configurable output modes
- **Loss Framework**: Extensible loss functions with proper padding token handling
- **Clustering-Based Selection**: GPU-accelerated clustering algorithms for data diversity
  - **K-Means + Elbow Method**: Automatic optimal k-value selection
  - **HDBSCAN**: Parameter-free density-based clustering
  - **GPU Silhouette Score**: Precise GPU computation replacing CPU bottlenecks
  - **Round-Robin Selection**: Ensures balanced selection from all clusters
  - **Cosine Distance**: Semantic clustering using MoE logits as features

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
- `TODO` - Development task tracking and progress notes
- `CLAUDE.md` - This file: comprehensive project documentation and AI guidance

### 📂 configs/
Configuration files for different pipeline stages using Hydra framework:
- `stage_1_pretrain.yaml` - Router training configuration (PEFT mode, learning rates, quality loss)
- `stage_2_selection.yaml` - Data selection configuration (clustering methods, selection ratios)
- `stage_3_finetune.yaml` - Target model fine-tuning configuration (LoRA parameters)
- `stage_4_evaluate.yaml` - Model evaluation configuration (lm-eval settings)
- `continue_selection.yaml` - Standalone clustering selection script configuration
- `accelerate_config_*.yaml` - Multi-GPU distributed training configurations

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
- `cluster_selection.py` - Main clustering-based selection coordinator class
- `kmeans_clustering.py` - GPU K-means with automatic k-value selection via Elbow Method
- `hdbscan_clustering.py` - GPU HDBSCAN with parameter auto-tuning and RAPIDS cuML support
- `gpu_metrics.py` - High-performance GPU silhouette coefficient computation for cosine distance
- `__init__.py` - Clustering package initialization and exports

#### 📂 src/stages/
Pipeline stage implementations following the four-stage architecture:
- `pretrain.py` - Stage 1 implementation: Select-MoE router pretraining with quality loss
- `selection.py` - Stage 2 implementation: router inference, data scoring, and clustering selection
- `finetune.py` - Stage 3 implementation: target model LoRA fine-tuning with selected data
- `evaluation.py` - Stage 4 implementation: model performance evaluation using lm-eval
- `__init__.py` - Stages package initialization and exports

#### 📂 src/training/
Training utilities and distributed computing support:
- `full_rank_finetuning.py` - Full-rank weight loading and saving utilities for router training
- `__init__.py` - Training package initialization and exports

#### 📂 src/utils/
- `__init__.py` - Utilities package initialization and exports

### 📂 scripts/
Executable scripts for various operations:
- `run_stage_1.sh` - Stage 1 router pretraining execution script
- `run_stage_2.sh` - Stage 2 data selection execution script  
- `run_stage_3.sh` - Stage 3 target model fine-tuning execution script
- `eval.sh` - Stage 4 model evaluation execution script
- `continue_selection.py` - Standalone clustering-based data selection script
- `convert_olmoe_to_select_moe.py` - Model conversion from OLMoE to Select-MoE format
- `compare_converted_model.py` - Model conversion verification and comparison
- `validate_gpu_silhouette.py` - GPU silhouette coefficient implementation validation

### 📂 ref_model/
Reference model implementations for comparison and development:
- `olmoe/` - Original OLMoE model implementation and configuration
- `moe_plus_plus/` - MoE++ model reference implementation

### 📂 examples/
- `comprehensive_analysis.py` - Data analysis and visualization tools for router outputs

### 📂 tools/
- `install.sh` - Development environment setup and tool installation script

### Key Entry Points
- `src/main.py` - Main pipeline coordinator with Hydra configuration management
- `scripts/continue_selection.py` - Standalone data selection with clustering algorithms
- `scripts/run_stage_*.sh` - Stage-specific execution scripts with parameter overrides
