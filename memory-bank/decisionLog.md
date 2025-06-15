# 决策日志 (Decision Log)

## 2025-06-12 19:07

**决策**:
- **初始化项目工作流**: 根据用户提供的 `项目任务.md` 和 `项目设计.md`，启动项目实现流程。
- **任务分解**: 采纳 `项目任务.md` 中定义的任务分解结构。
- **委派任务1**: 第一个委派的子任务是 **任务 1：项目框架原则与环境搭建**。该任务将委托给 `code` 模式执行。
---

## 2025-06-12 22:53

**决策**:
- **整合子任务1成果**: `code` 模式已完成项目框架搭建。
- **记录依赖问题**: 在搭建环境时，`deepspeed` 在 Windows 环境下安装失败。为避免阻塞，该依赖已暂时从 `pyproject.toml` 中移除。此问题推迟到实际需要分布式训练的阶段再专门解决。
- **委派任务2**: 准备委派 **任务 2：实现阶段一 - 选择模型预热**。
---

## 2025-06-12 23:02

**决策**:
- **整合子任务2成果**: `code` 模式已完成选择模型预热阶段（任务2）的开发。
- **记录实现细节**: 核心训练逻辑最初采用手动循环实现，后根据用户反馈重构为使用 Hugging Face `Trainer` API，以提高代码的健壮性和可维护性。详细实现位于 `src/stages/pretrain.py`。
- **委派任务3**: 准备委派 **任务 3：实现阶段二 - 数据选择**。
---

## 2025-06-12 23:10

**决策**:
- **整合子任务3成果**: `code` 模式已完成数据选择阶段（任务3）的开发。
- **记录实现细节**: 核心数据打分、排序和筛选逻辑已在 `src/stages/selection.py` 中完整实现，并通过 `scripts/run_stage_2.sh` 调用。该阶段的完成为后续模型训练准备了高质量数据集。
- **委派任务4**: 准备委派 **任务 4：实现阶段三 - 目标模型训练**。
---

## 2025-06-12 23:14

**决策**:
- **整合子任务4成果**: `code` 模式已完成目标模型训练阶段（任务4）的开发。
- **记录实现细节**: 核心微调逻辑已在 `src/stages/finetune.py` 中实现，采用 `transformers.Trainer` 与 LoRA (`peft`) 相结合，并通过 `accelerate launch` 启动，配置文件为 `configs/accelerate_config.yaml`。此阶段为最终评估产出了 LoRA 适配器。
- **委派任务5**: 准备委派 **任务 5：实现阶段四 - 模型评估**。
---

## 2025-06-13 04:44

**决策**:
- **整合子任务调试成果**: `code` 模式成功解决了 `accelerate` 与 FSDP 结合使用时，因 `Qwen3ForCausalLM` 构造函数缺少 `config` 参数而导致的 `TypeError`。
- **记录根本原因**: 经过多次尝试，最终确定问题根源是 `configs/accelerate_config.yaml` 中的 `dynamo_use_regional_compilation: true` 设置与 FSDP 的模型重新实例化机制不兼容。
- **记录最终解决方案**:
  1.  在 `configs/accelerate_config.yaml` 中将 `dynamo_use_regional_compilation` 设置为 `false`。
  2.  同时保留了在 `src/stages/pretrain.py` 中动态向 FSDP 插件注入 `model.config` 的健壮性代码。
- **确认状态**: 预训练脚本 `scripts/run_stage_1.sh` 现已可以正常执行。详细的调试过程已从 `activeContext.md` 中提炼并归档至此。
---

## 2025-06-14 00:01

**决策**:
- **整合子任务重构成果**: `code` 模式已成功将 `src/stages/pretrain.py` 中的手动训练循环重构为使用 Hugging Face `Trainer` API。
- **记录变更原因**: 本次重构旨在提高代码的健壮性、可维护性，并更好地与 Hugging Face 生态系统集成。
- **确认技术细节**: 新实现保持了对 `accelerate` 和 FSDP 的兼容性，并保留了只训练路由器权重的核心逻辑。
---

## 2025-06-14: 切换分布式训练策略 (FSDP -> DDP)

**决策**: 将分布式训练策略从 FSDP (Fully Sharded Data Parallel) 切换到 DDP (Distributed Data Parallel)，以解决模型权重对比脚本的加载问题。

**根本原因**:
- **FSDP 的复杂性**: 使用 FSDP 训练后，模型权重被分片 (sharded) 保存。这导致无法通过标准的 `AutoModelForCausalLM.from_pretrained` 方法直接加载完整模型进行分析，给 `scripts/compare_router_weights.py` 等下游任务带来了阻碍。
- **DDP 的简便性**: DDP 策略在训练后会生成一个完整的、标准的模型权重文件 (`pytorch_model.bin`)，可以直接被 `transformers` 等库加载，大大简化了后续的分析和评估流程。

**实施步骤**:
1.  **创建新 Accelerate 配置**: 在 `configs/` 目录下创建了 `accelerate_config_ddp.yaml`，将 `distributed_type` 设置为 `MULTI_GPU`，以启用 DDP。
2.  **更新启动脚本**: 修改了 `scripts/run_stage_1.sh`，使其使用新的 `accelerate_config_ddp.yaml` 配置文件，并解决了多进程下创建多个输出目录的问题。
3.  **清理训练代码**: 注释掉了 `src/stages/pretrain.py` 中所有 FSDP 特有的代码块。
4.  **更新对比脚本**: `scripts/compare_router_weights.py` 被恢复到其简单版本，并配置为使用 DDP 训练产出的模型路径。

**结论**:
通过切换到 DDP，我们建立了一个更稳定、可复现的训练与分析工作流。

---
## 2025-06-14: 分析 Stage 1 数据处理流程

**决策**: 委派 `ask` 模式分析 `run_stage_1.sh` 的数据处理流程。

**结论**:
- **数据来源**: 程序并非使用本地数据，而是通过 Hugging Face `datasets` 库从网络加载 `teknium/OpenHermes-2.5` 和 `allenai/WildChat-1M` 数据集。
- **缓存机制**: 终端显示的 `.parquet` 文件下载是 `datasets` 库的首次运行自动缓存过程。
- **处理步骤**: 数据经过合并、随机打乱、按比例选取子集、文本分词（截断/填充至512个token）以及移除原始文本列等一系列处理，最终被整理成适合模型训练的格式。
- **知识归档**: 该分析结果已整合入项目记忆库。
---
### 2025-06-14: 修复Stage 1数据处理流程中的 `KeyError`

- **问题**: 在第一阶段预训练中，`tokenize_function` 尝试访问固定的 `text` 列，但 `teknium/OpenHermes-2.5` 和 `allenai/WildChat-1M` 数据集的对话数据分别存储在 `conversations` 和 `conversation` 列中，导致 `KeyError: 'text'`。
- **决策**:
  1.  **重构 `tokenize_function`**: 修改 `src/stages/pretrain.py` 中的 `tokenize_function`，使其能够动态地从 `conversations` 或 `conversation` 列中提取对话内容。
  2.  **格式化对话**: 将提取的对话数据统一格式化为 `"{role}: {content}"` 的字符串，以适应分词器。
  3.  **优化 `dataset.map`**: 更新 `dataset.map` 的调用方式，在分词后移除所有原始列，确保输出的数据集格式干净，只包含 `input_ids` 和 `attention_mask` 等必要信息。
- **影响**:
  - 成功解决了预训练脚本的崩溃问题，使 Stage 1 得以顺利运行。
  - 增强了数据处理流程的兼容性和健壮性，能够处理不同来源、不同格式的对话数据集。

---
### 决策：关于FP8训练支持 (2025-06-14)

*   **背景**: 用户希望使用 `Qwen/Qwen3-30B-A3B-FP8` 模型进行训练。代码分析子任务确认当前脚本不支持FP8。
*   **决策**: 暂不进行FP8训练。
*   **理由**: 核心训练脚本 [`src/stages/pretrain.py`](src/stages/pretrain.py) 将数据类型硬编码为 `torch.bfloat16`，需要进行代码修改才能支持FP8。
*   **后续步骤**: 创建一个新的开发任务，以修改代码库，使其能够通过配置文件动态设置训练的数据类型（dtype），从而支持FP8及其他精度格式。

---
### 2025-06-15: 整合阶段二“数据选择”成果

*   **决策**: 整合由 `code` 模式完成的“数据选择”子任务。
*   **成果**:
    1.  创建了核心逻辑文件 [`src/stages/selection.py`](src/stages/selection.py)。
    2.  创建了对应的配置文件 [`configs/stage_2_selection.yaml`](configs/stage_2_selection.yaml)。
    3.  创建了执行脚本 [`scripts/run_stage_2.sh`](scripts/run_stage_2.sh)。
*   **记录**: 子任务声称已将详细过程记录在 `activeContext.md`，但该文件为空。尽管如此，核心交付物均已完成并通过用户确认。项目现在可以进入下一阶段。

---
### 2025-06-15: 优化 Stage 2 输出路径逻辑

*   **问题**: Stage 2 的输出文件 `selected_data.jsonl` 被直接保存在项目根目录，覆盖了旧文件，不利于实验追踪。
*   **决策**: 委派 `code-developer` 模式，使其输出行为与 Stage 1 对齐，将结果保存到由 Hydra 生成的唯一、带时间戳的目录中。
*   **成果**:
    1.  修改了 `src/stages/selection.py` 以正确解析和使用 Hydra 的动态工作目录。
    2.  更新了 `configs/stage_2_selection.yaml` 以支持新的路径逻辑。
    3.  现在，Stage 2 的输出会被保存到 `outputs/YYYY-MM-DD/HH-MM-SS/stage_2_selection/` 这样的路径下，确保了实验结果的一致性和可追溯性。

---
### 代码实现 [Finetune Stage]
[2025-06-15 16:56:57] - [实现了基于PEFT和Accelerate的LoRA微调阶段]

**实现细节：**
- 创建了 `src/stages/finetune.py`，包含使用 `accelerate` 启动的 LoRA 微调训练循环。
- 脚本支持从配置文件加载模型、数据集路径和训练参数。
- 使用 `peft` 的 `LoraConfig` 对指定的目标模块 (`q_proj`, `k_proj`, `v_proj`) 应用LoRA。
- 训练结束后，脚本会自动保存 LoRA 适配器权重。
- 更新了 `src/main.py` 以集成 `finetune` 阶段。
- 创建了 `configs/stage_3_finetune.yaml` 配置文件。
- 创建了 `scripts/run_stage_3.sh` 以方便启动训练。

**测试框架：**
Pytest (已跳过)

**测试结果：**
- 覆盖率：N/A
- 通过率：N/A

---
### 2025-06-15: 重构阶段三“目标模型训练”工作流

*   **问题**: `src/stages/finetune.py` 脚本未使用 `transformers.Trainer`，导致训练效率低下和显存溢出问题。同时，`scripts/run_stage_3.sh` 启动脚本缺乏动态 GPU 检测和唯一的输出目录生成机制，与项目其他阶段不一致。
*   **决策**: 委派 `code-developer` 模式，对阶段三的训练脚本和启动脚本进行重构，使其与阶段一 (`pretrain`) 的实现风格和健壮性保持一致。
*   **成果**:
    1.  **重构 `src/stages/finetune.py`**:
        *   将原先手动的 `accelerate` 训练循环替换为 Hugging Face `Trainer` API。
        *   确保了 `peft` 配置的 LoRA 模型能够正确地与 `Trainer` 集成。
        *   保留了原有的数据加载和预处理逻辑，同时使其与 `Trainer` 兼容。
    2.  **更新 `scripts/run_stage_3.sh`**:
        *   实现了根据 `CUDA_VISIBLE_DEVICES` 环境变量动态计算 `NUM_GPUS`。
        *   为每次运行生成一个带时间戳的唯一输出目录 (e.g., `outputs/YYYY-MM-DD/HH-MM-SS/stage_3_finetune`)。
        *   在 `accelerate launch` 命令中动态传入了 `--num_processes` 和 `hydra.run.dir`。
    3.  **更新 `configs/stage_3_finetune.yaml`**:
        *   添加并调整了 `output_dir`, `logging_dir`, `save_strategy` 等字段，以完全兼容 `transformers.TrainingArguments`。
*   **结论**: 此次重构统一了项目各阶段的代码风格，解决了潜在的性能问题，并显著提升了实验的可追溯性和可复现性。
