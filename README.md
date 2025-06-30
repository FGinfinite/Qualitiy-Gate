# Select-MoE: 基于混合专家模型的数据选择策略

## 项目简介

本项目是一个创新的数据选择实验，旨在探索一种利用混合专家模型（Mixture-of-Experts, MoE）进行高效数据筛选的方法。核心思想是：首先对一个小型MoE模型的Router（路由器）进行预热微调，使其具备数据质量的判别能力；然后，利用这个预热好的Router为大规模数据集打分，筛选出高质量的数据子集；最后，使用这些筛选出的高质量数据来微调一个更大规模的目标模型，并评估该数据选择策略的最终效果。

## 技术栈

本项目主要依赖以下技术和库：

-   **评估框架**: [`lm-eval`](https://github.com/EleutherAI/lm-evaluation-harness)
-   **启动与分布式训练**: [`accelerate`](https://github.com/huggingface/accelerate)
-   **配置管理**: [`hydra`](https://github.com/facebookresearch/hydra)
-   **核心模型与微调**: [`transformers`](https://github.com/huggingface/transformers), [`peft`](https://github.com/huggingface/peft)

## 工作流

整个实验流程被划分为四个核心阶段，每个阶段都有对应的执行脚本。

### 阶段一：选择模型预热 (Stage 1: Selector Pre-warming)

-   **目标**: 对一个小型的MoE模型进行全参数微调，特别是训练其Router权重，使其能够根据数据质量将数据路由到不同的专家。
-   **启动命令**:
    ```bash
    bash ./scripts/run_stage_1.sh
    ```

### 阶段二：数据选择 (Stage 2: Data Selection)

-   **目标**: 使用在阶段一中预热好的MoE模型，对大规模无标签数据进行推理。通过分析Router的激活权重，为每条数据计算一个质量分数，并筛选出分数最高的顶级数据。
-   **启动命令**:
    ```bash
    bash ./scripts/run_stage_2.sh
    ```

### 阶段三：目标模型微调 (Stage 3: Target Model Finetuning)

-   **目标**: 将阶段二筛选出的高质量数据子集用于微调一个更大、更强的目标模型。此阶段采用高效的LoRA（Low-Rank Adaptation）方法进行微调。
-   **启动命令**:
    ```bash
    bash ./scripts/run_stage_3.sh
    ```

### 阶段四：模型评估 (Stage 4: Evaluation)

-   **目标**: 使用业界标准的 `lm-eval` 框架，全面评估在筛选数据上微调后的目标模型的性能，并与基线模型进行对比，以验证数据选择策略的有效性。
-   **启动命令**:
    ```bash
    bash ./scripts/run_stage_4.sh
    ```

## 注意事项

### 评估阶段

为了确保 `lm-eval` 能够顺利运行，特别是针对 MMLU 数据集，强烈建议提前手动下载所需的数据文件。否则，评估脚本可能会因网络问题或文件校验失败而出错。

请执行以下命令下载 MMLU 数据集：
```bash
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download hails/mmlu_no_train --repo-type dataset 
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download cais/mmlu --repo-type dataset
```

如果您的环境无法直接访问 Hugging Face Hub，您可能还需要手动修改 `hails/mmlu_no_train` 的数据加载脚本，将其中的下载地址替换为可用的镜像地址。

## 待办事项

1.  **新增“垃圾桶专家”及其初始化策略**：目前阶段一仅微调了MoE的路由权重。未来的计划是实现动态增加“垃圾桶专家”的维度。当模型激活Top-K个专家时，将对应新增K个“垃圾桶专家”。这些新专家的权重将通过正态分布（均值为0，方差为0.02）进行初始化，类似于在词表中新增token的做法。

2.  **“垃圾桶”专家的行为特点**：“垃圾桶”专家在模型前向传播时不进行实际的复杂计算，而是直接输出一个符合维度形状的全零向量。这样设计的目的是为了在模型输出中引入一种“负向激励”，从而训练路由权重学会识别并避免将有价值的数据分配给这些无效的“垃圾桶”专家。
