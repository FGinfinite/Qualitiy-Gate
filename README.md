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

## 进行事项

1.  **新增“垃圾桶专家”及其初始化策略**：目前阶段一仅微调了MoE的路由权重。未来的计划是实现动态增加“垃圾桶专家”的维度。当模型激活Top-K个专家时，将对应新增K个“垃圾桶专家”。这些新专家的权重将通过正态分布（均值为0，方差为0.02）进行初始化，类似于在词表中新增token的做法。

2.  **“垃圾桶”专家的行为特点**：“垃圾桶”专家在模型前向传播时不进行实际的复杂计算，而是直接输出一个符合维度形状的全零向量。这样设计的目的是为了在模型输出中引入一种“负向激励”，从而训练路由权重学会识别并避免将有价值的数据分配给这些无效的“垃圾桶”专家。


## Custom Constraint Loss Design

为了引导Router（路由器）学习区分高质量和低质量数据，我们引入了一种特殊的约束损失函数。该损失函数的核心思想是：对于一个给定的输入，我们期望Router的Top-K专家选择概率之和（`ratio`）接近于1，而“垃圾桶”专家的选择概率之和（`1 - ratio`）接近于0。

这种机制通过一个定制的损失函数来实现，该函数受到Beta分布的启发，旨在将 `ratio` 值推向1。

### 核心参数

约束损失的设计主要由以下两个超参数控制：

*   `constraint_loss_weight` (`w_constraint`): 这个参数是约束损失在总损失中的权重。它决定了我们对“数据质量筛选”任务的重视程度。
    *   **作用**: 调整 `w_constraint` 可以平衡模型的两个目标：一是标准的交叉熵损失（`L_ce`），关注于预测下一个词的准确性；二是我们定义的约束损失（`L_constraint`），关注于路由的正确性。
    *   **直观理解**: `w_constraint` 越高，模型就越倾向于将数据清晰地分类到“好”或“坏”的类别中，即使这可能轻微影响其语言建模的性能。

*   `trash_can_loss_beta` (`β`): 这个参数控制约束损失函数对于“垃圾桶”专家激活的惩罚力度。
    *   **作用**: `β` 值决定了损失函数在 `ratio` 接近0（即“垃圾桶”专家被激活）时的梯度大小。一个较大的 `β` 值意味着对选择“垃圾桶”专家的行为施加更强的惩罚。
    *   **直观理解**: 如果将 `ratio` 想象成一个滑块，`β` 就是一个弹簧，当滑块试图滑向0时，`β` 越大的弹簧会用越大的力将其推回1。

### 计算公式

约束损失 `L_constraint` 的计算方式如下：

`L_constraint = -((α - 1) * log(ratio) + (β - 1) * log(1 - ratio))`

其中：
*   `ratio` 是Top-K专家的选择概率之和。
*   `α` (`trash_can_loss_alpha`) 和 `β` (`trash_can_loss_beta`) 是控制损失函数形态的参数。在我们的设计中，我们将 `α` 固定为1，从而简化公式并专注于 `β` 的影响。当 `α=1` 时，第一项 `(α - 1) * log(ratio)` 为0，公式简化为对 `log(1 - ratio)` 的惩罚。
*   `log` 是自然对数。

总损失 `L_total` 的计算公式为：

`L_total = L_ce + w_constraint * L_constraint`

*   `L_ce` 是标准的交叉熵损失。
*   `w_constraint` 是 `constraint_loss_weight` 参数。

### 协同工作机制

`w_constraint` 和 `β` 协同工作，共同塑造了Router的行为：

1.  **`β` 定义了“什么是错误”**: 它通过对 `1 - ratio` 的惩罚，明确了将数据路由到“垃圾桶”专家是模型需要避免的错误行为。
2.  **`w_constraint` 决定了“犯错的代价”**: 它将这个“错误”的严重性量化，并将其整合到模型的总学习目标中。

### 直观类比

*   想象一个分类任务，`β` 就像是定义了类别边界的清晰度。`β` 越大，边界越明确，模型越不能容忍模棱两可的分类。
*   `w_constraint` 则是这个分类任务在整个项目中的“重要性”或“优先级”。`w_constraint` 越高，意味着“正确分类”比其他任务（如语言建模）更重要。

### “垃圾桶”专家权重初始化策略

在模型的实现过程中，我们对新增的“垃圾桶”专家的路由权重初始化策略进行了一次重要的迭代。

最初，我们借用了模型自身配置文件 (`config.json`) 中的 `initializer_range` 参数（在 `allenai/OLMoE-1B-7B-0924` 模型中其值为 `0.02`）作为新权重初始化的标准差。这是一个快速且合理的起点，因为它遵循了模型原始设计者（AllenAI）为新层（如分类头）设定的初始化标准。

然而，为了增强代码的**可读性**和**可控性**，我们根据反馈进行了重构。我们将这个初始化过程从依赖一个通用的、可能在多处共享的 `initializer_range` 参数，转变为由两个在我们的训练配置文件 `configs/stage_1_pretrain.yaml` 中明确定义的、专门的超参数来控制：

*   `trash_can_init_mean`: 初始化正态分布的均值。
*   `trash_can_init_std`: 初始化正态分布的标准差。

这一改变使得我们对“垃圾桶”专家这一核心机制的控制更加**显式**和**精确**，避免了对模型内部配置的隐式依赖，从而提升了我们代码的模块化程度和长期可维护性。
