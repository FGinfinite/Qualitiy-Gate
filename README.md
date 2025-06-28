# 1. 预训练MoE

```bash
CUDA_VISIBLE_DEVICES=0,1  ./scripts/run_stage_1.sh
```

# 2. 模型推理与数据选择

```bash
CUDA_VISIBLE_DEVICES=0  ./scripts/run_stage_2.sh
```

# 3. 模型微调

```bash
CUDA_VISIBLE_DEVICES=0,1  ./scripts/run_stage_3.sh
```

# 4. 模型评估

```bash
CUDA_VISIBLE_DEVICES=0  ./scripts/run_stage_4.sh
```

# 注意事项

## 评估阶段

如果不提前下载以下mmlu数据集，那么lm_eval很有可能会出错。

```bash
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download hails/mmlu_no_train --repo-type dataset 
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download cais/mmlu --repo-type dataset
```

若本机无法访问HF，则还需要手动修改hails/mmlu_no_train的脚本，将其域名改为镜像网站。
