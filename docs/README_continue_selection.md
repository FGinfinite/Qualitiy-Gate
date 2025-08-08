# GPU显存不足时的数据选择恢复指南

## 问题描述

当运行数据选择脚本时，如果在距离矩阵计算阶段出现GPU显存不足错误，现在不会丢失宝贵的模型推理结果。系统会在模型推理完成后立即保存router_data，然后释放GPU内存再进行距离计算。

## 性能优化

现在整个数据选择流程使用GPU加速：
- **距离矩阵计算**：GPU加速的Wasserstein距离计算
- **FPS算法**：完全GPU向量化的最远点采样算法
- **预期加速比**：对于大规模数据（>10k样本），整体FPS性能提升50-200倍

## 自动恢复流程

1. **模型推理完成**：系统立即保存所有router_data到 `router_data/` 目录
2. **释放GPU内存**：删除模型实例，清理GPU缓存
3. **尝试距离计算**：如果成功，继续正常流程
4. **失败时提供指导**：显存不足时提供继续执行的具体命令

## 手动继续执行

如果距离计算失败，系统会提供类似以下的恢复命令：

```bash
# 基本命令
python scripts/continue_selection.py \
    --router_data_dir outputs/stage_2_selection/2025-08-08/17-53-20/router_data \
    --output_path outputs/stage_2_selection/2025-08-08/17-53-20/selected_data.jsonl \
    --selection_percentage 0.05

# 带两阶段选择的命令
python scripts/continue_selection.py \
    --router_data_dir outputs/stage_2_selection/2025-08-08/17-53-20/router_data \
    --output_path outputs/stage_2_selection/2025-08-08/17-53-20/selected_data.jsonl \
    --selection_percentage 0.05 \
    --importance_selection_percentage 0.1

# 禁用多样性选择（仅使用质量分数）
python scripts/continue_selection.py \
    --router_data_dir outputs/stage_2_selection/2025-08-08/17-53-20/router_data \
    --output_path outputs/stage_2_selection/2025-08-08/17-53-20/selected_data.jsonl \
    --selection_percentage 0.05 \
    --disable_diversity

# 调整批处理大小和FPS日志间隔
python scripts/continue_selection.py \
    --router_data_dir outputs/stage_2_selection/2025-08-08/17-53-20/router_data \
    --output_path outputs/stage_2_selection/2025-08-08/17-53-20/selected_data.jsonl \
    --selection_percentage 0.05 \
    --distance_batch_size 300 \
    --fps_log_interval 50
```

## 参数说明

- `--router_data_dir`: 包含已保存router_data文件的目录
- `--output_path`: 选择结果的输出文件路径  
- `--selection_percentage`: 最终数据选择比例
- `--importance_selection_percentage`: 两阶段选择的第一阶段比例（可选）
- `--disable_diversity`: 禁用多样性选择，仅使用质量分数排序
- `--distance_batch_size`: GPU批处理大小，显存不足时可调小
- `--fps_log_interval`: FPS进度日志输出间隔（默认: 100）
- `--device`: 指定计算设备（cuda/cpu/auto）
- `--verbose`: 启用详细日志输出

## 配置文件设置

在 `configs/stage_2_selection.yaml` 中可配置GPU FPS相关参数：

```yaml
distance_computation:
  distance_batch_size: 1000      # 距离计算批处理大小
  fps_log_interval: 100          # FPS日志输出间隔
  fps_memory_efficient: true     # 启用内存高效模式
```

## 优势

1. **数据安全**：模型推理结果不会因距离计算失败而丢失
2. **性能显著提升**：GPU加速的FPS算法，特别适合大规模数据
3. **内存优化**：推理和计算分离，避免内存冲突  
4. **灵活调参**：可多次尝试不同的批处理大小和参数
5. **完整功能**：保持所有原有的选择策略和功能