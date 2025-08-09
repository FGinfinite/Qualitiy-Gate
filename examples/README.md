# Select-MoE 数据选择可视化脚本

本目录包含三个可视化脚本，用于帮助理解Select-MoE数据选择过程中的关键概念：逐层二级路由余弦相似度计算和FPS（最远点采样）算法。

## 脚本概述

### 1. `visualize_cosine_similarity.py`
**余弦相似度计算可视化**

展示如何计算两个样本之间基于逐层二级路由的余弦相似度，包括：
- 样本间MoE路由概率分布对比
- 逐层余弦相似度可视化
- 逐层距离计算过程
- GPU vs CPU计算性能对比
- 样本间距离热力图

### 2. `visualize_fps_algorithm.py`  
**FPS算法流程可视化**

演示最远点采样（Farthest Point Sampling）算法的工作原理：
- 逐步展示FPS贪心选择过程
- 2D投影下的样本分布和选择结果
- 算法执行动画（可选）
- 多样性指标分析
- 与参考实现的一致性验证

### 3. `comprehensive_analysis.py`
**Select-MoE综合分析**

提供完整的数据选择管道分析：
- 质量门输出分析
- MoE路由模式分析  
- 样本间距离计算
- 不同选择策略对比（质量 vs 多样性 vs 随机）
- 综合性能评估

## 使用方法

### 环境准备

确保已安装所需依赖：
```bash
pip install matplotlib seaborn scikit-learn pandas torch
```

### 基本用法

#### 1. 余弦相似度可视化
```bash
# 基本用法
python examples/visualize_cosine_similarity.py outputs/stage_2_selection/2025-08-07/17-09-01/router_data/oasst1_router_data.pt

# 指定对比的样本和层
python examples/visualize_cosine_similarity.py outputs/stage_2_selection/2025-08-07/17-09-01/router_data/oasst1_router_data.pt \
    --sample1-idx 0 --sample2-idx 5 --layer-idx 2

# 保存图片
python examples/visualize_cosine_similarity.py outputs/stage_2_selection/2025-08-07/17-09-01/router_data/oasst1_router_data.pt \
    --save-plots --output-dir ./cosine_plots
```

#### 2. FPS算法可视化
```bash
# 基本用法
python examples/visualize_fps_algorithm.py outputs/stage_2_selection/2025-08-07/17-09-01/router_data/oasst1_router_data.pt

# 自定义参数
python examples/visualize_fps_algorithm.py outputs/stage_2_selection/2025-08-07/17-09-01/router_data/oasst1_router_data.pt \
    --max-samples 20 --n-select 6 --projection-method mds

# 保存静态图和动画
python examples/visualize_fps_algorithm.py outputs/stage_2_selection/2025-08-07/17-09-01/router_data/oasst1_router_data.pt \
    --save-plots --save-animation --output-dir ./fps_plots
```

#### 3. 综合分析
```bash
# 基本用法
python examples/comprehensive_analysis.py outputs/stage_2_selection/2025-08-07/17-09-01/router_data/oasst1_router_data.pt

# 自定义选择参数
python examples/comprehensive_analysis.py outputs/stage_2_selection/2025-08-07/17-09-01/router_data/oasst1_router_data.pt \
    --max-samples 30 --selection-ratio 0.3

# 保存分析结果
python examples/comprehensive_analysis.py outputs/stage_2_selection/2025-08-07/17-09-01/router_data/oasst1_router_data.pt \
    --save-plots --output-dir ./comprehensive_analysis
```

## 输出说明

### 可视化图表

1. **概率分布对比图**: 展示两个样本在某一层的专家选择概率分布差异
2. **距离热力图**: 显示多个样本间的余弦相似度距离矩阵
3. **FPS选择过程**: 2D投影空间中的样本分布和逐步选择过程
4. **多样性分析**: 选择结果的质量和多样性指标对比
5. **综合仪表板**: 包含质量门、MoE路由、距离计算和选择策略的全面分析

### 关键指标

- **余弦相似度距离**: 基于逐层二级路由向量余弦相似度的距离度量
- **质量分数**: 基于质量门输出的样本质量评估  
- **路由熵**: 样本在专家选择上的多样性指标
- **负载平衡度**: MoE专家使用的均衡程度
- **多样性提升比例**: FPS选择相对于随机选择的多样性改进

## 算法原理

### 逐层余弦相似度
- 计算两个向量的方向相似性，范围为[-1, 1]
- 在Select-MoE中用于衡量样本间二级路由概率分布的相似性
- 逐层计算后求和：对每层的MoE路由向量计算余弦相似度，然后相加
- 计算公式: `距离 = 1 - Σ(cosine_similarity(layer_i_prob_1, layer_i_prob_2))`
- 支持层权重扩展：未来可为不同层分配不同权重

### FPS算法
- 贪心策略：每次选择距离已选集合最远的点
- 确保选出的样本在特征空间中最大化分散
- 时间复杂度: O(n²k)，其中n为总样本数，k为选择数量
- 广泛用于点云处理和多样性采样

### Select-MoE管道
1. **质量评估**: 使用质量门评估样本质量
2. **路由分析**: 分析MoE专家选择模式
3. **距离计算**: 计算样本间基于逐层余弦相似度的距离
4. **多样性选择**: 使用FPS算法选择最多样化样本

## 常见参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--max-samples` | 用于分析的最大样本数 | 30-50 |
| `--n-select` | FPS算法选择的样本数 | 8 |
| `--selection-ratio` | 选择比例 | 0.2 (20%) |
| `--sample1-idx/sample2-idx` | 对比样本的索引 | 0, 1 |
| `--layer-idx` | 详细分析的层索引 | 0 |
| `--projection-method` | 2D投影方法 | 'mds' |
| `--save-plots` | 是否保存图片 | False |
| `--save-animation` | 是否保存动画 | False |

## 预期输出示例

运行脚本后，您将看到：

1. **控制台输出**: 详细的计算过程和统计信息
2. **可视化图表**: 交互式matplotlib图表
3. **保存文件** (如启用): PNG图片和GIF动画
4. **性能分析**: GPU vs CPU计算对比
5. **一致性检查**: 与参考实现的结果对比

## 故障排除

### 常见问题

1. **CUDA内存不足**: 减少`--max-samples`参数
2. **依赖包缺失**: 确保安装了所有必需包
3. **文件路径错误**: 检查路由数据文件路径是否正确
4. **动画保存失败**: 确保安装了pillow包用于GIF生成

### 性能优化

- 使用GPU加速余弦相似度距离计算
- 限制分析样本数量以提高运行速度
- 选择合适的投影方法(MDS vs PCA)

## 扩展用法

这些脚本可以作为基础模板，扩展用于：
- 不同数据集的对比分析
- 超参数敏感性分析  
- 新选择算法的性能评估
- 路由模式的深入研究

通过这些可视化工具，您可以更好地理解Select-MoE的数据选择机制，并为模型优化提供洞察。