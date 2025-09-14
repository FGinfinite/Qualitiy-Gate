[根目录](../../CLAUDE.md) > [src](../) > **clustering**

# 聚类算法模块 - GPU加速智能数据选择

## 变更记录 (Changelog)

**2025-09-14 22:43:47** - 模块文档初始化
- 创建GPU加速聚类模块文档
- 详细说明K-Means + Elbow方法
- 添加多GPU并行处理指南

## 模块职责

聚类算法模块负责实现基于GPU加速的数据聚类和选择策略，主要功能：
- **GPU K-Means聚类**：使用CUDA加速的K-Means算法
- **自动K值选择**：基于Elbow方法的最优聚类数确定
- **多GPU并行处理**：支持跨多GPU的并行K值搜索
- **轮询选择策略**：从各簇中均匀选择高质量数据
- **余弦距离度量**：基于MoE logits的语义聚类

## 入口与启动

### 主要入口点
- `src/clustering/cluster_selection.py` - 聚类选择主协调器
- `src/clustering/kmeans_clustering.py` - GPU K-Means核心实现
- `src/clustering/parallel_kmeans.py` - 多GPU并行K-Means
- `src/clustering/gpu_metrics.py` - GPU轮廓系数计算

### 基本使用
```python
from src.clustering.cluster_selection import ClusterBasedSelection

# 初始化聚类选择器
selector = ClusterBasedSelection(
    device=torch.device("cuda:0"),
    random_state=42,
    debug_print=True
)

# 执行聚类选择
selected_data = selector.select_data_by_clustering(
    scored_data=scored_data,
    all_logits_by_dataset=logits_data,
    target_count=1000,
    clustering_method="kmeans",
    clustering_params={"auto_k": True, "k_range": [10, 100]}
)
```

## 对外接口

### ClusterBasedSelection 主类
```python
class ClusterBasedSelection:
    """基于聚类的数据选择器"""

    def select_data_by_clustering(
        self,
        scored_data: List[Dict],                    # 评分数据列表
        all_logits_by_dataset: Dict[str, List[torch.Tensor]],  # MoE logits数据
        target_count: int,                          # 目标选择数量
        clustering_method: str = "kmeans",          # 聚类方法
        clustering_params: Dict = None              # 聚类参数
    ) -> List[Dict]:
        """
        使用聚类算法选择数据

        Returns:
            List[Dict]: 选择的数据列表，按质量分数排序
        """
```

### GPUKMeansClustering K-Means实现
```python
class GPUKMeansClustering:
    """GPU加速的K-Means聚类器"""

    def find_optimal_k_parallel(
        self,
        features: torch.Tensor,              # 特征矩阵 [N, D]
        k_range: List[int] = [10, 100],      # K值搜索范围
        parallel_processes: int = 4,          # 并行进程数
        gpu_allocation_strategy: str = "round_robin"  # GPU分配策略
    ) -> Tuple[int, List[float], Dict]:
        """
        多GPU并行搜索最优K值

        Returns:
            Tuple[int, List[float], Dict]: 最优K值, 惯性列表, 详细结果
        """

    def kmeans_cosine_gpu(
        self,
        X: torch.Tensor,                     # 输入数据 [N, D]
        k: int,                             # 聚类数
        max_iters: int = 300,               # 最大迭代次数
        tol: float = 1e-4                   # 收敛阈值
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        GPU K-Means聚类（余弦距离）

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 聚类标签, 聚类中心
        """
```

## 关键依赖与配置

### 依赖库
- `torch>=2.6.0` - CUDA张量运算
- `cuml-cu12>=25.8.0` - RAPIDS GPU机器学习库
- `scikit-learn>=1.3.0` - CPU聚类算法备选
- `numpy>=1.24.0` - 数值计算

### 配置参数

#### K-Means聚类参数
```yaml
clustering_method: "kmeans"
clustering_params:
  # 自动K值选择
  auto_k: true                      # 是否自动选择K值
  k_range: [10, 100]               # K值搜索范围

  # 手动K值设置（当auto_k=false时使用）
  k: 50                            # 手动指定的K值

  # 聚类算法参数
  max_iters: 300                   # 最大迭代次数
  tol: 1e-4                        # 收敛阈值

  # 多GPU并行配置
  enable_parallel_kmeans: false     # 是否启用多GPU并行
  parallel_processes: 4             # 并行进程数
  gpu_allocation_strategy: "round_robin"  # GPU分配策略："round_robin" 或 "balanced"
```

## 数据模型

### 聚类数据流
```
MoE Logits [N, num_experts]
    ↓
特征归一化 (L2 normalization)
    ↓
GPU K-Means聚类
    ├─ 自动K值搜索（Elbow方法）
    └─ 多GPU并行处理
    ↓
聚类结果 [N,] + 质量分数
    ↓
轮询选择策略
    └─ 从每个簇选择最高质量数据
    ↓
最终选择数据
```

### Elbow方法流程
```python
# 1. K值范围搜索
k_candidates = range(k_min, k_max + 1)

# 2. 并行计算每个K的惯性
inertias = []
for k in k_candidates:
    labels, centers = kmeans_cosine_gpu(features, k)
    inertia = compute_cosine_inertia(features, labels, centers)
    inertias.append(inertia)

# 3. 计算Elbow点
elbow_point = find_elbow_point(k_candidates, inertias)
optimal_k = k_candidates[elbow_point]
```

### 轮询选择算法
```python
# 从每个簇轮流选择最高质量的数据
cluster_data = defaultdict(list)
for data, cluster_id in zip(scored_data, cluster_labels):
    cluster_data[cluster_id].append(data)

# 按质量分数排序每个簇的数据
for cluster_id in cluster_data:
    cluster_data[cluster_id].sort(key=lambda x: x["quality_score"], reverse=True)

# 轮询选择
selected_data = []
while len(selected_data) < target_count:
    for cluster_id in cluster_data:
        if cluster_data[cluster_id] and len(selected_data) < target_count:
            selected_data.append(cluster_data[cluster_id].pop(0))
```

## 测试与质量

### 验证脚本
```bash
# GPU轮廓系数验证
python scripts/validate_gpu_silhouette.py

# 聚类算法性能测试
python -c "
from src.clustering.kmeans_clustering import GPUKMeansClustering
import torch
clusterer = GPUKMeansClustering(torch.device('cuda:0'))
print('GPU聚类器初始化成功')
"
```

### 性能基准
- **GPU vs CPU**: GPU加速通常比CPU快10-50倍
- **内存使用**: 大约需要特征矩阵大小的2-3倍GPU内存
- **并行效率**: 多GPU并行在K值搜索上可获得近线性加速

## 常见问题 (FAQ)

**Q: 如何选择合适的K值范围？**
A: 建议根据数据规模设置：小数据集[5,20]，中等数据集[10,50]，大数据集[20,100]。

**Q: 多GPU并行处理如何配置？**
A: 设置 `enable_parallel_kmeans=true` 和 `parallel_processes=N`，其中N通常为GPU数量的2-4倍。

**Q: 聚类算法内存不足怎么办？**
A: 减少 `parallel_processes` 数量，或使用更少的GPU设备。

**Q: 如何调试聚类过程？**
A: 启用 `debug_print=true` 获取详细的聚类日志和性能统计。

## 相关文件清单

### 核心实现文件
- `cluster_selection.py` - 聚类选择协调器（200+ 行）
- `kmeans_clustering.py` - K-Means核心算法（500+ 行）
- `parallel_kmeans.py` - 多GPU并行处理（300+ 行）
- `gpu_metrics.py` - GPU轮廓系数计算（200+ 行）
- `__init__.py` - 模块初始化

### 配置文件
- `../../configs/stage_2_selection.yaml` - 选择阶段配置
- `../../configs/continue_selection.yaml` - 独立选择配置
- `../../configs/batch_selection.yaml` - 批量选择配置

### 验证工具
- `../../scripts/validate_gpu_silhouette.py` - GPU算法验证
- `../../examples/comprehensive_analysis.py` - 聚类结果分析

### 相关文档
- `../../docs/gpu_fps_fixes_summary.md` - GPU性能优化总结