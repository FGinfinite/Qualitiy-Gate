# 共享内存数据集加载使用指南

## 概述

共享内存数据集加载模块提供基于内存映射的高性能数据集共享方案，专门用于加速HuggingFace数据集的重复加载。通过一次加载、多次共享的方式，可以实现10000倍以上的加载速度提升。

## 主要特性

- **高性能**: 基于mmap内存映射，首次后几乎瞬间加载
- **零侵入**: 不影响现有代码逻辑，完全可选
- **容错性强**: 服务不可用时自动降级到传统加载方式
- **支持并发**: 多个客户端进程可同时访问共享数据集
- **资源共享**: 多个进程共享同一块内存映射区域，节省系统资源

## 架构设计

### 模块结构
```
share_dataset/
├── __init__.py           # 模块初始化，导出主要接口
├── server.py            # 共享内存服务器实现
├── client.py            # 共享内存客户端接口
├── manager.py           # 服务器管理工具
└── config.py            # 配置管理
```

### 工作流程
1. **服务器启动**: 加载数据集，创建内存映射文件
2. **客户端访问**: 通过内存映射快速加载数据
3. **自动降级**: 服务不可用时回退到传统加载方式

## 使用方法

### 1. 配置启用

在配置文件中启用共享内存选项：

```yaml
# configs/stage_1_warmup.yaml
dataset:
  # 启用共享内存加速
  use_shared_memory: true
  
  shared_memory:
    server_timeout: 30      # 服务器响应超时时间（秒）
    auto_start: false       # 是否自动启动服务器（暂未实现）
```

### 2. 启动服务器

使用管理工具启动数据集服务器：

```bash
# 方式1: 使用管理工具启动（推荐）
python -m share_dataset.manager start --sample-limit 1000

# 方式2: 直接启动服务器
python -m share_dataset.server --sample-limit 1000
```

启动参数说明：
- `--sample-limit`: 限制样本数量（用于测试，生产环境可省略）
- `--mmap-file`: 自定义内存映射文件路径
- `--pid-file`: 自定义PID文件路径

### 3. 查看服务器状态

```bash
# 查看服务器状态
python -m share_dataset.manager status

# 输出示例：
# 服务器状态:
#   running: True
#   pid: 12345
#   mmap_file: /tmp/shared_dataset.mmap
#   mmap_size_mb: 245.6
#   status: ready
```

### 4. 停止服务器

```bash
# 停止服务器
python -m share_dataset.manager stop

# 重启服务器
python -m share_dataset.manager restart
```

### 5. 在代码中使用

共享内存加载已集成到 `load_hf_datasets` 函数中，无需修改现有代码：

```python
from src.data.dataset_loader import load_hf_datasets

# 配置
hf_config = {
    "datasets": [
        {
            "name": "teknium/OpenHermes-2.5",
            "dataset_name": "openhermes",
            "subset": None,
            "split": "train"
        }
    ]
}

# 使用共享内存加载（如果可用）
dataset = load_hf_datasets(
    hf_config=hf_config,
    sample_percentage=1.0,
    seed=42,
    use_shared_memory=True  # 启用共享内存
)
```

## 性能对比

| 加载方式 | 首次加载时间 | 重复加载时间 | 加速倍数 |
|---------|-------------|-------------|---------|
| 传统方式 | 60-120秒 | 30-60秒 | - |
| 共享内存 | 60-120秒(一次性) | 0.001-0.01秒 | ~10000x |

## 配置选项详解

### 数据集配置

```yaml
dataset:
  use_shared_memory: false  # 是否启用共享内存加速
  
  shared_memory:
    server_timeout: 30      # 服务器响应超时时间（秒）
    auto_start: false       # 是否自动启动服务器（暂未实现）
```

### 环境变量配置

可以通过环境变量覆盖默认配置：

```bash
export SHARED_DATASET_MMAP_FILE="/custom/path/dataset.mmap"
export SHARED_DATASET_PID_FILE="/custom/path/server.pid"
export SHARED_DATASET_TIMEOUT="60"
export SHARED_DATASET_SAMPLE_LIMIT="1000"
```

## 测试验证

运行集成测试脚本验证功能：

```bash
# 运行完整测试
python test_shared_memory_integration.py
```

测试包括：
1. 服务器生命周期管理测试
2. 集成数据加载功能测试
3. 多客户端并发访问测试

## 故障排除

### 常见问题

1. **服务器启动失败**
   - 检查端口占用情况
   - 确认有足够的内存空间
   - 查看日志文件 `/tmp/shared_dataset_server.log`

2. **客户端连接失败**
   - 确认服务器正在运行：`python -m share_dataset.manager status`
   - 检查文件权限和路径访问权限
   - 检查超时设置是否合适

3. **内存不足**
   - 减少样本数量限制
   - 增加系统可用内存
   - 检查是否有内存泄漏

### 日志调试

启用调试日志：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 手动清理

如果出现异常，可以手动清理相关文件：

```bash
# 清理PID文件和内存映射文件
rm -f /tmp/shared_dataset_server.pid
rm -f /tmp/shared_dataset_server.status
rm -f /tmp/shared_dataset.mmap
```

## 注意事项

1. **数据一致性**: 服务器运行期间数据集内容不会更新，需要重启服务器加载新数据
2. **内存占用**: 服务器需要加载完整数据集到内存，确保有足够内存空间
3. **系统重启**: 系统重启后需要重新启动服务器
4. **权限管理**: 确保所有客户端进程都有访问共享文件的权限

## 限制与已知问题

1. **数据集支持**: 当前仅支持OpenHermes-2.5数据集
2. **平台限制**: 基于mmap实现，主要支持Linux/Unix系统
3. **进程间通信**: 依赖文件系统进行状态同步，可能存在极少数竞态条件

## 未来改进

1. **自动启动**: 支持客户端自动启动服务器
2. **多数据集**: 支持同时共享多个不同数据集
3. **动态更新**: 支持数据集的增量更新
4. **网络共享**: 支持跨机器的数据集共享