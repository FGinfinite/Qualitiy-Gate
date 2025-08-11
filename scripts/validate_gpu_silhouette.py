#!/usr/bin/env python3
"""
极小数据集验证GPU轮廓系数实现的正确性

此脚本使用极小的数据集（10个样本，3维特征，2个聚类）来验证
GPU实现的轮廓系数与sklearn CPU版本的数值一致性。

运行方法:
python scripts/validate_gpu_silhouette.py
"""

import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from sklearn.metrics import silhouette_score

from src.clustering.gpu_metrics import gpu_silhouette_score_cosine


def create_tiny_test_data():
    """创建极小的测试数据集: 10个样本，3维特征，2个聚类"""
    # 设置随机种子确保可重现
    torch.manual_seed(42)
    np.random.seed(42)

    # 创建两个簇的中心
    center1 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    center2 = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)

    # 在每个中心周围生成样本
    cluster1_samples = []
    cluster2_samples = []

    # 簇1: 5个样本
    for i in range(5):
        noise = torch.randn(3) * 0.1  # 小噪声
        sample = center1 + noise
        cluster1_samples.append(sample)

    # 簇2: 5个样本
    for i in range(5):
        noise = torch.randn(3) * 0.1  # 小噪声
        sample = center2 + noise
        cluster2_samples.append(sample)

    # 合并数据
    data = torch.stack(cluster1_samples + cluster2_samples)  # [10, 3]
    labels = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # [10]

    # 标准化数据（余弦距离要求）
    data_normalized = torch.nn.functional.normalize(data, p=2, dim=1)

    return data_normalized, labels


def validate_gpu_silhouette():
    """验证GPU轮廓系数实现的正确性"""
    print("=" * 60)
    print("GPU轮廓系数验证脚本")
    print("=" * 60)

    # 1. 创建极小测试数据
    print("1. 创建测试数据...")
    data, labels = create_tiny_test_data()
    print(f"   数据形状: {data.shape}")
    print(f"   标签分布: {torch.bincount(labels).tolist()}")
    print(f"   数据类型: {data.dtype}")

    # 2. 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   使用设备: {device}")
    data = data.to(device)
    labels = labels.to(device)

    # 3. 计算GPU版本轮廓系数
    print("\n2. 计算GPU版本轮廓系数...")
    try:
        gpu_score = gpu_silhouette_score_cosine(data, labels)
        print(f"   GPU轮廓系数: {gpu_score:.10f}")
    except Exception as e:
        print(f"   ❌ GPU计算失败: {e}")
        return False

    # 4. 计算CPU版本轮廓系数（sklearn）
    print("\n3. 计算sklearn CPU版本轮廓系数...")
    try:
        data_cpu = data.cpu().numpy()
        labels_cpu = labels.cpu().numpy()
        cpu_score = silhouette_score(data_cpu, labels_cpu, metric="cosine")
        print(f"   CPU轮廓系数: {cpu_score:.10f}")
    except Exception as e:
        print(f"   ❌ CPU计算失败: {e}")
        return False

    # 5. 比较结果
    print("\n4. 验证结果...")
    absolute_error = abs(gpu_score - cpu_score)
    relative_error = absolute_error / abs(cpu_score) if abs(cpu_score) > 1e-10 else 0.0

    print(f"   绝对误差: {absolute_error:.2e}")
    print(f"   相对误差: {relative_error:.2e}")

    # 设定误差阈值
    tolerance = 1e-6

    if absolute_error < tolerance:
        print(f"   ✅ 验证通过! 误差 {absolute_error:.2e} < {tolerance:.2e}")
        return True
    else:
        print(f"   ❌ 验证失败! 误差 {absolute_error:.2e} >= {tolerance:.2e}")
        return False


def test_edge_cases():
    """测试边界情况"""
    print("\n" + "=" * 60)
    print("边界情况测试")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 测试1: 只有一个聚类
    print("\n1. 测试单聚类情况...")
    data = torch.randn(5, 3, device=device)
    data = torch.nn.functional.normalize(data, p=2, dim=1)
    labels = torch.zeros(5, dtype=torch.long, device=device)  # 所有样本都在簇0

    gpu_score = gpu_silhouette_score_cosine(data, labels)
    print(f"   单聚类轮廓系数: {gpu_score:.6f} (应该为0.0)")
    assert abs(gpu_score - 0.0) < 1e-6, "单聚类轮廓系数应该为0"

    # 测试2: 每个样本都是独立聚类
    print("\n2. 测试每个样本独立聚类...")
    data = torch.randn(3, 3, device=device)
    data = torch.nn.functional.normalize(data, p=2, dim=1)
    labels = torch.arange(3, device=device)  # [0, 1, 2]

    gpu_score = gpu_silhouette_score_cosine(data, labels)
    print(f"   独立聚类轮廓系数: {gpu_score:.6f} (应该为0.0)")
    assert abs(gpu_score - 0.0) < 1e-6, "独立聚类轮廓系数应该为0"

    print("   ✅ 边界情况测试通过!")


def main():
    """主函数"""
    print("开始GPU轮廓系数验证...\n")

    try:
        # 主要验证
        success = validate_gpu_silhouette()

        if success:
            # 边界情况测试
            test_edge_cases()
            print("\n" + "=" * 60)
            print("🎉 所有验证通过! GPU轮廓系数实现正确")
            print("=" * 60)
            return 0
        else:
            print("\n" + "=" * 60)
            print("❌ 验证失败! GPU实现有问题")
            print("=" * 60)
            return 1

    except Exception as e:
        print(f"\n❌ 验证过程出现异常: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
