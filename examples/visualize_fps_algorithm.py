#!/usr/bin/env python3
"""
最远点采样（FPS）算法可视化脚本

最远点采样（Farthest Point Sampling）是一种贪心算法，用于从大量样本中选择具有最大多样性的子集。
在Select-MoE项目中，FPS用于在计算出样本间Wasserstein距离后，选择最多样化的数据样本。

算法流程：
1. 随机选择一个初始点
2. 对于每个剩余的候选点，计算它到已选点集的最小距离
3. 选择最小距离最大的点加入已选集合
4. 重复步骤2-3直到选够指定数量的点

这确保了选出的每个点都尽可能远离已选点，从而获得最大的多样性。
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import seaborn as sns
from scipy.stats import wasserstein_distance
from src.stages.selection import load_router_data, farthest_point_sampling
import argparse


plt.rcParams["font.sans-serif"] = ["Maple Mono NF CN"]


def compute_distance_matrix(moe_logits, max_samples=50):
    """计算样本间的Wasserstein距离矩阵"""
    n_samples = min(len(moe_logits), max_samples)
    selected_indices = np.linspace(0, len(moe_logits) - 1, n_samples, dtype=int)

    moe_subset = moe_logits[selected_indices]  # [n_samples, L, E]
    distance_matrix = np.zeros((n_samples, n_samples))

    print(f"计算{n_samples}个样本的距离矩阵...")

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            # 转换为概率分布
            probs_i = torch.softmax(moe_subset[i].float(), dim=-1)
            probs_j = torch.softmax(moe_subset[j].float(), dim=-1)

            # 计算总Wasserstein距离
            total_dist = 0
            for layer_idx in range(probs_i.shape[0]):
                prob1 = probs_i[layer_idx].numpy()
                prob2 = probs_j[layer_idx].numpy()
                expert_indices = np.arange(len(prob1))
                layer_dist = wasserstein_distance(expert_indices, expert_indices, prob1, prob2)
                total_dist += layer_dist

            distance_matrix[i, j] = total_dist
            distance_matrix[j, i] = total_dist

    return distance_matrix, selected_indices


def fps_step_by_step(distance_matrix, n_select, seed=42):
    """逐步执行FPS算法，记录每一步的状态"""
    np.random.seed(seed)
    n_total = distance_matrix.shape[0]

    if n_select >= n_total:
        return list(range(n_total)), []

    # 记录每一步的状态
    steps = []

    # 1. 随机选择初始点
    initial_point = np.random.randint(0, n_total)
    selected_indices = [initial_point]

    # 记录初始状态
    min_distances = np.full(n_total, np.inf)
    min_distances[initial_point] = 0

    steps.append(
        {
            "step": 0,
            "selected_indices": selected_indices.copy(),
            "min_distances": min_distances.copy(),
            "new_point": initial_point,
            "max_min_distance": 0,
            "candidates": [],
            "description": f"初始化：随机选择起始点 {initial_point}",
        }
    )

    # 2. 贪心选择剩余点
    for step in range(1, n_select):
        candidates_info = []
        max_min_distance = -1.0
        best_candidate = -1

        # 对每个未选择的点，计算到已选点集的最小距离
        for candidate in range(n_total):
            if candidate in selected_indices:
                continue

            # 计算candidate到已选点集的最小距离
            min_distance_to_selected = min(
                distance_matrix[candidate, selected_idx] for selected_idx in selected_indices
            )

            # 更新全局最小距离记录
            min_distances[candidate] = min(min_distances[candidate], min_distance_to_selected)

            candidates_info.append({"index": candidate, "min_distance": min_distance_to_selected, "is_best": False})

            # 选择最小距离最大的点
            if min_distance_to_selected > max_min_distance:
                max_min_distance = min_distance_to_selected
                best_candidate = candidate

        # 标记最佳候选点
        for info in candidates_info:
            if info["index"] == best_candidate:
                info["is_best"] = True

        selected_indices.append(best_candidate)
        min_distances[best_candidate] = 0

        steps.append(
            {
                "step": step,
                "selected_indices": selected_indices.copy(),
                "min_distances": min_distances.copy(),
                "new_point": best_candidate,
                "max_min_distance": max_min_distance,
                "candidates": candidates_info.copy(),
                "description": f"第{step}步：选择点{best_candidate}，最小距离={max_min_distance:.4f}",
            }
        )

    return selected_indices, steps


def create_2d_projection(distance_matrix, method="mds"):
    """将高维距离矩阵投影到2D空间用于可视化"""
    from sklearn.manifold import MDS
    from sklearn.decomposition import PCA

    if method == "mds":
        # 使用多维缩放（MDS）
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, max_iter=3000, eps=1e-9)
        coords_2d = mds.fit_transform(distance_matrix)
    else:
        # 使用PCA（需要先转换距离矩阵）
        # 使用双中心化技术将距离矩阵转换为内积矩阵
        n = distance_matrix.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        K = -0.5 * H @ (distance_matrix**2) @ H
        pca = PCA(n_components=2, random_state=42)
        coords_2d = pca.fit_transform(K)

    return coords_2d


def visualize_fps_static(distance_matrix, selected_indices, steps, coords_2d, sample_ids=None, save_path=None):
    """创建FPS算法的静态可视化"""
    n_total = distance_matrix.shape[0]
    n_select = len(selected_indices)

    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # 1. 距离矩阵热力图
    ax1 = axes[0]
    im1 = ax1.imshow(distance_matrix, cmap="viridis", aspect="auto")
    ax1.set_title("样本间距离矩阵", fontsize=12)
    ax1.set_xlabel("样本索引")
    ax1.set_ylabel("样本索引")

    # 标记选中的样本
    for i, idx in enumerate(selected_indices):
        ax1.axhline(idx, color="red", alpha=0.7, linewidth=2)
        ax1.axvline(idx, color="red", alpha=0.7, linewidth=2)

    plt.colorbar(im1, ax=ax1, label="Wasserstein距离")

    # 2. 2D投影图显示选择过程
    ax2 = axes[1]

    # 绘制所有点
    scatter = ax2.scatter(coords_2d[:, 0], coords_2d[:, 1], c="lightgray", s=100, alpha=0.6, label="未选择")

    # 按选择顺序绘制已选点
    colors = plt.cm.rainbow(np.linspace(0, 1, n_select))
    for i, idx in enumerate(selected_indices):
        ax2.scatter(
            coords_2d[idx, 0],
            coords_2d[idx, 1],
            c=[colors[i]],
            s=150,
            marker="*",
            edgecolors="black",
            linewidth=2,
            label=f"第{i}个选择",
        )

        # 标注选择顺序
        ax2.annotate(
            f"{i}",
            (coords_2d[idx, 0], coords_2d[idx, 1]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=12,
            fontweight="bold",
            color="white",
        )

    ax2.set_title("FPS选择结果（2D投影）", fontsize=12)
    ax2.set_xlabel("第一主成分")
    ax2.set_ylabel("第二主成分")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # 3. 选择步骤的最小距离变化
    ax3 = axes[2]
    step_numbers = [s["step"] for s in steps[1:]]  # 跳过初始步骤
    max_min_distances = [s["max_min_distance"] for s in steps[1:]]

    ax3.plot(step_numbers, max_min_distances, "o-", linewidth=2, markersize=8)
    ax3.set_title("每步选择的最小距离", fontsize=12)
    ax3.set_xlabel("选择步骤")
    ax3.set_ylabel("最大最小距离")
    ax3.grid(True, alpha=0.3)

    # 4. 最终选择的样本分布（距离直方图）
    ax4 = axes[3]

    # 计算选中样本间的距离
    selected_distances = []
    for i, idx1 in enumerate(selected_indices):
        for j, idx2 in enumerate(selected_indices[i + 1 :], i + 1):
            selected_distances.append(distance_matrix[idx1, idx2])

    # 计算所有样本间的距离作为对比
    all_distances = []
    for i in range(n_total):
        for j in range(i + 1, n_total):
            all_distances.append(distance_matrix[i, j])

    ax4.hist(all_distances, bins=30, alpha=0.5, label="所有样本对", density=True, color="lightblue")
    ax4.hist(selected_distances, bins=15, alpha=0.7, label="选中样本对", density=True, color="red")
    ax4.set_title("距离分布对比", fontsize=12)
    ax4.set_xlabel("Wasserstein距离")
    ax4.set_ylabel("密度")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. 算法步骤详情
    ax5 = axes[4]
    ax5.axis("off")

    # 显示算法步骤信息
    step_text = "FPS算法步骤：\n\n"
    for i, step in enumerate(steps[: min(8, len(steps))]):  # 只显示前8步
        step_text += f"{step['description']}\n"

    if len(steps) > 8:
        step_text += f"...\n共{len(steps)}步"

    ax5.text(
        0.05,
        0.95,
        step_text,
        transform=ax5.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
    )

    # 6. 多样性指标
    ax6 = axes[5]

    # 计算多样性指标
    avg_all_distance = np.mean(all_distances)
    avg_selected_distance = np.mean(selected_distances) if selected_distances else 0
    min_selected_distance = min(selected_distances) if selected_distances else 0
    max_selected_distance = max(selected_distances) if selected_distances else 0

    diversity_ratio = avg_selected_distance / avg_all_distance if avg_all_distance > 0 else 0

    metrics_text = f"""多样性指标：

所有样本平均距离: {avg_all_distance:.4f}
选中样本平均距离: {avg_selected_distance:.4f}
选中样本距离范围: {min_selected_distance:.4f} - {max_selected_distance:.4f}

多样性提升比例: {diversity_ratio:.2f}x
选择比例: {n_select}/{n_total} ({100 * n_select / n_total:.1f}%)

算法特点：
• 贪心策略确保每次选择都最大化多样性
• 选中的样本在空间中尽可能分散
• 时间复杂度: O(n²k), n=总样本数, k=选择数
"""

    ax6.text(
        0.05,
        0.95,
        metrics_text,
        transform=ax6.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )
    ax6.axis("off")

    plt.suptitle(f"最远点采样(FPS)算法可视化 - 从{n_total}个样本中选择{n_select}个", fontsize=16, y=0.98)
    plt.tight_layout()

    plt.rcParams["font.sans-serif"] = ["Maple Mono NF CN"]

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ 已保存FPS静态可视化: {save_path}")

    return fig


def create_fps_animation(distance_matrix, steps, coords_2d, save_path=None):
    """创建FPS算法的动画演示"""
    n_total = distance_matrix.shape[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 设置ax1 - 2D投影
    ax1.set_xlim(coords_2d[:, 0].min() - 0.1, coords_2d[:, 0].max() + 0.1)
    ax1.set_ylim(coords_2d[:, 1].min() - 0.1, coords_2d[:, 1].max() + 0.1)
    ax1.set_title("FPS选择过程动画")
    ax1.set_xlabel("第一主成分")
    ax1.set_ylabel("第二主成分")

    # 设置ax2 - 最小距离图
    max_steps = len(steps) - 1
    max_distance = max(s["max_min_distance"] for s in steps[1:])
    ax2.set_xlim(0, max_steps)
    ax2.set_ylim(0, max_distance * 1.1)
    ax2.set_title("选择过程中的最小距离变化")
    ax2.set_xlabel("选择步骤")
    ax2.set_ylabel("最大最小距离")
    ax2.grid(True, alpha=0.3)

    # 初始化绘图元素
    unselected_scatter = ax1.scatter([], [], c="lightgray", s=100, alpha=0.6)
    selected_scatter = ax1.scatter([], [], s=150, marker="*", edgecolors="black", linewidth=2)
    candidate_scatter = ax1.scatter([], [], c="orange", s=120, marker="o", alpha=0.8)
    best_candidate_scatter = ax1.scatter([], [], c="red", s=200, marker="o", alpha=1.0)

    (distance_line,) = ax2.plot([], [], "o-", linewidth=2, markersize=8, color="blue")
    current_point = ax2.scatter([], [], c="red", s=100, marker="o", zorder=5)

    text_box = ax1.text(
        0.02,
        0.98,
        "",
        transform=ax1.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
    )

    def animate(frame):
        if frame >= len(steps):
            return

        step = steps[frame]
        selected_indices = step["selected_indices"]

        # 更新左图 - 2D投影
        # 未选择的点
        unselected_mask = np.ones(n_total, dtype=bool)
        unselected_mask[selected_indices] = False
        unselected_coords = coords_2d[unselected_mask]

        # 更新散点图数据
        unselected_scatter.set_offsets(unselected_coords)

        # 已选择的点
        if selected_indices:
            selected_coords = coords_2d[selected_indices]
            colors = plt.cm.rainbow(np.linspace(0, 1, len(selected_indices)))
            selected_scatter.set_offsets(selected_coords)
            selected_scatter.set_color(colors)

        # 候选点（除了第0步）
        if frame > 0 and "candidates" in step:
            candidate_indices = [c["index"] for c in step["candidates"] if not c["is_best"]]
            best_candidate_idx = [c["index"] for c in step["candidates"] if c["is_best"]]

            if candidate_indices:
                candidate_coords = coords_2d[candidate_indices]
                candidate_scatter.set_offsets(candidate_coords)
            else:
                candidate_scatter.set_offsets(np.empty((0, 2)))

            if best_candidate_idx:
                best_coords = coords_2d[best_candidate_idx]
                best_candidate_scatter.set_offsets(best_coords)
            else:
                best_candidate_scatter.set_offsets(np.empty((0, 2)))
        else:
            candidate_scatter.set_offsets(np.empty((0, 2)))
            best_candidate_scatter.set_offsets(np.empty((0, 2)))

        # 更新右图 - 距离变化
        if frame > 0:
            step_numbers = [s["step"] for s in steps[1 : frame + 1]]
            distances = [s["max_min_distance"] for s in steps[1 : frame + 1]]
            distance_line.set_data(step_numbers, distances)

            # 当前点
            current_point.set_offsets([(step["step"], step["max_min_distance"])])
        else:
            distance_line.set_data([], [])
            current_point.set_offsets(np.empty((0, 2)))

        # 更新文本
        text_box.set_text(f"步骤 {frame}: {step['description']}")

        return (
            unselected_scatter,
            selected_scatter,
            candidate_scatter,
            best_candidate_scatter,
            distance_line,
            current_point,
            text_box,
        )

    anim = animation.FuncAnimation(fig, animate, frames=len(steps), interval=1500, blit=False, repeat=True)

    plt.tight_layout()

    if save_path:
        # 保存为GIF
        print("正在生成FPS算法动画，请稍候...")
        anim.save(save_path, writer="pillow", fps=0.67, dpi=150)
        print(f"✓ 已保存FPS动画: {save_path}")

    return fig, anim


def main():
    parser = argparse.ArgumentParser(description="FPS算法可视化演示")
    parser.add_argument("router_data_path", help="路由数据文件路径(.pt格式)")
    parser.add_argument("--max-samples", type=int, default=30, help="使用的最大样本数 (默认: 30)")
    parser.add_argument("--n-select", type=int, default=8, help="选择的样本数 (默认: 8)")
    parser.add_argument("--save-plots", action="store_true", help="保存图片到文件")
    parser.add_argument("--save-animation", action="store_true", help="保存动画到文件")
    parser.add_argument("--output-dir", default="./outputs/visual_figs/fps_plots", help="图片保存目录")
    parser.add_argument("--projection-method", choices=["mds", "pca"], default="mds", help="2D投影方法 (默认: mds)")

    args = parser.parse_args()

    # 创建输出目录
    if args.save_plots or args.save_animation:
        os.makedirs(args.output_dir, exist_ok=True)

    print("最远点采样(FPS)算法可视化演示")
    print("=" * 70)

    # 1. 加载数据
    print(f"加载路由数据: {args.router_data_path}")
    router_data = load_router_data(args.router_data_path)

    moe_logits = router_data["moe_logits"]  # [N, L, E]
    sample_ids = router_data["sample_ids"]
    dataset_name = router_data["dataset_name"]

    print(f"数据集: {dataset_name}")
    print(f"总样本数: {len(sample_ids)}")
    print(f"将使用前{args.max_samples}个样本进行演示")
    print()

    # 2. 计算距离矩阵
    distance_matrix, selected_indices_for_demo = compute_distance_matrix(moe_logits, args.max_samples)
    n_total = distance_matrix.shape[0]
    n_select = min(args.n_select, n_total - 1)

    print(f"距离矩阵大小: {distance_matrix.shape}")
    print(f"将从{n_total}个样本中选择{n_select}个")
    print()

    # 3. 执行FPS算法
    print("执行FPS算法...")
    fps_selected, fps_steps = fps_step_by_step(distance_matrix, n_select)

    print(f"✓ FPS选择完成，选中样本索引: {fps_selected}")
    print(f"✓ 记录了{len(fps_steps)}个步骤")
    print()

    # 4. 创建2D投影
    print(f"创建2D投影 (方法: {args.projection_method})...")
    coords_2d = create_2d_projection(distance_matrix, args.projection_method)
    print("✓ 2D投影完成")
    print()

    # 5. 创建静态可视化
    print("创建静态可视化...")
    save_path_static = os.path.join(args.output_dir, "fps_static_visualization.png") if args.save_plots else None
    fig_static = visualize_fps_static(distance_matrix, fps_selected, fps_steps, coords_2d, sample_ids, save_path_static)
    plt.show()

    # 6. 创建动画（可选）
    if args.save_animation:
        print("创建FPS算法动画...")
        save_path_anim = os.path.join(args.output_dir, "fps_animation.gif")
        fig_anim, anim = create_fps_animation(distance_matrix, fps_steps, coords_2d, save_path_anim)
        # 显示第一帧
        plt.show()

    # 7. 验证与selection.py中实现的一致性
    print("验证与selection.py中FPS实现的一致性...")
    reference_selected = farthest_point_sampling(distance_matrix, n_select, seed=42)

    if fps_selected == reference_selected:
        print("✓ 与selection.py中的FPS实现结果一致")
    else:
        print("⚠ 与selection.py中的FPS实现结果不一致")
        print(f"  当前实现: {fps_selected}")
        print(f"  参考实现: {reference_selected}")

    print()
    print("=" * 70)
    print("可视化完成！")
    print()
    print("FPS算法关键概念：")
    print("1. 贪心策略：每次选择距离已选点集最远的点")
    print("2. 多样性最大化：确保选出的样本在特征空间中尽可能分散")
    print("3. 时间复杂度：O(n²k)，其中n是总样本数，k是选择数量")
    print("4. 应用场景：在计算Wasserstein距离后，选择最多样化的数据子集")
    print("5. 优点：简单高效，能有效避免选择相似的样本")


if __name__ == "__main__":
    main()
