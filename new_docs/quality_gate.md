下面是**“只用门控 g + token 困惑度权重”的核心方法**（前向即可、一次排序、步骤清晰）。

---

# 0. 记号与目标

* 数据集样本数 **N**；第 (i) 个样本长度 (T_i)；模型层数 **L**。
* 预热后的模型（主干冻结）在前向时输出：

  * (G_i\in(0,1)^{L\times T_i})：第 (l) 层第 (t) 个 token 的门控 (g_{i,l,t})（Sigmoid 后）。
  * (\mathbf{ppl}_i\in\mathbb R^{T_i})：每个 token 的困惑度（(\exp(\text{NLL}))）。
* 目标：为每个样本算出**质量分数** (q_i)，按 (q) 降序取 **top-k%**。

---

# 1. 阶段一 · 预热（Warm-up）

* 冻结主干，仅训练门控参数；正常交叉熵训练若干步/epoch。
* 若使用质量正则 (L_{\text{quality}})：**先样本内（layer&token）平均，再样本间平均**，避免长序列偏置。
  [
  \overline g^{(x)}=\frac{1}{L}\sum_{l=1}^{L}\frac{1}{T_x}\sum_{t=1}^{T_x}g_{l,t}(x),\quad
  L_{\text{quality}}=\frac{1}{B}\sum_{x\in\mathcal B}\overline g^{(x)}
  ]

---

# 2. 阶段二 · 统计收集（仅前向）

对每个样本 (i) 做一次前向，取出 (G_i) 与 (\mathbf{ppl}_i)。

**2.1 样本内 token 权重（高熵更重要）**
给定超参 (\alpha>0)（默认 1），定义
[
w_{i,t}=\frac{(\mathrm{ppl}*{i,t})^{\alpha}}{\sum*{u=1}^{T_i}(\mathrm{ppl}*{i,u})^{\alpha}+\varepsilon},\quad
\varepsilon=10^{-8}.
]
性质：(w*{i,t}\ge0)、(\sum_t w_{i,t}=1)，困惑度越高权重越大。

**2.2 按 token 加权得到逐层分数并堆叠**
[
s_{i,l}=\sum_{t=1}^{T_i} w_{i,t},g_{i,l,t},\qquad
S=\big[s_{i,l}\big]\in\mathbb R^{N\times L}.
]

> 形状关键：这里得到的是 **[N, L]** 矩阵（样本 × 层）。

---

# 3. 阶段三 · 逐层“一步映射”到统一尺度并加权

目的：把不同层的列拉到统一尺度，同时**让均值小（更“会关门”）的层权重大**。
先算每列统计量（跨样本）：
[
a_l=\min_i S_{i,l},\quad
b_l=\max_i S_{i,l},\quad
\mu_l=\frac{1}{N}\sum_{i=1}^N S_{i,l}.
]

**一步映射（把“列内 min–max 归一化” 与 “按均值的反比加权”合并）**：
[
\boxed{;
R_{i,l}=\frac{S_{i,l}-a_l}
{\big(\max(b_l-a_l,\varepsilon)\big)\cdot\big(\max(\mu_l,\varepsilon)\big)}; }
]
得到 (R\in\mathbb R^{N\times L})。
直觉：同列内，(R) 对 (S) 单调；分母里的 ((b_l-a_l)) 统一尺度，(\mu_l) 越小该层权重越大。

> 如需更稳，可把第二个因子改为 ((\mu_l+\tau))，(\tau\in[10^{-4},10^{-3}])。

---

# 4. 样本质量分数与选择

[
q_i=\frac{1}{L}\sum_{l=1}^{L} R_{i,l}\quad(\text{得到 }[N]\text{ 向量})
]
对 ({q_i}) **降序排序**，取 **top-k%** 作为高质量样本子集。

---

# 5. 伪代码（可直接实现）

```python
# 超参
alpha = 1.0
eps = 1e-8
top_ratio = 0.30  # 例：取前30%

# ===== Phase 2: collect =====
S_rows = []  # list of [L]
for x in dataset:  # 仅前向
    logits, cache = model.forward_with_gates(x, return_intermediates=True)
    G = cache.gates_sigmoid            # [L, T_i]
    ppl = compute_token_ppl(logits, x.targets)  # [T_i], ppl = exp(nll)

    w = (ppl ** alpha)
    w = w / (w.sum() + eps)            # [T_i], sum=1
    s = (G * w[None, :]).sum(axis=1)   # [L]  按token加权求和
    S_rows.append(s)

S = stack(S_rows, axis=0)              # [N, L]

# ===== Phase 3: one-shot column mapping =====
a  = S.min(axis=0)                     # [L]
b  = S.max(axis=0)                     # [L]
mu = S.mean(axis=0)                    # [L]

den = np.maximum(b - a, eps) * np.maximum(mu, eps)  # [L]
R = (S - a[None, :]) / den[None, :]    # [N, L]

# ===== Score & pick =====
q = R.mean(axis=1)                     # [N]
k = int(np.ceil(top_ratio * len(q)))
idx = np.argsort(-q)                   # 降序
selected_ids = [dataset.ids[j] for j in idx[:k]]
```

---

# 6. 超参与稳定性

* (\alpha)：强调高熵 token（默认 1；>1 更偏重高熵；<1 更均衡）。
* (\varepsilon=1\mathrm{e}{-8})：统一的数值下界，防分母为 0。
* （可选）(\tau)：把 (\max(\mu_l,\varepsilon)) 换成 ((\mu_l+\tau)) 可更稳。
* 复杂度：仅前向；内存主要是暂存 (G) 以聚合成 (S)。

---

# 7. 快速检查清单

* [ ] 预热仅更新门控；若用 (L_{\text{quality}})，遵循“样本内→样本间”聚合。
* [ ] 收集阶段只前向，得到 **(G_i:[L,T_i]), (\mathbf{ppl}_i:[T_i])**。
* [ ] 形成 **(S:[N,L])**；按“一步映射”得到 **(R:[N,L])**。
* [ ] (q=\text{mean}_L(R))；按 (q) 降序取 **top-k%**。

这样流程就**前向一次、步骤清晰、零反传**，可以直接落地。
