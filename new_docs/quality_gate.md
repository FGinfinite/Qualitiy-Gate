# 质量门控数据选择——核心方法（v1）

> 三阶段流程；一次项正则；**单次排序**；**零统计量**；**负曲率惩罚的非对称分母**。
> 目标：给工程直接落地用，指导实现与联调。

---

## 0. 记号与目标

* 模型共有 **L** 个门控层；样本 (x) 的长度为 (T_x)。
* 第 (l) 层门控：(u_l = g_l \odot y_l)，其中 (g_l\in(0,1))（按 token 输出），(y_l) 为门控前的特征。
* 训练损失：
  [
  L_{\text{total}} = L_{\text{lm}} + \lambda,L_{\text{quality}}
  ]
* **一次项正则**（修正的聚合定义，避免长序列偏置）：
  [
  \overline g^{(x)} = \frac1L \sum_{l=1}^L \frac1{T_x} \sum_{t=1}^{T_x} g_{l,t}(x),\qquad
  L_{\text{quality}} = \frac1B \sum_{x\in\mathcal B} \overline g^{(x)}
  ]
  （先**样本内**做 layer&token 平均，再**样本间**做 batch 平均。）

---

## 1. 阶段一 · 预热（Warm-up）

**目的**：让门控具备基本可区分性；冻结主干，**仅训练门控参数**。
**做法**：

1. 冻结 backbone；启用门控（Sigmoid/Soft gate）。
2. 用上式 (L_{\text{total}}) 正常训练若干步/epoch。
3. 优化器：AdamW/SGD 皆可；门控 lr 建议为常规微调 lr 的 **1/5～1/10**。
4. 其他约束/排序逻辑 **不在本阶段** 使用。

---

## 2. 阶段二 · 统计收集（Scoring Prep）

**输出**：对每个样本 (x) 收集三条 **按层向量**（长度 **L**），**不做层平均**，便于后续自定义分析。

* (\mathbf g(x) = [\bar g_l]*{l=1}^L)，其中 (\bar g_l = \frac1{T_x}\sum_t g*{l,t}(x))。
* (\mathbf A(x) = [\bar A_l]*{l=1}^L)，其中 (\bar A_l = \frac1{T_x}\sum_t \big(\partial L*{\text{lm}}/\partial g_{l,t}\big))。
* (\mathbf B(x) = [\bar B_l]*{l=1}^L)，其中 (\bar B_l = \frac1{T_x}\sum_t \big(y*{l,t}^\top H_{u_l}y_{l,t}\big))（HVP 实现即可）。

> 说明：**只需一次前/反传**。A 用 autograd；B 用 Hessian-Vector Product 沿方向 (y_l)。

**伪代码（高层）**：

```python
model.eval(); freeze(backbone=True)

for x in dataloader:
    out, cache = model.forward_with_gates(x, return_intermediates=True)  # cache: y_l, g_l, u_l per token
    L_lm = out.lm_loss

    # A: ∂L_lm/∂g_l  (list[L] of [T_x])
    A_tok = autograd.grad(L_lm, cache.gates, retain_graph=True)

    # B: y^T H_u y  via HVP (list[L] of [T_x])
    B_tok = hessian_vector_product(L_lm, cache.u, cache.y)

    # token-mean per layer -> vectors [L]
    g_vec = token_mean(cache.gates)      # [L]
    A_vec = token_mean(A_tok)            # [L]
    B_vec = token_mean(B_tok)            # [L]

    dump(x_id, g_vec, A_vec, B_vec)      # 不做 layer-mean；直接保存 [L]
```

---

## 3. 阶段三 · 排序与选数（Ranking & Pick）

**逐层打分 → 样本聚合 → 单次排序选前 r%**。不做两段排序；不引入额外统计量。

### 3.1 逐层 D₂ 分数（一次项正则）

[
\boxed{
\mathrm{D2}*l(x)=
\frac{\big[-(A_l(x)+\lambda)\big]*+^2}
{2\big(,\tau + \max(B_l(x)+\mu,0);+;\kappa,\max(-(B_l(x)+\mu),0),\big)}
}
]

* 分子：净驱动力（收益−固定成本）的**正部平方**。
* 分母：**非对称惩罚**分母

  * (\tau>0)：极小下界（防近零爆分）；
  * (\max(B+\mu,0))：正曲率按原值付“成本”；
  * (\kappa\max(-(B+\mu),0))：**负曲率（不可信）加重惩罚**，(\kappa>1)。

**推荐默认**：(\tau=1\mathrm{e}{-4}\sim 1\mathrm{e}{-3})，(\kappa=10)，(\mu=0)。

### 3.2 样本聚合（默认：层平均，可自定义）

[
\mathrm{D2}*{\text{sample}}(x)=\frac{1}{L}\sum*{l=1}^L \mathrm{D2}_l(x)
]

> 也可换为 `topk_mean`、按层权重平均等；本规范默认用 layer-mean，因阶段二已保存 ([L]) 向量，后续可灵活替换。

### 3.3 排序与选择

* 对全集样本计算 (\mathrm{D2}_{\text{sample}}(x))，**降序排序**；
* 选取前 **r%**（或前 **K** 条），得到高质量子集。

**伪代码（评分与选数）**：

```python
tau, kappa, mu = 1e-4, 10.0, 0.0

def D2_layerwise(A_vec, B_vec, lam):
    num = np.maximum(-(A_vec + lam), 0.0)**2                   # [L]
    den = tau + np.maximum(B_vec + mu, 0.0) + kappa*np.maximum(-(B_vec + mu), 0.0)
    return num / (2.0 * den)                                   # [L]

scores = {}
for x_id in dataset_ids:
    g_vec, A_vec, B_vec = load(x_id)                           # 全是 [L]
    d2_vec = D2_layerwise(A_vec, B_vec, λ)
    scores[x_id] = d2_vec.mean()                               # 默认 layer-mean

selected = top_percent(scores, r)                              # 单次排序，取前 r%
save_selected(selected)
```

---

## 4. 工程接口（建议）

### 4.1 阶段二输出（按样本）

```json
{
  "sample_id": "string",
  "g": [float; L],   // 每层 token-mean 的门控值
  "A": [float; L],   // 每层 token-mean 的一阶项 ∂L_lm/∂g_l
  "B": [float; L]    // 每层 token-mean 的方向二阶 y^T H y
}
```

### 4.2 阶段三输入/输出

* 输入：上述 JSONL / Parquet；参数 `{lambda, tau, kappa, mu, r}`。
* 输出：`selected_ids: List[sample_id]`（以及可选的 `score` 侧写）。

---

## 5. 落地注意事项

* **稳定性**：(\tau) 给最小曲率，已杜绝“趋零爆分”；(\kappa>1) 保证负曲率必被降权。
* **一致性**：A/B 的 token-mean 与层索引必须与门控实现一致；不同批次的统计口径保持一致。
* **可扩展**：若未来切换到**二次正则**，把分子 (A+\lambda) 改为 (A+2\lambda g)，分母加 (+2\lambda) 即可：
  (\mathrm{D2}*l=\frac{[-(A_l+2\lambda g_l)]*+^2}{2(\tau+\max(B_l+\mu+2\lambda,0)+\kappa\max(-(B_l+\mu+2\lambda),0))})。

---

## 6. 快速检查清单（上线前）

* [ ] 预热阶段仅门控参数更新；`L_quality` 采用 **样本内→样本间** 聚合。
* [ ] 阶段二产物为 **[L] 向量**（不做层平均）；数值范围检查无 NaN/Inf。
* [ ] 阶段三使用 **非对称分母**；(\tau,\kappa,\mu) 为**常数**，无批内统计。
* [ ] 单次排序选前 r% 输出；保存打分以便复现与回溯。

> 以上即为“核心方法”最终版：三阶段清晰、接口明确、代码可直接实现。
