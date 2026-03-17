# GNN 教程：从消息传递到图分类

本教程结合粒子堆积项目的具体场景，从数学原理出发解释经典 GNN 的工作方式。

---

## 1. 问题背景

我们的任务是：给定一个**四面体接触图**，预测下一步应该在哪个开放面放一个什么类型的新粒子。

图的结构：
- **节点** = 四面体，类型由大球数量决定，共 5 种（SSSS/SSSL/SSLL/SLLL/LLLL）
- **边** = 两个四面体共享一个三角面，类型由面上大球数量决定，共 4 种（SSS/SSL/SLL/LLL）

输入均为整数，例如：

```
node_types = [2, 1, 3, 2]       # 4 个四面体的类型
edge_src   = [0, 1, 1, 2]       # 边的起点
edge_dst   = [1, 0, 2, 1]       # 边的终点（双向）
edge_types = [1, 1, 2, 2]       # 每条边对应的面类型
```

---

## 2. Embedding：整数 → 向量

GNN 的第一步是把整数索引转换为连续向量，这通过可学习的查找表完成。

### 数学定义

$$\mathbf{h}_i = E_{\text{node}}[\text{node\_type}_i] \in \mathbb{R}^d$$

$$\mathbf{e}_{ij} = E_{\text{edge}}[\text{edge\_type}_{ij}] \in \mathbb{R}^d$$

其中 $E_{\text{node}} \in \mathbb{R}^{5 \times d}$，$E_{\text{edge}} \in \mathbb{R}^{4 \times d}$ 是两个可学习的权重矩阵。

### 直觉

```
E_node（5×d 矩阵）：

       d=128
     ┌──────────────┐
 0   │  0.3  -0.1 … │  ← SSSS 的向量表示
 1   │ -0.2   0.5 … │  ← SSSL
 2   │  0.1   0.8 … │  ← SSLL       输入整数 2，取出第 2 行
 3   │  0.7  -0.3 … │  ← SLLL
 4   │ -0.5   0.2 … │  ← LLLL
     └──────────────┘
```

整数只是查表的索引，进入网络计算的始终是向量。这些向量通过反向传播与其他参数一起训练。

---

## 3. 消息传递：一层 GNN 的数学

经典 GNN 的每一层由三个步骤组成。

### 3.1 生成消息（Message）

节点 $j$ 沿边 $(i,j)$ 向节点 $i$ 发送的消息，由节点 $j$ 的嵌入和边类型嵌入**拼接**后经 MLP 生成：

$$\mathbf{m}_{ij} = \sigma\!\left( W_{\text{msg}} \cdot \text{cat}[\mathbf{h}_j,\, \mathbf{e}_{ij}] + \mathbf{b} \right)$$

展开矩阵乘法（令 $W_{\text{msg}} = [W_A \mid W_B]$，$W_A, W_B \in \mathbb{R}^{d \times d}$）：

$$\mathbf{m}_{ij} = \sigma\!\left( W_A \mathbf{h}_j + W_B \mathbf{e}_{ij} + \mathbf{b} \right)$$

**关键**：$W_A$ 和 $W_B$ 是完全独立的权重矩阵。节点内容和边类型分别经过不同的线性变换后叠加，再过非线性激活函数。这意味着：

> 同一个邻居节点，通过不同类型的面传来的消息完全不同。

对比若使用加法而非拼接：

$$\mathbf{m}_{ij}^{\text{加法}} = \sigma\!\left( W(\mathbf{h}_j + \mathbf{e}_{ij}) \right)$$

此时 $\mathbf{h}_j$ 和 $\mathbf{e}_{ij}$ 被强制共享同一个权重矩阵，表达能力退化。

### 3.2 聚合（Aggregate）

节点 $i$ 收集所有邻居发来的消息，取均值：

$$\mathbf{M}_i = \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} \mathbf{m}_{ij}$$

其中 $\mathcal{N}(i)$ 是节点 $i$ 的邻居集合。常见聚合方式：

| 方式 | 公式 | 特点 |
|------|------|------|
| mean | $\frac{1}{\|N\|}\sum m_{ij}$ | 稳定，不受度数影响 |
| sum  | $\sum m_{ij}$ | 能感知邻居数量 |
| max  | $\max_j m_{ij}$ | 保留最显著特征 |

### 3.3 更新（Update）

将节点自身的旧状态 $\mathbf{h}_i$ 与聚合结果 $\mathbf{M}_i$ 拼接，经 MLP 得到新状态：

$$\mathbf{h}_i' = \text{LayerNorm}\!\left( \text{MLP}\!\left( \text{cat}[\mathbf{h}_i,\, \mathbf{M}_i] \right) \right)$$

同样是拼接而非相加，保证自身信息和邻居信息的独立变换。

---

## 4. 多层 GNN 的感受野

每叠加一层 GNN，每个节点能"看到"的范围扩大一跳：

```
1 层：只看直接邻居（1 跳）

        [ j1 ]──[ i ]──[ j2 ]
                 ↑
            只聚合 j1, j2

2 层：能看到邻居的邻居（2 跳）

   [k1]──[ j1 ]──[ i ]──[ j2 ]──[k2]
           ↑              ↑
      第2层时 i 已经包含了 k1, k2 的信息

3 层：3 跳，以此类推
```

对于我们 ~100 个节点的图，3 层 GNN 感受野已经能覆盖大部分图结构。

---

## 5. 全局读出（Readout）

消息传递完成后，所有节点的嵌入 $\{\mathbf{h}_i'\}$ 需要汇总为一个**全局向量**，用于打分。

$$\mathbf{g} = W_{\text{out}} \cdot \text{cat}\!\left[ \frac{1}{N}\sum_i \mathbf{h}_i',\; \max_i \mathbf{h}_i' \right]$$

mean pooling 捕捉图的平均特征，max pooling 捕捉最突出的局部特征，两者互补。

---

## 6. 候选打分

得到全局向量 $\mathbf{g}$ 后，对每个候选点计算得分：

$$\text{score}_c = \text{MLP}\!\left( \text{cat}[\mathbf{g},\; \mathbf{e}_{\text{face}}^c,\; \mathbf{e}_{\text{tet}}^c] \right)$$

其中：
- $\mathbf{e}_{\text{face}}^c$：候选 $c$ 所在开放面的类型嵌入（"往哪里放"）
- $\mathbf{e}_{\text{tet}}^c$：候选 $c$ 形成的新四面体类型嵌入（"放什么"）

最终通过 softmax 转为概率分布，采样得到动作。

---

## 7. 完整数据流

```
整数输入
  node_types (N,)  ──→  node_emb  ──→  x: (N, d)
  edge_types (E,)  ──→  edge_emb  ──→  e: (E, d)
                                            │
                          ┌─────────────────┘
                          ▼
              ┌──────────────────────────┐
              │   GNN Layer × 3          │
              │                          │
              │  m_ij = MLP(cat[h_j, e_ij])   消息生成
              │  M_i  = mean{ m_ij }          聚合
              │  h_i' = MLP(cat[h_i, M_i])    更新
              └──────────┬───────────────┘
                         │  x: (N, d)
                         ▼
              mean(x) + max(x) → Linear → g: (d,)
                                               │
              face_types    → face_emb ────────┤
              new_tet_types → tet_emb  ────────┤
                                               ▼
                              MLP(cat[g, f, t]) → scores (C,)
                                               │
                                          softmax
                                               │
                                       action_idx → 放置新粒子
```

---

## 8. GNN vs Graph Transformer

| | 经典 GNN | Graph Transformer |
|---|---|---|
| 感受野 | k 层 = k 跳 | 1 层 = 全图 |
| 边特征 | 拼接进消息内容 | 作为注意力偏置 |
| 孤立节点 | 第 1 层无法更新 | 通过全局注意力仍可更新 |
| 计算复杂度 | O(E·d) | O(N²·d) |
| 适合场景 | 图大而稀疏 | 图小（~100节点） |

对于本项目，两者均可，经典 GNN 实现更简单，Graph Transformer 在早期图稀疏时略有优势。
