import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


class GraphTransformerLayer(nn.Module):
    """
    单层 Graph Transformer，带边类型偏置（Graphormer 风格）。

    标准 Transformer 对所有节点对做全局注意力，但忽略了图的连接结构。
    本层在注意力分数中加入可学习的边偏置项，使"相邻四面体之间"的注意力
    与"不相邻四面体之间"的注意力有所区别：

        score(i→j) = (Q_i · K_j) / sqrt(d_k) + bias[edge_type(i,j)]

    其中 edge_type_idx:
        0     → 无边（i,j 不相邻）
        1-4   → 面类型 0-3（+1 偏移，避免与"无边"混淆）

    边偏置初始化为 0，训练过程中自动学习不同面类型对注意力的影响。
    图规模（~100 节点）足够小，可以做 O(N²) 的全局注意力。
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, n_edge_types: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads
        self.d_model = d_model

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)

        # edge_bias: Embedding(n_edge_types+1, n_heads)
        # 每种边类型对每个注意力头学习一个独立的标量偏置
        # idx 0 = 无边, idx 1..n_edge_types = 面类型 0..n_edge_types-1
        self.edge_bias = nn.Embedding(n_edge_types + 1, n_heads)
        nn.init.zeros_(self.edge_bias.weight)   # 初始无先验，从零开始学习

        self.ff    = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                   nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, edge_idx_mat: torch.Tensor) -> torch.Tensor:
        """
        x:            (N, d_model)  — 当前节点嵌入
        edge_idx_mat: (N, N) int64  — 0=无边, 1-4=边类型+1

        返回: (N, d_model)  — 更新后的节点嵌入
        """
        N = x.size(0)
        H, d_k = self.n_heads, self.d_k

        # 投影为多头 Q/K/V，形状 (N, H, d_k)
        Q = self.W_q(x).view(N, H, d_k)
        K = self.W_k(x).view(N, H, d_k)
        V = self.W_v(x).view(N, H, d_k)

        # 注意力分数 (H, N, N)：Q_i 与 K_j 的点积，缩放后加边偏置
        # einsum 'ihd,jhd->hij'：对每个头 h，计算 N×N 的点积矩阵
        scores = torch.einsum('ihd,jhd->hij', Q, K) / (d_k ** 0.5)

        # 边偏置：edge_idx_mat → (N,N,H)，permute 后与 scores 形状对齐
        bias   = self.edge_bias(edge_idx_mat).permute(2, 0, 1)   # (H, N, N)
        scores = scores + bias

        attn = torch.softmax(scores, dim=-1)   # (H, N, N)，对源节点 j 归一化

        # 加权求和：einsum 'hij,jhd->ihd'：每个头 h 中，节点 i 聚合所有 j 的值
        # reshape 后合并多头 → (N, d_model)
        out = torch.einsum('hij,jhd->ihd', attn, V).reshape(N, self.d_model)
        out = self.W_o(out)

        # Pre-Norm 残差连接（先 Add 后 LayerNorm）
        x = self.norm1(x + out)
        x = self.norm2(x + self.ff(x))
        return x


class GraphEncoder(nn.Module):
    """
    将四面体图编码为全局向量。

    输入: graph_obs（节点类型数组 + 边列表）
    输出: 全局嵌入 (d_model,)

    读出策略：mean pooling + max pooling 拼接后线性映射。
      - mean pooling 捕捉图的"平均特征"（全局组成）
      - max pooling 捕捉"最突出的局部特征"（极端结构）
      两者互补，比单一 pooling 更具表达能力。

    空图（step 0）：直接返回零向量，代表"空集嵌入到原点"。
    此时 CandidateScorer 仅依赖 new_tet_type 嵌入来选择初始四面体，
    相当于让网络学习一个关于初始构型的先验偏好。
    """
    def __init__(self, cfg: Config):
        super().__init__()
        d = cfg.d_model

        # 节点嵌入：5 种四面体类型（SSSS/SSSL/SSLL/SLLL/LLLL）
        self.node_emb = nn.Embedding(cfg.n_tet_types, d)

        self.layers = nn.ModuleList([
            GraphTransformerLayer(d, cfg.n_heads, cfg.d_ff, cfg.n_face_types)
            for _ in range(cfg.n_layers)
        ])

        # 读出层：将 mean+max 拼接的 2d 向量压缩到 d
        self.readout = nn.Linear(2 * d, d)

    def forward(self, graph_obs: dict, device: torch.device) -> torch.Tensor:
        """
        graph_obs: 来自 env._get_graph_obs() 的字典
        返回: (d_model,) 全局嵌入；空图返回零向量
        """
        n_nodes = graph_obs['n_nodes']
        d       = self.node_emb.embedding_dim

        # Step 0：空图，返回零向量作为"空集原点"
        if n_nodes == 0:
            return torch.zeros(d, device=device)

        node_types = torch.from_numpy(graph_obs['node_types']).to(device)  # (N,)
        x          = self.node_emb(node_types)                             # (N, d)

        # 构建稠密边类型矩阵 (N, N)
        # 0=无边（默认），1-4=face_type+1（+1 偏移以区分"无边"和"面类型 0"）
        edge_idx_mat = torch.zeros(n_nodes, n_nodes, dtype=torch.long, device=device)
        if graph_obs['n_edges'] > 0:
            src    = torch.from_numpy(graph_obs['edge_src']).to(device)
            dst    = torch.from_numpy(graph_obs['edge_dst']).to(device)
            etypes = torch.from_numpy(graph_obs['edge_types']).to(device) + 1
            edge_idx_mat[src, dst] = etypes   # 已是双向边，src/dst 均已包含

        for layer in self.layers:
            x = layer(x, edge_idx_mat)   # 逐层更新节点嵌入 (N, d)

        # mean + max pooling → readout → (d,)
        global_emb = self.readout(
            torch.cat([x.mean(0), x.max(0).values], dim=-1)
        )
        return global_emb


class CandidateScorer(nn.Module):
    """
    对所有候选打分，输出未归一化 logit，由调用方做 softmax 或 log_softmax。

    每个候选由两个整数描述（均为图结构信息，无坐标）：
        face_type:    开放面的类型（0-3），step 0 时为 4（"无面"特殊 token）
        new_tet_type: 放入新粒子后形成的四面体类型（0-4）
                      = face_type + new_ptype（0 or 1）

    打分公式：
        score = MLP( global_emb || face_emb || new_tet_emb )

    其中 || 表示拼接。三段向量分别贡献：
        global_emb   → 当前图的整体结构信息（"现在是什么状态"）
        face_emb     → 候选所在开放面的局部类型（"往哪里放"）
        new_tet_emb  → 放入后形成的新四面体类型（"放什么"）
    """
    def __init__(self, cfg: Config):
        super().__init__()
        d = cfg.d_model

        # face_type 嵌入：size = n_face_types+1（含 step 0 的"无面" token，idx=4）
        self.face_emb    = nn.Embedding(cfg.n_face_types + 1, d)
        # new_tet_type 嵌入：size = n_tet_types（0-4，共 5 种四面体）
        self.new_tet_emb = nn.Embedding(cfg.n_tet_types, d)

        # 两层 MLP：3d → d_hidden → 1（打分）
        self.mlp = nn.Sequential(
            nn.Linear(3 * d, cfg.d_hidden), nn.GELU(),
            nn.Linear(cfg.d_hidden, 1)
        )

    def forward(self, global_emb: torch.Tensor,
                face_types: torch.Tensor,
                new_tet_types: torch.Tensor) -> torch.Tensor:
        """
        global_emb:    (d_model,)      — 来自 GraphEncoder 的全局嵌入
        face_types:    (N_cands,) int  — 每个候选的开放面类型
        new_tet_types: (N_cands,) int  — 每个候选形成的新四面体类型
        → scores:      (N_cands,)      — 未归一化 logit
        """
        n      = face_types.size(0)
        # 将全局嵌入广播到每个候选
        g_emb  = global_emb.unsqueeze(0).expand(n, -1)    # (N_cands, d)
        f_emb  = self.face_emb(face_types)                 # (N_cands, d)
        nt_emb = self.new_tet_emb(new_tet_types)           # (N_cands, d)

        # 拼接三段特征后过 MLP，squeeze 去掉最后的 1 维
        return self.mlp(torch.cat([g_emb, f_emb, nt_emb], dim=-1)).squeeze(-1)


class PackingPolicy(nn.Module):
    """
    完整策略网络：GraphEncoder + CandidateScorer。

    前向流程：
        graph_obs → GraphEncoder → global_emb (d,)
        cand_obs  ─────────────────────────────────┐
                                                   ↓
                                          CandidateScorer → scores (N_cands,)

    训练时逐样本调用 forward_single，梯度通过各自的计算图回传到共享参数。
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg     = cfg
        self.encoder = GraphEncoder(cfg)
        self.scorer  = CandidateScorer(cfg)

    def forward_single(self, graph_obs: dict, cand_obs: dict,
                       device: torch.device) -> torch.Tensor:
        """
        单步前向推理（采集和训练均调用此接口）。
        返回 (N_cands,) 未归一化分数，调用方负责 softmax/log_softmax。
        """
        global_emb    = self.encoder(graph_obs, device)
        face_types    = torch.from_numpy(cand_obs['face_types']).to(device)
        new_tet_types = torch.from_numpy(cand_obs['new_tet_types']).to(device)
        return self.scorer(global_emb, face_types, new_tet_types)
