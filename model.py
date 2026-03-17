import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


class GraphTransformerLayer(nn.Module):
    """
    单层 Graph Transformer，带边类型偏置。

    注意力分数：score(i,j) = (Q_i · K_j) / sqrt(d_k) + bias(edge_type(i,j))
    edge_type_idx: 0 = 无边, 1-4 = 面类型 0-3（+1 偏移）
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

        # 每条边对每个注意力头贡献一个标量偏置
        # idx 0 = 无边, idx 1..n_edge_types = 面类型 0..n_edge_types-1
        self.edge_bias = nn.Embedding(n_edge_types + 1, n_heads)
        nn.init.zeros_(self.edge_bias.weight)   # 初始化为 0：无先验偏置

        self.ff    = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                   nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, edge_idx_mat: torch.Tensor) -> torch.Tensor:
        """
        x:            (N, d_model)
        edge_idx_mat: (N, N) int — 0=无边, 1-4=边类型+1
        """
        N = x.size(0)
        H, d_k = self.n_heads, self.d_k

        Q = self.W_q(x).view(N, H, d_k)   # (N, H, d_k)
        K = self.W_k(x).view(N, H, d_k)
        V = self.W_v(x).view(N, H, d_k)

        # 注意力分数 (H, N, N)
        scores = torch.einsum('ihd,jhd->hij', Q, K) / (d_k ** 0.5)

        # 边偏置 (N, N, H) → (H, N, N)
        bias   = self.edge_bias(edge_idx_mat).permute(2, 0, 1)
        scores = scores + bias

        attn = torch.softmax(scores, dim=-1)   # (H, N, N)

        # 聚合值向量 (N, H, d_k) → (N, d_model)
        out = torch.einsum('hij,jhd->ihd', attn, V).reshape(N, self.d_model)
        out = self.W_o(out)

        x = self.norm1(x + out)
        x = self.norm2(x + self.ff(x))
        return x


class GraphEncoder(nn.Module):
    """
    将四面体图编码为全局向量。

    输入: 图的节点类型和边类型
    输出: 全局嵌入 (d_model,) —— mean pooling + max pooling 拼接后线性映射
    空图: 输出零向量（step 0 的初始状态）
    """
    def __init__(self, cfg: Config):
        super().__init__()
        d = cfg.d_model

        # 节点嵌入：5 种四面体类型
        self.node_emb = nn.Embedding(cfg.n_tet_types, d)

        # Graph Transformer 层
        self.layers = nn.ModuleList([
            GraphTransformerLayer(d, cfg.n_heads, cfg.d_ff, cfg.n_face_types)
            for _ in range(cfg.n_layers)
        ])

        # 读出：mean + max → d
        self.readout = nn.Linear(2 * d, d)

    def forward(self, graph_obs: dict, device: torch.device) -> torch.Tensor:
        """
        返回 (d_model,) 全局嵌入。空图返回零向量。
        """
        n_nodes = graph_obs['n_nodes']
        d       = self.node_emb.embedding_dim

        if n_nodes == 0:
            return torch.zeros(d, device=device)

        node_types = torch.from_numpy(graph_obs['node_types']).to(device)  # (N,)
        x          = self.node_emb(node_types)                             # (N, d)

        # 构建边类型矩阵 (N, N)：0=无边, 1-4=face_type+1
        edge_idx_mat = torch.zeros(n_nodes, n_nodes, dtype=torch.long, device=device)
        if graph_obs['n_edges'] > 0:
            src       = torch.from_numpy(graph_obs['edge_src']).to(device)
            dst       = torch.from_numpy(graph_obs['edge_dst']).to(device)
            etypes    = torch.from_numpy(graph_obs['edge_types']).to(device) + 1  # +1 偏移
            edge_idx_mat[src, dst] = etypes

        for layer in self.layers:
            x = layer(x, edge_idx_mat)    # (N, d)

        global_emb = self.readout(
            torch.cat([x.mean(0), x.max(0).values], dim=-1)
        )                                 # (d,)
        return global_emb


class CandidateScorer(nn.Module):
    """
    对所有候选打分。

    每个候选由 (face_type, new_tet_type) 描述:
        - face_type:    0-3 = 面类型, 4 = 无面（step 0）
        - new_tet_type: 0-4 = 新四面体类型

    score = MLP(global_emb || face_emb || new_tet_emb)
    """
    def __init__(self, cfg: Config):
        super().__init__()
        d = cfg.d_model

        # face_type 嵌入：0-3=真实面, 4=无面（step 0）
        self.face_emb    = nn.Embedding(cfg.n_face_types + 1, d)
        self.new_tet_emb = nn.Embedding(cfg.n_tet_types,      d)

        self.mlp = nn.Sequential(
            nn.Linear(3 * d, cfg.d_hidden), nn.GELU(),
            nn.Linear(cfg.d_hidden, 1)
        )

    def forward(self, global_emb: torch.Tensor,
                face_types: torch.Tensor,
                new_tet_types: torch.Tensor) -> torch.Tensor:
        """
        global_emb:    (d_model,)
        face_types:    (N_cands,) int
        new_tet_types: (N_cands,) int
        → scores:      (N_cands,)
        """
        n      = face_types.size(0)
        g_emb  = global_emb.unsqueeze(0).expand(n, -1)    # (N_cands, d)
        f_emb  = self.face_emb(face_types)                 # (N_cands, d)
        nt_emb = self.new_tet_emb(new_tet_types)           # (N_cands, d)

        return self.mlp(torch.cat([g_emb, f_emb, nt_emb], dim=-1)).squeeze(-1)


class PackingPolicy(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg     = cfg
        self.encoder = GraphEncoder(cfg)
        self.scorer  = CandidateScorer(cfg)

    def forward_single(self, graph_obs: dict, cand_obs: dict,
                       device: torch.device) -> torch.Tensor:
        """
        单样本前向。返回 (N_cands,) 未归一化分数。
        """
        global_emb    = self.encoder(graph_obs, device)
        face_types    = torch.from_numpy(cand_obs['face_types']).to(device)
        new_tet_types = torch.from_numpy(cand_obs['new_tet_types']).to(device)
        return self.scorer(global_emb, face_types, new_tet_types)
