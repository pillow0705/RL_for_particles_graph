import numpy as np
import torch
import torch.nn as nn

from config import Config


class GNNLayer(nn.Module):
    """
    单层消息传递 GNN。

    流程：
      1. 消息生成：m_ij = MLP( cat[h_j, e_ij] )
      2. 聚合：    M_i  = mean{ m_ij | j ∈ N(i) }
      3. 更新：    h_i' = LayerNorm( h_i + MLP( cat[h_i, M_i] ) )
    """
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.upd_mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_ff), nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, edge_src, edge_dst, edge_emb):
        N   = x.size(0)
        msg = self.msg_mlp(torch.cat([x[edge_src], edge_emb], dim=-1))
        agg = torch.zeros(N, x.size(1), device=x.device)
        agg.scatter_add_(0, edge_dst.unsqueeze(-1).expand_as(msg), msg)
        degree = torch.bincount(edge_dst, minlength=N).float()
        agg    = agg / degree.clamp(min=1).unsqueeze(-1)
        h      = self.upd_mlp(torch.cat([x, agg], dim=-1))
        return self.norm(x + h)


class ParticleGNN(nn.Module):
    """
    粒子接触图的 GNN 编码器。

    输入：graph_obs 字典（节点类型 + 双向边列表）
    输出：全局嵌入向量 (d_model,)

    节点嵌入：Embedding(n_ptypes=2, d)    — 小球 / 大球
    边嵌入：  Embedding(n_edge_types=3, d) — SS / SL / LL
    读出：    mean pooling + max pooling → Linear(2d → d)
    """
    def __init__(self, cfg: Config):
        super().__init__()
        d = cfg.d_model
        self.node_emb = nn.Embedding(cfg.n_ptypes,     d)
        self.edge_emb = nn.Embedding(cfg.n_edge_types, d)
        self.layers   = nn.ModuleList([GNNLayer(d, cfg.d_ff)
                                       for _ in range(cfg.n_layers)])
        self.readout  = nn.Linear(2 * d, d)

    def forward(self, graph_obs: dict, device: torch.device) -> torch.Tensor:
        n = graph_obs['n_nodes']
        d = self.node_emb.embedding_dim

        if n == 0:
            return torch.zeros(d, device=device)

        node_types = torch.from_numpy(graph_obs['node_types']).to(device)
        x          = self.node_emb(node_types)   # (N, d)

        if graph_obs['n_edges'] > 0:
            edge_src   = torch.from_numpy(graph_obs['edge_src']).to(device)
            edge_dst   = torch.from_numpy(graph_obs['edge_dst']).to(device)
            edge_types = torch.from_numpy(graph_obs['edge_types']).to(device)
            edge_emb   = self.edge_emb(edge_types)
        else:
            edge_src = edge_dst = edge_emb = None

        for layer in self.layers:
            if edge_emb is not None:
                x = layer(x, edge_src, edge_dst, edge_emb)

        global_emb = self.readout(
            torch.cat([x.mean(0), x.max(0).values], dim=-1)
        )
        return global_emb


class PackingPolicy(nn.Module):
    """
    双路差分策略网络。

    对每个候选位置构造 G_new = G_old ∪ {新节点, 新接触边}，然后：
        g_old = GNN_old(G_old)
        g_new = GNN_new(G_new)
        score = MLP(g_new - g_old)

    GNN_old 与 GNN_new 架构相同，但参数独立：
        GNN_old 学习"当前结构的全局表示"
        GNN_new 学习"加入候选后的全局表示"
        差值    捕获"这个动作带来的结构变化"
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg     = cfg
        d            = cfg.d_model
        self.gnn_old = ParticleGNN(cfg)
        self.gnn_new = ParticleGNN(cfg)
        self.scorer  = nn.Sequential(
            nn.Linear(d, cfg.d_hidden), nn.GELU(),
            nn.Linear(cfg.d_hidden, 1)
        )

    def _make_g_new_obs(self, graph_obs: dict, cand: dict) -> dict:
        """
        根据候选信息，构造加入该候选后的图观测 G_new。
        G_new = G_old + 一个新节点（index = N_old）+ 双向接触边。
        边类型 = new_ptype + 接触粒子类型（0=SS, 1=SL, 2=LL）。
        """
        N         = graph_obs['n_nodes']
        new_ptype = cand['new_ptype']
        touching  = cand['touching']
        old_ntypes = graph_obs['node_types']

        # 新节点：附加到末尾
        new_node_types = np.append(old_ntypes, new_ptype).astype(np.int64)

        # 新边：新节点与每个接触粒子之间的双向边
        add_src, add_dst, add_etype = [], [], []
        for j in touching:
            etype = new_ptype + int(old_ntypes[j])
            add_src   += [N, j]
            add_dst   += [j, N]
            add_etype += [etype, etype]

        new_edge_src   = np.concatenate([
            graph_obs['edge_src'],
            np.array(add_src,   dtype=np.int64)
        ])
        new_edge_dst   = np.concatenate([
            graph_obs['edge_dst'],
            np.array(add_dst,   dtype=np.int64)
        ])
        new_edge_types = np.concatenate([
            graph_obs['edge_types'],
            np.array(add_etype, dtype=np.int64)
        ])

        return {
            'node_types': new_node_types,
            'edge_src':   new_edge_src,
            'edge_dst':   new_edge_dst,
            'edge_types': new_edge_types,
            'n_nodes':    N + 1,
            'n_edges':    len(new_edge_src),
        }

    def forward_single(self, graph_obs: dict, cand_obs: dict,
                       device: torch.device) -> torch.Tensor:
        """
        单步前向推理。
        GNN_old 对 G_old 运行一次；GNN_new 对每个候选的 G_new 各运行一次。
        返回 (n_cands,) 未归一化分数。
        """
        n_cands = cand_obs['n_cands']
        if n_cands == 0:
            return torch.zeros(0, device=device)

        g_old = self.gnn_old(graph_obs, device)   # (d,)

        scores = []
        for cand in cand_obs['candidates']:
            g_new_obs = self._make_g_new_obs(graph_obs, cand)
            g_new     = self.gnn_new(g_new_obs, device)  # (d,)
            score     = self.scorer(g_new - g_old)        # (1,)
            scores.append(score)

        return torch.cat(scores)   # (n_cands,)
