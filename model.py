import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


class GNNLayer(nn.Module):
    """
    单层消息传递 GNN。

    三步流程：
      1. 消息生成：m_ij = MLP( cat[h_j, e_ij] )
         W_A 作用于节点嵌入，W_B 作用于边嵌入，独立变换后叠加激活。
         同一个邻居通过不同类型的面传来的消息因此完全不同。

      2. 聚合：M_i = mean{ m_ij | j ∈ N(i) }
         对节点 i 的所有邻居消息取均值，消除度数差异的影响。

      3. 更新：h_i' = LayerNorm( MLP( cat[h_i, M_i] ) )
         自身旧状态与聚合消息拼接，经 MLP 生成新状态。

    同一层内所有边共享同一组参数（权重共享），不同层参数独立。
    """
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        # 消息 MLP：输入 cat[h_j, e_ij]，维度 2*d_model
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        # 更新 MLP：输入 cat[h_i, M_i]，维度 2*d_model
        self.upd_mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_ff), nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor,
                edge_src: torch.Tensor,
                edge_dst: torch.Tensor,
                edge_emb: torch.Tensor) -> torch.Tensor:
        """
        x:        (N, d_model)  节点嵌入
        edge_src: (E,)          每条边的起点索引
        edge_dst: (E,)          每条边的终点索引
        edge_emb: (E, d_model)  每条边的类型嵌入

        返回: (N, d_model)  更新后的节点嵌入
        """
        N = x.size(0)

        # 1. 消息生成：对每条边，拼接发送方节点嵌入和边嵌入
        msg = self.msg_mlp(
            torch.cat([x[edge_src], edge_emb], dim=-1)
        )                                              # (E, d_model)

        # 2. 聚合：将消息累加到接收方节点，再除以度数取均值
        agg = torch.zeros(N, x.size(1), device=x.device)
        agg.scatter_add_(0, edge_dst.unsqueeze(-1).expand_as(msg), msg)
        degree = torch.bincount(edge_dst, minlength=N).float()
        agg = agg / degree.clamp(min=1).unsqueeze(-1)  # (N, d_model)

        # 3. 更新：拼接自身旧状态和聚合消息，过 MLP + 残差 + LayerNorm
        h = self.upd_mlp(torch.cat([x, agg], dim=-1))
        return self.norm(x + h)


class GraphEncoder(nn.Module):
    """
    将四面体图编码为全局向量。

    输入: graph_obs（节点类型 + 边列表）
    输出: 全局嵌入 (d_model,)

    流程:
      node_types → node_emb → x (N, d)
      edge_types → edge_emb → e (E, d)
      x, e → GNNLayer × n_layers → x' (N, d)
      x' → mean pooling + max pooling → cat → Linear → g (d,)

    空图（step 0）：返回零向量，代表"空集嵌入到原点"。
    此时 CandidateScorer 仅凭 new_tet_type 嵌入选择初始四面体，
    相当于网络学习一个关于初始构型的先验偏好。
    """
    def __init__(self, cfg: Config):
        super().__init__()
        d = cfg.d_model

        # 节点嵌入：5 种四面体类型（SSSS/SSSL/SSLL/SLLL/LLLL）
        self.node_emb = nn.Embedding(cfg.n_tet_types, d)
        # 边嵌入：4 种面类型（SSS/SSL/SLL/LLL）
        self.edge_emb = nn.Embedding(cfg.n_face_types, d)

        self.layers = nn.ModuleList([
            GNNLayer(d, cfg.d_ff)
            for _ in range(cfg.n_layers)
        ])

        # 读出层：mean + max 拼接的 2d → d
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

        node_types = torch.from_numpy(graph_obs['node_types']).to(device)
        x          = self.node_emb(node_types)   # (N, d)

        if graph_obs['n_edges'] > 0:
            edge_src   = torch.from_numpy(graph_obs['edge_src']).to(device)
            edge_dst   = torch.from_numpy(graph_obs['edge_dst']).to(device)
            edge_types = torch.from_numpy(graph_obs['edge_types']).to(device)
            edge_emb   = self.edge_emb(edge_types)   # (E, d)
        else:
            # 无边时跳过消息传递，节点嵌入保持初始状态
            edge_src = edge_dst = edge_emb = None

        for layer in self.layers:
            if edge_emb is not None:
                x = layer(x, edge_src, edge_dst, edge_emb)
            # 无边则节点嵌入不更新（孤立节点退化为纯嵌入查表）

        # mean + max pooling → readout → (d,)
        global_emb = self.readout(
            torch.cat([x.mean(0), x.max(0).values], dim=-1)
        )
        return global_emb


class CandidateScorer(nn.Module):
    """
    对所有候选打分，输出未归一化 logit。

    每个候选由两个整数描述（纯图结构信息，无坐标）：
        face_type:    开放面的类型（0-3），step 0 时为 4（无面 token）
        new_tet_type: 放入新粒子后形成的四面体类型（0-4）

    打分公式：
        score = MLP( cat[global_emb, face_emb, new_tet_emb] )

    三段输入各自贡献：
        global_emb   → 当前图的整体结构（现在是什么状态）
        face_emb     → 候选所在开放面的类型（往哪里放）
        new_tet_emb  → 放入后形成的新四面体类型（放什么）
    """
    def __init__(self, cfg: Config):
        super().__init__()
        d = cfg.d_model

        # face_type 嵌入：size = n_face_types+1，含 step 0 的无面 token（idx=4）
        self.face_emb    = nn.Embedding(cfg.n_face_types + 1, d)
        self.new_tet_emb = nn.Embedding(cfg.n_tet_types, d)

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
        → scores:      (N_cands,) 未归一化 logit
        """
        n      = face_types.size(0)
        g_emb  = global_emb.unsqueeze(0).expand(n, -1)   # (N_cands, d)
        f_emb  = self.face_emb(face_types)                # (N_cands, d)
        nt_emb = self.new_tet_emb(new_tet_types)          # (N_cands, d)

        return self.mlp(torch.cat([g_emb, f_emb, nt_emb], dim=-1)).squeeze(-1)


class PackingPolicy(nn.Module):
    """
    完整策略网络：GraphEncoder（GNN）+ CandidateScorer。

    前向流程：
        graph_obs → GraphEncoder → global_emb (d,)
        cand_obs  ─────────────────────────────────┐
                                                   ↓
                                          CandidateScorer → scores (C,)
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg     = cfg
        self.encoder = GraphEncoder(cfg)
        self.scorer  = CandidateScorer(cfg)

    def forward_single(self, graph_obs: dict, cand_obs: dict,
                       device: torch.device) -> torch.Tensor:
        """
        单步前向推理。返回 (N_cands,) 未归一化分数。
        """
        global_emb    = self.encoder(graph_obs, device)
        face_types    = torch.from_numpy(cand_obs['face_types']).to(device)
        new_tet_types = torch.from_numpy(cand_obs['new_tet_types']).to(device)
        return self.scorer(global_emb, face_types, new_tet_types)
