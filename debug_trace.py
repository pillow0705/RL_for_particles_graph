#!/usr/bin/env python3
"""
debug_trace.py — 完整数据生成过程调试工具

模拟一次完整的 Episode（单次数据生成），逐步打印所有中间数据：
  1. 每一步图（graph）状态的完整变化
  2. 候选集（candidates）详情
  3. GNN 前向传播：每层输出统计、g_old、g_new_all、diff、scores
  4. 策略概率分布：softmax 分布、熵、top-k 候选详情
  5. 动作选择结果及其属性
  6. 环境 step 反馈：候选集统计（过滤/新增数量）
  7. Episode 结束后：return、advantage 计算的完整过程
  8. 训练模拟：逐样本 log_pa、loss、梯度范数（裁剪前后）

输出文件: debug_trace.txt
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config
from env import ConstructEnv
from model import PackingPolicy, GNNLayer


# ═══════════════════════════════════════════════════════════════════════════════
# 格式化辅助函数
# ═══════════════════════════════════════════════════════════════════════════════

W = 90  # 行宽


def sep(f, char='═', n=W):
    f.write(char * n + '\n')


def header(f, title):
    pad_l = (W - len(title) - 2) // 2
    pad_r = W - len(title) - 2 - pad_l
    sep(f, '═')
    f.write(f"{'═' * pad_l} {title} {'═' * pad_r}\n")
    sep(f, '═')
    f.write('\n')


def subheader(f, title):
    f.write(f"\n{'─' * 4} {title} {'─' * (W - 6 - len(title))}\n")


def tstats(t: torch.Tensor, name: str) -> str:
    """张量统计摘要（单行）。"""
    if t.numel() == 0:
        return f"{name}: [空张量 shape={list(t.shape)}]"
    d = t.detach().float()
    return (f"{name}: shape={list(t.shape)}"
            f"  min={d.min().item():+.4f}"
            f"  max={d.max().item():+.4f}"
            f"  mean={d.mean().item():+.4f}"
            f"  std={d.std(correction=0).item():.4f}")


def arr_hist(arr: np.ndarray, bins=5) -> str:
    """简单的数组分布直方图（字符串）。"""
    if len(arr) == 0:
        return "[]"
    counts, edges = np.histogram(arr, bins=bins)
    parts = []
    for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
        parts.append(f"[{lo:.3f},{hi:.3f}):{c}")
    return "  ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# GNN 中间层 Hook（记录每层的输入/输出统计）
# ═══════════════════════════════════════════════════════════════════════════════

class LayerRecorder:
    """为 GNNLayer 注册 forward hook，记录每层 x_in / x_out / msg / agg 的统计。"""

    def __init__(self):
        self.records: list[dict] = []
        self._hooks = []

    def attach(self, gnn_module):
        """gnn_module 是 ParticleGNN 实例。"""
        self.records = []
        for i, layer in enumerate(gnn_module.layers):
            record = {'layer_idx': i}
            self.records.append(record)

            def make_hook(rec, idx):
                def hook(module, inp, out):
                    x_in  = inp[0]   # (N, d)
                    x_out = out      # (N, d)
                    rec['x_in']  = tstats(x_in,  f"  层{idx} x_in ")
                    rec['x_out'] = tstats(x_out, f"  层{idx} x_out")
                return hook
            h = layer.register_forward_hook(make_hook(record, i))
            self._hooks.append(h)

    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def print_to(self, f, gnn_name='GNN'):
        f.write(f"  {gnn_name} 各层中间输出统计:\n")
        for rec in self.records:
            f.write(f"    {rec.get('x_in',  '—')}\n")
            f.write(f"    {rec.get('x_out', '—')}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 打印图状态
# ═══════════════════════════════════════════════════════════════════════════════

def print_graph(f, graph_obs: dict, indent='  '):
    n  = graph_obs['n_nodes']
    e  = graph_obs['n_edges']
    nt = graph_obs['node_types']
    et = graph_obs['edge_types']

    n_s = int((nt == 0).sum())
    n_l = int((nt == 1).sum())
    e_ss = int((et == 0).sum())
    e_sl = int((et == 1).sum())
    e_ll = int((et == 2).sum())

    f.write(f"{indent}节点数 N={n}  [S={n_s}, L={n_l}]\n")
    f.write(f"{indent}边   数 E={e}  [SS={e_ss}, SL={e_sl}, LL={e_ll}]  (双向计)\n")
    f.write(f"{indent}node_types = {nt.tolist()}\n")

    if 0 < e <= 30:
        src_list = graph_obs['edge_src'].tolist()
        dst_list = graph_obs['edge_dst'].tolist()
        et_list  = graph_obs['edge_types'].tolist()
        pairs = [f"{s}→{d}[{t}]" for s, d, t in zip(src_list, dst_list, et_list)]
        f.write(f"{indent}边列表: {', '.join(pairs)}\n")
    elif e > 30:
        src_list = graph_obs['edge_src'][:20].tolist()
        dst_list = graph_obs['edge_dst'][:20].tolist()
        et_list  = graph_obs['edge_types'][:20].tolist()
        pairs = [f"{s}→{d}[{t}]" for s, d, t in zip(src_list, dst_list, et_list)]
        f.write(f"{indent}边列表(前20条): {', '.join(pairs)} ...\n")

    # 每个节点的度
    if n > 0 and e > 0:
        degrees = np.bincount(graph_obs['edge_dst'], minlength=n)
        f.write(f"{indent}节点度数: min={degrees.min()}  max={degrees.max()}  "
                f"mean={degrees.mean():.2f}  degrees={degrees.tolist()}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 打印候选集
# ═══════════════════════════════════════════════════════════════════════════════

def print_candidates(f, cand_obs: dict, indent='  ', max_show=15):
    n     = cand_obs['n_cands']
    cands = cand_obs['candidates']

    n_s = sum(1 for c in cands if c['new_ptype'] == 0)
    n_l = sum(1 for c in cands if c['new_ptype'] == 1)
    coords = [len(c['touching']) for c in cands]

    f.write(f"{indent}候选数 C={n}  [小球S={n_s}, 大球L={n_l}]\n")
    if n > 0:
        f.write(f"{indent}配位数: min={min(coords)}  max={max(coords)}  "
                f"mean={np.mean(coords):.2f}\n")
        f.write(f"{indent}配位数分布: {arr_hist(np.array(coords))}\n")

    show = min(n, max_show)
    if show > 0:
        f.write(f"{indent}前 {show} 个候选 [idx | ptype | coord | touching]:\n")
        for i, c in enumerate(cands[:show]):
            pt = 'S' if c['new_ptype'] == 0 else 'L'
            f.write(f"{indent}  [{i:3d}] {pt}  coord={len(c['touching'])}  "
                    f"touching={c['touching']}\n")
        if n > max_show:
            f.write(f"{indent}  ... (省略 {n - max_show} 个)\n")


# ═══════════════════════════════════════════════════════════════════════════════
# GNN 前向传播（带中间层记录）
# ═══════════════════════════════════════════════════════════════════════════════

def run_gnn_forward(f, policy: PackingPolicy,
                    graph_obs: dict, cand_obs: dict,
                    device: torch.device, temperature: float):
    """
    执行完整 GNN 前向传播，打印所有中间数据，返回 (scores, log_probs, probs)。
    若无候选，返回 (None, None, None)。
    """
    n_cands = cand_obs['n_cands']
    if n_cands == 0:
        f.write("  候选数=0，跳过 GNN 前向传播。\n")
        return None, None, None

    subheader(f, "GNN 前向传播")

    rec_old = LayerRecorder()
    rec_new = LayerRecorder()

    with torch.no_grad():
        # ── GNN_old ────────────────────────────────────────────────────────
        rec_old.attach(policy.gnn_old)
        g_old = policy.gnn_old(graph_obs, device)   # (d,)
        rec_old.detach()

        # ── GNN_new (batched) ──────────────────────────────────────────────
        rec_new.attach(policy.gnn_new)
        g_new_all = policy._forward_gnn_new_batched(graph_obs, cand_obs, device)  # (C, d)
        rec_new.detach()

        # ── Diff & Scores ──────────────────────────────────────────────────
        diff   = g_new_all - g_old.unsqueeze(0)       # (C, d)
        scores = policy.scorer(diff).squeeze(-1)       # (C,)

        # ── Policy Distribution ────────────────────────────────────────────
        log_probs = F.log_softmax(scores / temperature, dim=0)
        probs     = log_probs.exp()

    # GNN_old 层级输出
    f.write("\n  [GNN_old] 各层激活统计:\n")
    for rec in rec_old.records:
        f.write(f"    {rec.get('x_in',  '—')}\n")
        f.write(f"    {rec.get('x_out', '—')}\n")
    f.write(f"  {tstats(g_old, 'g_old (readout)')}\n")

    # GNN_new 层级输出
    f.write("\n  [GNN_new / batched] 各层激活统计 (大图合并后):\n")
    for rec in rec_new.records:
        f.write(f"    {rec.get('x_in',  '—')}\n")
        f.write(f"    {rec.get('x_out', '—')}\n")
    f.write(f"  {tstats(g_new_all, 'g_new_all (readout, C×d)')}\n")

    # Diff & Scores
    f.write(f"\n  {tstats(diff,   'diff = g_new - g_old')}\n")
    f.write(f"  {tstats(scores, 'scores (scorer MLP)')}\n")

    # Policy distribution
    entropy   = -(probs * log_probs).sum().item()
    max_ent   = np.log(n_cands) if n_cands > 1 else 0.0
    f.write(f"\n  温度 T={temperature}\n")
    f.write(f"  {tstats(probs, 'probs (softmax)')}\n")
    f.write(f"  策略熵 H={entropy:.4f}  (最大熵 H_max={max_ent:.4f}  "
            f"有效比 {entropy/max(max_ent, 1e-8):.2%})\n")

    # Top-5 candidates
    top_k = min(5, n_cands)
    top_vals, top_idx = torch.topk(probs, top_k)
    f.write(f"\n  Top-{top_k} 候选 (rank | idx | prob | score | ptype | coord | touching):\n")
    for rank, (idx, prob_val) in enumerate(zip(top_idx.tolist(), top_vals.tolist())):
        c  = cand_obs['candidates'][idx]
        pt = 'S' if c['new_ptype'] == 0 else 'L'
        f.write(f"    rank{rank+1}: [{idx:3d}]  prob={prob_val:.5f}  "
                f"score={scores[idx].item():+.4f}  ptype={pt}  "
                f"coord={len(c['touching'])}  touching={c['touching']}\n")

    return scores, log_probs, probs


# ═══════════════════════════════════════════════════════════════════════════════
# 动作选择结果
# ═══════════════════════════════════════════════════════════════════════════════

def print_action(f, action_idx: int, cand_obs: dict,
                 scores: torch.Tensor, probs: torch.Tensor,
                 log_probs: torch.Tensor, indent='  '):
    c  = cand_obs['candidates'][action_idx]
    pt = 'S' if c['new_ptype'] == 0 else 'L'

    f.write(f"{indent}选择动作 action_idx={action_idx}\n")
    f.write(f"{indent}  ptype={pt}  coord={len(c['touching'])}  "
            f"touching={c['touching']}\n")
    f.write(f"{indent}  score={scores[action_idx].item():+.4f}  "
            f"prob={probs[action_idx].item():.5f}  "
            f"log_prob={log_probs[action_idx].item():.5f}\n")

    rank = int((probs > probs[action_idx]).sum().item()) + 1
    f.write(f"{indent}  在概率排名中位于第 {rank}/{cand_obs['n_cands']} 位\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 训练模拟
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_training(f, trajectory: dict, cfg: Config, policy: PackingPolicy):
    """模拟一次完整训练步骤，打印 return / advantage / loss / 梯度 所有数据。"""
    device = cfg.device
    steps  = trajectory['steps']
    phi_f  = trajectory['phi_final']
    T      = len(steps)

    header(f, "TRAINING SIMULATION — 模拟训练步骤")

    # ── 1. Return ──────────────────────────────────────────────────────────
    subheader(f, "① Return 计算")
    returns = np.array([phi_f] * T, dtype=np.float32)
    f.write(f"  phi_final = {phi_f:.6f}\n")
    f.write(f"  Episode 长度 T = {T}\n")
    f.write(f"  所有步骤 return = phi_final = {phi_f:.6f}  "
            f"（无折扣，稀疏奖励）\n")
    f.write(f"  returns array = {returns[:5].tolist()} ... (共 {T} 个)\n")

    # ── 2. Advantage ───────────────────────────────────────────────────────
    subheader(f, "② Advantage 计算（跨所有样本归一化）")
    ret_mean = float(returns.mean())
    ret_std  = float(returns.std()) + 1e-8
    adv_raw  = (returns - ret_mean) / ret_std

    f.write(f"  ret_mean = {ret_mean:.6f}\n")
    f.write(f"  ret_std  = {ret_std:.8f}  (含 1e-8 防零)\n")
    f.write(f"  注意：单 episode 时所有 return 相同 → std≈0 → advantage 退化。\n")
    f.write(f"        实际训练跨 {cfg.samples_per_iter} 个 episode，advantage 才有意义。\n")
    f.write(f"  normalized_adv: mean={adv_raw.mean():.6f}  std={adv_raw.std():.6f}\n")

    # ── 3. Advantage 过滤 ──────────────────────────────────────────────────
    subheader(f, "③ Advantage 过滤（advantage_filter_ratio）")
    ratio     = cfg.advantage_filter_ratio
    norm_abs  = np.abs(adv_raw)
    threshold = float(np.quantile(norm_abs, ratio))
    mask      = norm_abs >= threshold
    n_kept    = int(mask.sum())

    f.write(f"  advantage_filter_ratio = {ratio}\n")
    f.write(f"  过滤阈值 = quantile({ratio}, |adv|) = {threshold:.6f}\n")
    f.write(f"  保留样本: {n_kept} / {T}  ({100*n_kept/max(T,1):.1f}%)\n")
    f.write(f"  过滤后 adv: {adv_raw[mask][:10].tolist()} ...\n")

    # ── 4. 训练步骤 ────────────────────────────────────────────────────────
    subheader(f, "④ 训练步骤（mini-batch 梯度累积）")

    all_samples = [{'graph_obs': s['graph_obs'],
                    'cand_obs':  s['cand_obs'],
                    'action':    s['action'],
                    'return':    float(phi_f)} for s in steps]

    kept_samples = [s for s, k in zip(all_samples, mask) if k]
    if not kept_samples:
        f.write("  过滤后无有效样本，跳过训练。\n")
        return

    batch = kept_samples[:cfg.batch_size]
    batch_size = len(batch)
    f.write(f"  mini-batch 大小 = {batch_size}  "
            f"(从 {n_kept} 个过滤后样本取前 {batch_size})\n")
    f.write(f"  学习率 lr = {cfg.lr}\n")
    f.write(f"  温度 T    = {cfg.temperature}\n\n")

    # 该 batch 的 advantage 值
    batch_advs_np = np.array([(s['return'] - ret_mean) / ret_std for s in batch])
    batch_advs    = torch.tensor(batch_advs_np, dtype=torch.float32, device=device)

    f.write(f"  batch advantages: {batch_advs_np.tolist()}\n\n")

    # 训练模式并设置 optimizer
    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    optimizer.zero_grad()

    # 表头
    f.write(f"  {'idx':>4}  {'n_cands':>7}  {'action':>6}  {'score_a':>9}  "
            f"{'log_pa':>9}  {'adv':>9}  {'loss_contrib':>13}\n")
    f.write(f"  {'─'*4}  {'─'*7}  {'─'*6}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*13}\n")

    total_loss_val = 0.0
    log_pa_list    = []
    score_a_list   = []

    for s_idx, (s, adv_t) in enumerate(zip(batch, batch_advs)):
        scores_t  = policy.forward_single(s['graph_obs'], s['cand_obs'], device)
        log_probs = F.log_softmax(scores_t / cfg.temperature, dim=0)
        action    = s['action']
        log_pa    = log_probs[action]
        loss      = -(log_pa * adv_t) / batch_size
        loss.backward()

        lv = loss.item()
        sv = scores_t[action].item()
        lp = log_pa.item()

        total_loss_val += lv
        log_pa_list.append(lp)
        score_a_list.append(sv)

        f.write(f"  {s_idx:>4}  {s['cand_obs']['n_cands']:>7}  {action:>6}  "
                f"{sv:>+9.4f}  {lp:>9.4f}  {adv_t.item():>+9.4f}  "
                f"{lv:>+13.6f}\n")

    f.write(f"\n  批次总 loss = {total_loss_val:.6f}\n")
    f.write(f"  log_pa 统计: min={min(log_pa_list):.4f}  "
            f"max={max(log_pa_list):.4f}  mean={np.mean(log_pa_list):.4f}\n")
    f.write(f"  score_a 统计: min={min(score_a_list):.4f}  "
            f"max={max(score_a_list):.4f}  mean={np.mean(score_a_list):.4f}\n")

    # ── 5. 梯度范数（裁剪前）─────────────────────────────────────────────
    subheader(f, "⑤ 梯度范数（裁剪前，max_norm=1.0）")
    param_info = []
    global_norm_sq = 0.0
    for name, param in policy.named_parameters():
        if param.grad is not None:
            gn = param.grad.norm(2).item()
            global_norm_sq += gn ** 2
            param_info.append((name, param.shape, gn))
        else:
            param_info.append((name, param.shape, None))

    global_norm = global_norm_sq ** 0.5
    f.write(f"  全局梯度范数 (L2) = {global_norm:.6f}\n\n")
    f.write(f"  {'参数名':60s}  {'shape':20s}  {'grad_norm':>10}\n")
    f.write(f"  {'─'*60}  {'─'*20}  {'─'*10}\n")
    for name, shape, gn in param_info:
        gn_str = f"{gn:.6f}" if gn is not None else "None (no grad)"
        f.write(f"  {name:60s}  {str(list(shape)):20s}  {gn_str:>10}\n")

    # ── 6. 梯度裁剪 & Optimizer Step ──────────────────────────────────────
    subheader(f, "⑥ 梯度裁剪 & Optimizer Step")
    nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)

    clipped_norm_sq = 0.0
    for param in policy.parameters():
        if param.grad is not None:
            clipped_norm_sq += param.grad.norm(2).item() ** 2
    clipped_norm = clipped_norm_sq ** 0.5

    f.write(f"  裁剪前全局范数 = {global_norm:.6f}\n")
    f.write(f"  裁剪后全局范数 = {clipped_norm:.6f}  "
            f"({'已裁剪' if global_norm > 1.0 else '未超出阈值，未裁剪'})\n")
    f.write(f"  执行 optimizer.step() (Adam, lr={cfg.lr})\n")
    optimizer.step()
    f.write(f"  参数更新完成。\n")

    # ── 7. 参数变化量 ─────────────────────────────────────────────────────
    subheader(f, "⑦ 关键参数更新量（Δw 统计）")
    # 注意：这里只做一次前向+反向，参数已更新，无法对比更新前后。
    # 故仅汇报此时参数的基本统计（可作参考）。
    f.write(f"  （本工具仅做单次模拟，下为更新后参数的 L2 范数）\n\n")
    f.write(f"  {'参数名':60s}  {'param_norm':>12}\n")
    f.write(f"  {'─'*60}  {'─'*12}\n")
    for name, param in policy.named_parameters():
        pn = param.data.norm(2).item()
        f.write(f"  {name:60s}  {pn:>12.6f}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════════════════

def main(output_file='debug_trace.txt', max_steps=200, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg    = Config()
    device = cfg.device

    policy = PackingPolicy(cfg).to(device)
    policy.eval()   # episode 采集阶段不需要梯度

    env = ConstructEnv(cfg)

    print(f"[debug_trace] 开始调试，输出文件: {output_file}")
    print(f"[debug_trace] 设备={device}  max_steps={max_steps}")

    with open(output_file, 'w', encoding='utf-8') as f:

        # ── 文件头 ─────────────────────────────────────────────────────────
        header(f, "RL 粒子堆积调试追踪报告")
        f.write(f"生成时间  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"设备      : {device}\n")
        f.write(f"随机种子  : seed={seed}\n\n")
        f.write(f"环境参数:\n")
        f.write(f"  target_N={cfg.target_N}  max_particles={cfg.max_particles}  "
                f"target_phi={cfg.target_phi}\n")
        f.write(f"  d_small={cfg.d_small}  d_large={cfg.d_large}  "
                f"collision_tol={cfg.collision_tol}  edge_tol={cfg.edge_tol}\n")
        f.write(f"  max_candidates={cfg.max_candidates}\n\n")
        f.write(f"模型参数:\n")
        f.write(f"  d_model={cfg.d_model}  n_layers={cfg.n_layers}  "
                f"d_ff={cfg.d_ff}  d_hidden={cfg.d_hidden}\n\n")
        f.write(f"训练超参:\n")
        f.write(f"  temperature={cfg.temperature}  lr={cfg.lr}  "
                f"batch_size={cfg.batch_size}  train_epochs={cfg.train_epochs}\n")
        f.write(f"  advantage_filter_ratio={cfg.advantage_filter_ratio}  "
                f"gamma={cfg.gamma}\n")
        f.write('\n')

        # 模型参数数量
        total_params = sum(p.numel() for p in policy.parameters())
        f.write(f"模型总参数量: {total_params:,}\n")
        f.write(f"  GNN_old:  {sum(p.numel() for p in policy.gnn_old.parameters()):,}\n")
        f.write(f"  GNN_new:  {sum(p.numel() for p in policy.gnn_new.parameters()):,}\n")
        f.write(f"  Scorer:   {sum(p.numel() for p in policy.scorer.parameters()):,}\n")
        f.write('\n')

        # ── EPISODE ────────────────────────────────────────────────────────
        header(f, "EPISODE 生成过程")

        graph_obs, cand_obs = env.reset()
        traj_steps = []
        step_idx   = 0
        done       = False

        while not done and step_idx < max_steps:

            sep(f, '─')
            f.write(f"\n{'■' * 3}  Step {step_idx:4d}  "
                    f"[当前粒子数 N={env.n}  phi={env.get_phi():.5f}]  "
                    f"{'■' * 3}\n\n")

            # 图状态
            subheader(f, "图（Graph）状态")
            print_graph(f, graph_obs)

            # 候选集
            subheader(f, "候选集（Candidates）")
            print_candidates(f, cand_obs)

            # GNN 前向
            scores, log_probs, probs = run_gnn_forward(
                f, policy, graph_obs, cand_obs, device, cfg.temperature)

            if scores is None:
                f.write("  候选为空，Episode 自然终止。\n")
                done = True
                break

            # 动作选择
            subheader(f, "动作选择（随机采样）")
            with torch.no_grad():
                action_idx = torch.multinomial(probs, 1).item()
            print_action(f, action_idx, cand_obs, scores, probs, log_probs)

            # 环境 step
            subheader(f, "环境 Step 结果")
            (next_graph_obs, next_cand_obs), reward, done = env.step(action_idx)

            cs = env._last_cand_stats
            f.write(f"  候选集变化: 前={cs.get('n_before','?')}  "
                    f"已过滤={cs.get('n_filtered','?')}  "
                    f"新增={cs.get('n_added','?')}  "
                    f"后={cs.get('n_after','?')}\n")
            f.write(f"  reward={reward:.6f}  done={done}\n")
            f.write(f"  更新后 phi={env.get_phi():.6f}  N={env.n}\n")

            # 保存步骤
            traj_steps.append({
                'graph_obs':  graph_obs,
                'cand_obs':   cand_obs,
                'action':     action_idx,
                'reward':     reward,
                'cand_stats': dict(cs),
            })

            graph_obs = next_graph_obs
            cand_obs  = next_cand_obs
            step_idx += 1

            # 检查 cand 是否耗尽
            if not done and cand_obs['n_cands'] == 0:
                f.write("\n  候选集耗尽，Episode 强制终止。\n")
                done = True
                break

        # ── Episode 结束总结 ───────────────────────────────────────────────
        sep(f, '═')
        f.write(f"\n{'■' * 3}  Episode 结束  {'■' * 3}\n\n")
        f.write(f"  实际步数 T = {step_idx}\n")
        f.write(f"  最终粒子数 N = {env.n}\n")
        f.write(f"  最终 phi = {env.get_phi():.6f}\n")
        f.write(f"  盒子边长 L = {env.L:.4f}\n")

        # 接触图统计
        final_graph = env._get_graph_obs()
        nt = final_graph['node_types']
        et = final_graph['edge_types']
        f.write(f"\n  最终图摘要:\n")
        f.write(f"    节点: N={env.n}  [S={int((nt==0).sum())}, L={int((nt==1).sum())}]\n")
        f.write(f"    边:   E={final_graph['n_edges']}  "
                f"[SS={int((et==0).sum())}, SL={int((et==1).sum())}, "
                f"LL={int((et==2).sum())}]\n")
        if env.n > 0:
            degrees = np.bincount(final_graph['edge_dst'], minlength=env.n)
            f.write(f"    平均配位数 Z = {degrees.mean():.3f}  "
                    f"(min={degrees.min()}, max={degrees.max()})\n")
        f.write('\n')

        # ── 训练模拟 ───────────────────────────────────────────────────────
        trajectory = {
            'steps':     traj_steps,
            'phi_final': env.get_phi(),
            'final_pos': np.array(env.pos),
            'final_rad': np.array(env.rad),
            'L':         env.L,
        }

        if traj_steps:
            simulate_training(f, trajectory, cfg, policy)

        # ── 报告结尾 ───────────────────────────────────────────────────────
        header(f, "调试报告结束")
        f.write(f"  文件路径: {os.path.abspath(output_file)}\n")
        f.write(f"  结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"[debug_trace] 完成！报告已保存到: {output_file}")
    print(f"  总步数={step_idx}  N={env.n}  phi={env.get_phi():.6f}")


if __name__ == '__main__':
    main(output_file='debug_trace.txt', max_steps=200, seed=42)
