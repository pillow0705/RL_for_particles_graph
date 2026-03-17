import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from config import Config
from model import PackingPolicy


class Trainer:
    def __init__(self, policy: PackingPolicy, cfg: Config):
        self.policy    = policy
        self.cfg       = cfg
        self.optimizer = optim.Adam(policy.parameters(), lr=cfg.lr)
        self._ckpt     = None

    def backup(self):
        self._ckpt = {k: v.cpu().clone() for k, v in self.policy.state_dict().items()}

    def rollback(self):
        if self._ckpt is None:
            return
        self.policy.load_state_dict(self._ckpt)
        for pg in self.optimizer.param_groups:
            pg['lr'] *= 0.5
        print(f"  [回滚] 模型已恢复，lr 调整为 {self.optimizer.param_groups[0]['lr']:.2e}")

    def current_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']

    def _compute_returns(self, traj):
        phi = traj['phi_final']
        return [phi] * len(traj['steps'])

    def train(self, trajectories):
        cfg    = self.cfg
        device = cfg.device
        policy = self.policy

        all_samples = []
        for traj in trajectories:
            returns = self._compute_returns(traj)
            for step, G in zip(traj['steps'], returns):
                all_samples.append({**step, 'return': G})

        if len(all_samples) == 0:
            return 0.0, 0.0

        all_returns = np.array([s['return'] for s in all_samples])
        ret_mean    = all_returns.mean()
        ret_std     = all_returns.std() + 1e-8

        if cfg.advantage_filter_ratio > 0:
            norm_adv  = np.abs((all_returns - ret_mean) / ret_std)
            threshold = np.quantile(norm_adv, cfg.advantage_filter_ratio)
            all_samples = [s for s, a in zip(all_samples, norm_adv) if a >= threshold]
            if len(all_samples) == 0:
                return 0.0, 0.0

        filtered_adv = np.array([(s['return'] - ret_mean) / ret_std for s in all_samples])
        adv_var      = float(np.var(filtered_adv))

        total_loss = 0.0
        n_updates  = 0

        for epoch in range(cfg.train_epochs):
            np.random.shuffle(all_samples)

            for start in range(0, len(all_samples), cfg.batch_size):
                mb = all_samples[start: start + cfg.batch_size]
                if not mb:
                    continue

                advantages = torch.tensor(
                    [(s['return'] - ret_mean) / ret_std for s in mb],
                    dtype=torch.float32, device=device)

                # 梯度累积：逐样本前向+backward，激活值即时释放。
                # 等价于对整个 mini-batch 求平均 loss，但显存只需一个样本的量。
                self.optimizer.zero_grad()
                batch_loss_val = 0.0
                for s, adv in zip(mb, advantages):
                    scores    = policy.forward_single(s['graph_obs'], s['cand_obs'], device)
                    log_probs = F.log_softmax(scores / cfg.temperature, dim=0)
                    log_pa    = log_probs[s['action']]
                    # 除以 batch 大小以保持与原版相同的梯度尺度
                    loss = -(log_pa * adv) / len(mb)
                    loss.backward()
                    batch_loss_val += loss.item()

                nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += batch_loss_val
                n_updates  += 1

        return total_loss / max(n_updates, 1), adv_var
