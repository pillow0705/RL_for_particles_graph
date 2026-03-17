import numpy as np
import torch
import multiprocessing as mp

from config import Config
from model import PackingPolicy
from env import ConstructEnv


def _worker_collect_episode(args):
    policy_state_dict, greedy, temperature, seed = args
    torch.set_num_threads(1)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cfg = Config()

    if policy_state_dict is not None:
        policy = PackingPolicy(cfg)
        policy.load_state_dict(policy_state_dict)
        policy.eval()
    else:
        policy = None

    env              = ConstructEnv(cfg)
    graph_obs, cand_obs = env.reset()
    traj_steps       = []
    done             = False

    while not done:
        n_cands = cand_obs['n_cands']
        if n_cands == 0:
            break

        if policy is None:
            action_idx = int(np.random.randint(0, n_cands))
        else:
            with torch.no_grad():
                scores = policy.forward_single(graph_obs, cand_obs,
                                               device=torch.device('cpu'))
            scores_np = scores.numpy()

            if greedy:
                action_idx = int(np.argmax(scores_np))
            else:
                scores_np -= scores_np.max()
                probs = np.exp(scores_np / temperature)
                probs /= probs.sum()
                action_idx = int(np.random.choice(n_cands, p=probs))

        (next_graph_obs, next_cand_obs), reward, done = env.step(action_idx)

        traj_steps.append({
            'graph_obs':  graph_obs,
            'cand_obs':   cand_obs,
            'action':     action_idx,
            'reward':     reward,
            'cand_stats': dict(env._last_cand_stats),
        })

        graph_obs, cand_obs = next_graph_obs, next_cand_obs

    phi_final = env.get_phi()
    return {
        'steps':     traj_steps,
        'phi_final': phi_final,
        'final_pos': np.array(env.pos,  dtype=np.float64).copy(),
        'final_rad': np.array(env.rad,  dtype=np.float64).copy(),
        'L':         env.L,
    }


class DataCollector:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def collect(self, policy, n_samples, greedy=False):
        cfg = self.cfg

        if policy is not None:
            state_dict = {k: v.cpu() for k, v in policy.state_dict().items()}
        else:
            state_dict = None

        seeds     = np.random.randint(0, 2**31, size=n_samples).tolist()
        args_list = [(state_dict, greedy, cfg.temperature, seed) for seed in seeds]

        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=min(cfg.num_workers, n_samples)) as pool:
            trajectories = pool.map(_worker_collect_episode, args_list)

        phis = [t['phi_final'] for t in trajectories]
        print(f"  [采集完成] {n_samples} 条  "
              f"phi: mean={np.mean(phis):.4f}  "
              f"max={np.max(phis):.4f}  "
              f"min={np.min(phis):.4f}")
        return trajectories
