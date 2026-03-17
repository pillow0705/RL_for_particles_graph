import datetime
import json
import pathlib
import pickle
import sys

import numpy as np
import torch

from config import Config

DATA_DIR = pathlib.Path("data")


class _Tee:
    """将 stdout 同时输出到终端和日志文件。"""
    def __init__(self, *files):
        self._files = files

    def write(self, data):
        for f in self._files:
            f.write(data)

    def flush(self):
        for f in self._files:
            f.flush()


def create_experiment_dir() -> pathlib.Path:
    """在 experiments/ 下以当前时间创建实验目录，格式 YYYYMMDD_HHMMSS。"""
    import datetime
    base    = pathlib.Path("experiments")
    base.mkdir(exist_ok=True)
    name    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = base / name
    exp_dir.mkdir()
    return exp_dir


def save_config(exp_dir: pathlib.Path):
    """将 Config 所有超参序列化到 config.json。"""
    d = {}
    for k in sorted(vars(Config)):
        if k.startswith('_'):
            continue
        v = getattr(Config, k)
        if callable(v):
            continue
        if isinstance(v, np.ndarray):
            d[k] = v.tolist()
        elif isinstance(v, torch.device):
            d[k] = str(v)
        else:
            d[k] = v
    with open(exp_dir / "config.json", 'w', encoding='utf-8') as f:
        json.dump(d, f, indent=2, ensure_ascii=False)


# =====================================================================
# 最优堆积保存
# =====================================================================
def save_best_packing(trajs: list, best_phi: float, exp_dir: pathlib.Path) -> float:
    """
    检查本轮轨迹中是否有超过历史最优 phi 的结果。
    若有，保存模型权重由调用方负责；此函数负责写入堆积坐标文件。
    返回更新后的 best_phi。
    """
    for traj in trajs:
        if traj['phi_final'] > best_phi:
            best_phi     = traj['phi_final']
            best_pos_arr = traj['final_pos']
            best_rad_arr = traj['final_rad']
            best_L       = traj['L']

            conf_path = exp_dir / "best_packing.conf"
            with open(conf_path, 'w') as f:
                for p, r in zip(best_pos_arr, best_rad_arr):
                    f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {r:.6f}\n")
                f.write(f"{best_L:.6f} {best_L:.6f} {best_L:.6f}\n")

    return best_phi


# =====================================================================
# 轨迹数据持久化
# =====================================================================
def save_trajectories(trajectories: list, run_dir: pathlib.Path, iteration: int):
    """
    将本轮采集的轨迹追加保存到 data/ 目录。
    文件名: data/{run_name}_iter{NNN}.pkl
    每个文件携带时间戳和来源信息，可独立读取。
    """
    DATA_DIR.mkdir(exist_ok=True)
    filename = DATA_DIR / f"{run_dir.name}_iter{iteration:03d}.pkl"
    payload  = {
        'timestamp'   : datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'run_dir'     : str(run_dir),
        'iteration'   : iteration,
        'trajectories': trajectories,
    }
    with open(filename, 'wb') as f:
        pickle.dump(payload, f)
    return filename


def load_all_trajectories(data_dir: pathlib.Path = DATA_DIR) -> list:
    """
    加载 data/ 下所有 .pkl 文件，合并为一个轨迹列表。
    按文件名（即时间顺序）排序，打印每个文件的来源和数量。
    """
    files = sorted(data_dir.glob("*.pkl"))
    if not files:
        print(f"[data] {data_dir} 下没有找到任何轨迹文件。")
        return []

    all_trajs = []
    for f in files:
        with open(f, 'rb') as fp:
            payload = pickle.load(fp)
        trajs = payload['trajectories']
        print(f"  [data] {f.name}  "
              f"时间={payload['timestamp']}  "
              f"轨迹数={len(trajs)}  "
              f"phi_mean={np.mean([t['phi_final'] for t in trajs]):.4f}")
        all_trajs.extend(trajs)

    print(f"  [data] 共加载 {len(all_trajs)} 条轨迹（来自 {len(files)} 个文件）")
    return all_trajs
