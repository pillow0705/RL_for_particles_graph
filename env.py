import numpy as np
from itertools import combinations

from config import Config
from physics import (pbc_diff, solve_three_spheres,
                     check_collision, check_single_collision)


class ConstructEnv:
    """
    双元球堆积环境（粒子接触图版本）。

    状态表示
    --------
    节点 = 粒子（类型：0=小球，1=大球）
    边   = 两粒子近似相切（gap < tol * r_sum），双向存储
    边类型 = 两端粒子类型之和：0=SS，1=SL，2=LL

    该表示为坐标无关的拓扑图，对平移和旋转不变。

    序列生成流程
    ------------
    reset():  随机放置初始 4 个两两相切的粒子，建立初始接触图和候选集。
    step(a):  放置第 a 号候选，更新接触图、三元组集和候选集。

    候选集维护
    ----------
    与原版相同：维护三元组集 _triplet_set 和候选字典 _candidate_set。
    每个候选额外记录 touching（已知接触粒子的索引列表），用于在模型中
    增量构造 G_new = G_old ∪ {新节点, 新边}。

    观测格式
    --------
    graph_obs = {
        'node_types': (N,) int64,    # 0=小, 1=大
        'edge_src':   (2E,) int64,   # 双向边，源节点
        'edge_dst':   (2E,) int64,   # 双向边，目标节点
        'edge_types': (2E,) int64,   # 0=SS, 1=SL, 2=LL
        'n_nodes': int,
        'n_edges': int,              # 等于 2E（含双向）
    }
    cand_obs = {
        'n_cands': int,
        'candidates': [              # 长度 n_cands 的列表
            {
                'new_ptype': int,        # 0 或 1
                'touching':  list[int],  # 接触该候选的粒子索引
            },
            ...
        ],
    }
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        r_S     = cfg.d_small / 2
        r_L     = cfg.d_large / 2
        avg_vol = 0.5 * (4/3 * np.pi * r_S**3) + 0.5 * (4/3 * np.pi * r_L**3)
        self.L  = (cfg.target_N * avg_vol / cfg.target_phi) ** (1/3)
        self.reset()

    # ------------------------------------------------------------------
    # 环境接口
    # ------------------------------------------------------------------

    def reset(self):
        """清空所有状态，随机放置初始 4 个粒子，返回 (graph_obs, cand_obs)。"""
        # 物理状态
        self.pos            = []   # list of ndarray(3,)
        self.rad            = []   # list of float
        self.particle_types = []   # list of int: 0=S, 1=L
        self.n              = 0

        # 接触图状态（粒子级别）
        self.p_node_types = []   # 粒子类型列表，与 self.particle_types 同步
        self.p_edge_src   = []   # 双向接触边
        self.p_edge_dst   = []
        self.p_edge_types = []   # 0=SS, 1=SL, 2=LL

        # 接触对（用于候选集维护）
        self.contact_pairs = set()   # set of (min_idx, max_idx)

        # 候选集合（与原版结构一致）
        self._candidate_set    = {}   # cid → ndarray[x,y,z,r,coord]
        self._cand_info        = {}   # cid → {triplet, new_ptype, touching}
        self._triplet_set      = {}   # (i,j,k) → set of cid
        self._cand_to_triplets = {}   # cid → set of (i,j,k)
        self._cand_counter     = 0
        self.current_candidates = []
        self._last_cand_stats   = {}

        # 放置初始 4 个粒子并初始化候选集
        self._init_four_particles()
        self._init_sets()
        self._update_current_candidates()

        return self._get_graph_obs(), self._get_cand_obs()

    def step(self, action_idx):
        """
        放置第 action_idx 号候选，更新所有状态。
        返回 ((graph_obs, cand_obs), reward, done)。
        """
        cfg = self.cfg

        if action_idx >= len(self.current_candidates):
            return (self._get_graph_obs(), self._get_cand_obs()), 0.0, True

        chosen    = self.current_candidates[action_idx]
        pos_new   = np.array(chosen['pos'])
        r_new     = chosen['r']
        new_ptype = chosen['new_ptype']
        touching  = chosen['touching']   # 已知接触粒子索引（候选生成时记录）
        cid       = chosen['cid']

        self._remove_candidate(cid)

        # 将新粒子加入物理状态
        n_new = self.n
        self.pos.append(pos_new % self.L)
        self.rad.append(r_new)
        self.particle_types.append(new_ptype)
        self.p_node_types.append(new_ptype)
        self.n += 1

        # 更新接触图：为新粒子和所有接触粒子添加双向边
        for m in touching:
            self.contact_pairs.add((min(m, n_new), max(m, n_new)))
            etype = new_ptype + self.particle_types[m]
            self.p_edge_src  += [n_new, m]
            self.p_edge_dst  += [m, n_new]
            self.p_edge_types += [etype, etype]

        # 过滤与新粒子碰撞的旧候选，并更新接触信息
        n_before_filter = len(self._candidate_set)
        self._filter_candidates(pos_new % self.L, r_new, n_new)
        n_filtered = n_before_filter - len(self._candidate_set)

        # 为新粒子加入所有包含它的新三元组
        n_before_add = len(self._candidate_set)
        self._add_new_triplets(n_new)
        n_added = len(self._candidate_set) - n_before_add

        self._last_cand_stats = {
            'n_before':   n_before_filter,
            'n_filtered': n_filtered,
            'n_added':    n_added,
            'n_after':    len(self._candidate_set),
        }

        self._update_current_candidates()

        done     = (self.n >= cfg.max_particles)
        cand_obs = self._get_cand_obs()
        if cand_obs['n_cands'] == 0 and not done:
            done = True
        reward = self.get_phi() if done else 0.0
        return (self._get_graph_obs(), cand_obs), reward, done

    def get_phi(self):
        box_vol   = self.L ** 3
        total_vol = sum((4/3) * np.pi * r**3 for r in self.rad)
        return total_vol / box_vol

    # ------------------------------------------------------------------
    # 观测构建
    # ------------------------------------------------------------------

    def _get_graph_obs(self):
        return {
            'node_types': np.array(self.p_node_types, dtype=np.int64),
            'edge_src':   np.array(self.p_edge_src,   dtype=np.int64),
            'edge_dst':   np.array(self.p_edge_dst,   dtype=np.int64),
            'edge_types': np.array(self.p_edge_types, dtype=np.int64),
            'n_nodes':    self.n,
            'n_edges':    len(self.p_edge_src),
        }

    def _get_cand_obs(self):
        n = len(self.current_candidates)
        return {
            'n_cands': n,
            'candidates': [
                {
                    'new_ptype': c['new_ptype'],
                    'touching':  list(c['touching']),
                }
                for c in self.current_candidates
            ],
        }

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def _init_four_particles(self):
        """
        随机分配 4 个粒子的 S/L 类型，几何方法摆放成两两相切的四面体：
          p0 居中，p1 在 +x 方向，p2 在 xy 平面，p3 用三球解析解确定。
        """
        cfg    = self.cfg
        ptypes = [np.random.randint(0, 2) for _ in range(4)]
        radii  = [cfg.d_large/2 if t == 1 else cfg.d_small/2 for t in ptypes]
        r0, r1, r2, r3 = radii

        p0  = np.array([self.L/2, self.L/2, self.L/2])
        p1  = p0 + np.array([r0 + r1, 0.0, 0.0])
        d01 = r0 + r1;  d02 = r0 + r2;  d12 = r1 + r2
        x2  = (d02**2 - d12**2 + d01**2) / (2 * d01)
        y2  = np.sqrt(max(0.0, d02**2 - x2**2))
        p2  = p0 + np.array([x2, y2, 0.0])
        valid, p3, _ = solve_three_spheres(p0, r0, p1, r1, p2, r2, r3)
        if not valid:
            p3 = p0 + np.array([0.0, 0.0, r0 + r3])

        for p, r, t in zip([p0, p1, p2, p3], radii, ptypes):
            self.pos.append(p % self.L)
            self.rad.append(r)
            self.particle_types.append(t)
            self.p_node_types.append(t)
        self.n = 4

        # 初始 4 粒子两两相切，建立所有接触对和接触边
        for i in range(4):
            for j in range(i + 1, 4):
                self.contact_pairs.add((i, j))
                etype = ptypes[i] + ptypes[j]
                self.p_edge_src   += [i, j]
                self.p_edge_dst   += [j, i]
                self.p_edge_types += [etype, etype]

    # ------------------------------------------------------------------
    # 候选集合增量维护
    # ------------------------------------------------------------------

    def _new_cand_id(self):
        cid = self._cand_counter
        self._cand_counter += 1
        return cid

    def _check_and_collect_touching(self, sol, r_new, all_pos, all_rad):
        """
        同时做碰撞检测并收集接触粒子索引。
        返回 (collision: bool, touching: list[int])。
        sol 可以是展开（unwrapped）坐标，pbc_diff 负责处理。
        """
        touching = []
        tol = self.cfg.collision_tol
        for m in range(len(all_pos)):
            r_sum = all_rad[m] + r_new
            gap   = np.linalg.norm(pbc_diff(all_pos[m], sol, self.L)) - r_sum
            if gap < -tol * r_sum:
                return True, []
            if gap < tol * r_sum:
                touching.append(m)
        return False, touching

    def _process_triplet(self, i, j, k):
        """
        对三元组 (i,j,k) 生成合法候选点。
        每种新粒子类型（S/L）各求两个解，通过碰撞检测后加入候选集。
        同时记录接触粒子索引列表（touching），供模型构造 G_new 使用。
        """
        cfg     = self.cfg
        all_pos = np.array(self.pos, dtype=np.float64)
        all_rad = np.array(self.rad, dtype=np.float64)
        p_i, r_i = all_pos[i], all_rad[i]
        p_j, r_j = all_pos[j], all_rad[j]
        p_k, r_k = all_pos[k], all_rad[k]

        # PBC 展开：将 j、k 的坐标对齐到以 i 为原点的近邻像
        p_j = p_i + pbc_diff(p_j, p_i, self.L)
        p_k = p_i + pbc_diff(p_k, p_i, self.L)

        triplet_cids = set()

        for new_ptype, r_new in enumerate(cfg.diameters / 2.0):
            valid, sol1, sol2 = solve_three_spheres(
                p_i, r_i, p_j, r_j, p_k, r_k, r_new)
            if not valid:
                continue
            for sol in (sol1, sol2):
                collision, touching = self._check_and_collect_touching(
                    sol, r_new, all_pos, all_rad)
                if not collision:
                    cid = self._new_cand_id()
                    self._candidate_set[cid]    = np.array(
                        [*(sol % self.L), r_new, float(len(touching))],
                        dtype=np.float32)
                    self._cand_info[cid]        = {
                        'triplet':   (i, j, k),
                        'new_ptype': new_ptype,
                        'touching':  touching,
                    }
                    self._cand_to_triplets[cid] = {(i, j, k)}
                    triplet_cids.add(cid)

        if triplet_cids:
            self._triplet_set[(i, j, k)] = triplet_cids

    def _init_sets(self):
        for i, j, k in combinations(range(self.n), 3):
            self._process_triplet(i, j, k)

    def _add_new_triplets(self, new_idx):
        # 只处理 j-k 也互相接触的三元组，避免候选集随 N 平方增长
        for j, k in combinations(range(new_idx), 2):
            if (min(j, k), max(j, k)) in self.contact_pairs:
                self._process_triplet(new_idx, j, k)

    def _filter_candidates(self, new_pos_wrapped, new_rad, new_idx):
        """
        过滤与刚放置粒子（new_idx）碰撞的候选，并为接触的候选更新
        touching 列表（追加 new_idx），同步更新 coordination 计数。
        """
        new_pos_np = np.array(new_pos_wrapped, dtype=np.float64)
        to_delete  = []
        for cid, feat in self._candidate_set.items():
            sol   = feat[:3].astype(np.float64)
            r_c   = float(feat[3])
            collision, touching = check_single_collision(
                sol, r_c, new_pos_np, new_rad, self.L, self.cfg.collision_tol)
            if collision:
                to_delete.append(cid)
            elif touching:
                feat[4] += 1.0
                info = self._cand_info[cid]
                if new_idx not in info['touching']:
                    info['touching'].append(new_idx)
        for cid in to_delete:
            self._remove_candidate(cid)

    def _remove_candidate(self, cid):
        if cid not in self._candidate_set:
            return
        del self._candidate_set[cid]
        del self._cand_info[cid]
        for triplet in self._cand_to_triplets.get(cid, set()):
            if triplet in self._triplet_set:
                self._triplet_set[triplet].discard(cid)
                if not self._triplet_set[triplet]:
                    del self._triplet_set[triplet]
        del self._cand_to_triplets[cid]

    def _update_current_candidates(self):
        """
        构建有序候选列表（配位数高优先，同配位数则大球优先）。
        超过 max_candidates 时截断。
        """
        cands = []
        for cid, feat in self._candidate_set.items():
            info = self._cand_info[cid]
            cands.append({
                'cid':       cid,
                'pos':       feat[:3].tolist(),
                'r':         float(feat[3]),
                'coord':     float(feat[4]),
                'new_ptype': info['new_ptype'],
                'touching':  info['touching'],
            })
        cands.sort(key=lambda c: (-c['coord'], -c['r']))
        if len(cands) > self.cfg.max_candidates:
            cands = cands[:self.cfg.max_candidates]
        self.current_candidates = cands
