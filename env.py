import numpy as np
from itertools import combinations

from config import Config
from physics import (pbc_diff_njit, solve_three_spheres_njit,
                     check_collision_njit, check_single_collision_njit)


class ConstructEnv:
    """
    双元球堆积环境（图结构版本）。

    核心思想
    --------
    传统方法将粒子坐标直接输入神经网络，依赖绝对坐标系。
    本版本改用"接触四面体图"作为状态表示：
      - 节点 = 四面体（由 4 个互相接触的粒子构成），类型由大球数量决定（0-4）
      - 边   = 两个四面体共享一个三角面，边类型由面上大球数量决定（0-3）
    该表示对平移和旋转不变，反映结构本质而非绝对位置。

    序列生成流程
    ------------
    Step 0（初始化）:
        空图（零向量）→ 神经网络选择初始四面体类型（0-4）→ 放置 4 个粒子。
    Step 1+:
        当前图 → 神经网络选择候选（开放面 × 新粒子类型）→ 放置 1 个粒子。
        每次放置后，检测所有以新粒子为顶点的接触四面体，全部加入图。

    候选的含义
    ----------
    每个候选 = 一个"开放面"（仅属于一个已有四面体的三角面）+ 一种新粒子类型。
    网络仅看到候选的图结构信息（face_type, new_tet_type），不看坐标。

    观测格式
    --------
    graph_obs = {
        'node_types': (N_tets,) int,   # 四面体类型 0-4
        'edge_src':   (N_edges,) int,
        'edge_dst':   (N_edges,) int,
        'edge_types': (N_edges,) int,  # 共享面类型 0-3
        'n_nodes': int, 'n_edges': int,
    }
    cand_obs = {
        'face_types':    (N_cands,) int,  # 0-3=真实面类型，4=无面（step 0）
        'new_tet_types': (N_cands,) int,  # 0-4，= face_type + new_ptype
        'n_cands': int,
    }
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        # 假设 S/L 各占 50%，用平均粒子体积推算周期盒子边长 L
        r_S      = cfg.d_small / 2
        r_L      = cfg.d_large / 2
        avg_vol  = 0.5 * (4/3 * np.pi * r_S**3) + 0.5 * (4/3 * np.pi * r_L**3)
        self.L   = (cfg.target_N * avg_vol / cfg.target_phi) ** (1/3)
        self.reset()

    # ------------------------------------------------------------------
    # 环境接口
    # ------------------------------------------------------------------
    def reset(self):
        # 物理状态
        self.pos            = []   # list of ndarray(3,)
        self.rad            = []   # list of float
        self.particle_types = []   # list of int: 0=S, 1=L
        self.n              = 0
        self.initialized    = False   # 初始四面体是否已放置

        # 图状态
        self.tet_node_types = []   # tet_idx → tet_type (0-4)
        self.edge_src       = []
        self.edge_dst       = []
        self.edge_types     = []
        self.face_to_tet    = {}   # sorted (i,j,k) → tet_idx

        # 接触对（物理意义：两粒子近似相切）
        self.contact_pairs  = set()   # set of (min_idx, max_idx)

        # 候选集合
        self._candidate_set    = {}   # cand_id → ndarray[x,y,z,r,coord]
        self._cand_graph       = {}   # cand_id → dict(triplet, new_ptype, face_type, new_tet_type)
        self._triplet_set      = {}   # (i,j,k) → set of cand_id
        self._cand_to_triplets = {}   # cand_id → set of (i,j,k)
        self._cand_counter     = 0
        self.current_candidates = []  # 当前步有序候选列表（dict格式）

        self._last_cand_stats = {}

        return self._get_graph_obs(), self._get_cand_obs_init()

    def step(self, action_idx):
        if not self.initialized:
            return self._step_init(action_idx)
        return self._step_normal(action_idx)

    def get_phi(self):
        box_vol   = self.L ** 3
        total_vol = sum((4/3) * np.pi * r**3 for r in self.rad)
        return total_vol / box_vol

    # ------------------------------------------------------------------
    # 观测构建
    # ------------------------------------------------------------------
    def _get_graph_obs(self):
        return {
            'node_types': np.array(self.tet_node_types, dtype=np.int64),
            'edge_src':   np.array(self.edge_src,       dtype=np.int64),
            'edge_dst':   np.array(self.edge_dst,       dtype=np.int64),
            'edge_types': np.array(self.edge_types,     dtype=np.int64),
            'n_nodes':    len(self.tet_node_types),
            'n_edges':    len(self.edge_src),
        }

    def _get_cand_obs_init(self):
        """Step 0: 5 种初始四面体类型作为候选（face_type=4 表示无面）"""
        return {
            'face_types':    np.full(5, 4, dtype=np.int64),
            'new_tet_types': np.arange(5, dtype=np.int64),
            'n_cands':       5,
        }

    def _get_cand_obs(self):
        n = len(self.current_candidates)
        if n == 0:
            return {'face_types': np.array([], dtype=np.int64),
                    'new_tet_types': np.array([], dtype=np.int64),
                    'n_cands': 0}
        return {
            'face_types':    np.array([c['face_type']    for c in self.current_candidates], dtype=np.int64),
            'new_tet_types': np.array([c['new_tet_type'] for c in self.current_candidates], dtype=np.int64),
            'n_cands':       n,
        }

    # ------------------------------------------------------------------
    # Step 0：初始四面体
    # ------------------------------------------------------------------
    def _step_init(self, tet_type):
        """
        tet_type ∈ {0,1,2,3,4} = 大球数量（即四面体类型）。
        随机打乱 4 个粒子的 S/L 排列，再用几何方法摆放使它们两两相切：
          p0 居中，p1 在 +x 方向与 p0 相切，
          p2 在 xy 平面内与 p0/p1 相切（余弦定理求坐标），
          p3 用三球解算器确定（两个解取第一个），无解时退化到 +z 方向。
        """
        cfg = self.cfg
        n_L = tet_type
        n_S = 4 - n_L
        ptypes = [0] * n_S + [1] * n_L
        np.random.shuffle(ptypes)   # 随机打乱，避免固定的 S/L 位置偏差
        radii = [cfg.d_large/2 if t == 1 else cfg.d_small/2 for t in ptypes]

        r0, r1, r2, r3 = radii
        p0 = np.array([self.L/2, self.L/2, self.L/2])
        p1 = p0 + np.array([r0 + r1, 0.0, 0.0])

        # 余弦定理：在 xy 平面求 p2，使 d(p0,p2)=r0+r2，d(p1,p2)=r1+r2
        d01 = r0 + r1; d02 = r0 + r2; d12 = r1 + r2
        x2  = (d02**2 - d12**2 + d01**2) / (2 * d01)
        y2  = np.sqrt(max(0.0, d02**2 - x2**2))
        p2  = p0 + np.array([x2, y2, 0.0])

        valid, p3, _ = solve_three_spheres_njit(p0, r0, p1, r1, p2, r2, r3)
        if not valid:
            p3 = p0 + np.array([0.0, 0.0, r0 + r3])

        self.pos            = [p % self.L for p in [p0, p1, p2, p3]]
        self.rad            = list(radii)
        self.particle_types = list(ptypes)
        self.n              = 4
        self.initialized    = True

        # 初始 4 粒子两两相切
        for i in range(4):
            for j in range(i + 1, 4):
                self.contact_pairs.add((i, j))

        # 建初始四面体图节点
        self._add_tet(0, 1, 2, 3)

        # 初始化候选集合
        self._init_sets()
        self._update_current_candidates()

        done = (self.n >= cfg.max_particles)
        cand_obs = self._get_cand_obs()
        if cand_obs['n_cands'] == 0 and not done:
            done = True
        reward = self.get_phi() if done else 0.0
        return (self._get_graph_obs(), cand_obs), reward, done

    # ------------------------------------------------------------------
    # Step 1+：选择候选
    # ------------------------------------------------------------------
    def _step_normal(self, action_idx):
        cfg = self.cfg

        if action_idx >= len(self.current_candidates):
            return (self._get_graph_obs(), self._get_cand_obs()), 0.0, True

        chosen    = self.current_candidates[action_idx]
        pos_new   = np.array(chosen['pos'])
        r_new     = chosen['r']
        new_ptype = chosen['new_ptype']
        cid       = chosen['cid']

        self._remove_candidate(cid)

        self.pos.append(pos_new % self.L)
        self.rad.append(r_new)
        self.particle_types.append(new_ptype)
        n_new = self.n
        self.n += 1

        # 找出所有与新粒子接触的旧粒子（相对容差：gap < tol * r_sum，与 physics.py 一致）
        all_pos = np.array(self.pos, dtype=np.float64)
        all_rad = np.array(self.rad, dtype=np.float64)
        touching = []
        for m in range(n_new):
            dm    = pbc_diff_njit(all_pos[m], all_pos[n_new], self.L)
            dist  = np.sqrt(np.sum(dm**2))
            r_sum = all_rad[m] + all_rad[n_new]
            gap   = dist - r_sum
            if gap < cfg.collision_tol * r_sum:
                touching.append(m)
                self.contact_pairs.add((min(m, n_new), max(m, n_new)))

        # 检测所有以 n_new 为第四顶点的新四面体。
        # 条件：(a,b,c,n_new) 中的 4 个粒子两两接触，即：
        #   a,b,c 都在 touching 中（均与 n_new 接触），
        #   且 a-b、a-c、b-c 也均在 contact_pairs 中（两两接触）。
        # 这保证了四面体的 6 条棱都是接触边，结构自洽。
        for abc in combinations(touching, 3):
            a, b, c = sorted(abc)
            if ((min(a,b), max(a,b)) in self.contact_pairs and
                (min(a,c), max(a,c)) in self.contact_pairs and
                (min(b,c), max(b,c)) in self.contact_pairs):
                self._add_tet(a, b, c, n_new)

        # 更新候选集合
        n_before_filter = len(self._candidate_set)
        self._filter_candidates(pos_new, r_new)
        n_filtered = n_before_filter - len(self._candidate_set)

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

    # ------------------------------------------------------------------
    # 图维护
    # ------------------------------------------------------------------
    def _add_tet(self, i, j, k, l):
        """
        将四面体 (i,j,k,l) 加入图，并自动连接共享面的已有四面体。

        四面体类型 = 4 个顶点中大球（type=1）的数量，范围 0-4。
        面类型     = 3 个顶点中大球的数量，范围 0-3。

        遍历 4 个三角面（C(4,3)=4）：
          - 若该面已在 face_to_tet 中：说明与已有四面体共享此面，建双向边。
          - 若未出现：记录为"开放面"，等待后续四面体来闭合。
        """
        particles = tuple(sorted([i, j, k, l]))
        tet_type  = sum(self.particle_types[p] for p in particles)
        tet_idx   = len(self.tet_node_types)
        self.tet_node_types.append(tet_type)

        for face_particles in combinations(particles, 3):
            face      = tuple(sorted(face_particles))
            face_type = sum(self.particle_types[p] for p in face)
            if face in self.face_to_tet:
                # 此面已属于另一四面体 → 两者相邻，加双向边
                other = self.face_to_tet[face]
                self.edge_src.append(tet_idx);  self.edge_dst.append(other)
                self.edge_types.append(face_type)
                self.edge_src.append(other);    self.edge_dst.append(tet_idx)
                self.edge_types.append(face_type)
            else:
                # 开放面：记录归属，等待被下一个四面体闭合
                self.face_to_tet[face] = tet_idx

    # ------------------------------------------------------------------
    # 候选集合增量维护（沿用原版逻辑，额外记录图信息）
    # ------------------------------------------------------------------
    def _new_cand_id(self):
        cid = self._cand_counter
        self._cand_counter += 1
        return cid

    def _process_triplet(self, i, j, k):
        """
        对三元组 (i,j,k) 生成所有合法候选点。

        对每种新粒子类型（S 或 L），调用三球解算器求出新球与三个已有球均相切的位置
        （最多两个解，分别位于三球所在平面的两侧）。通过碰撞检测后加入候选集合。

        图信息记录：
          face_type    = i,j,k 中大球数量（0-3），代表该开放面的类型
          new_tet_type = face_type + new_ptype（0-4），代表新生成四面体的类型

        PBC 处理：将 p_j, p_k 展开到以 p_i 为中心的坐标系，确保三球解算器
        看到的是连续空间中的坐标，而非被周期边界截断的坐标。
        """
        cfg     = self.cfg
        all_pos = np.array(self.pos,  dtype=np.float64)
        all_rad = np.array(self.rad,  dtype=np.float64)
        p_i, r_i = all_pos[i], all_rad[i]
        p_j, r_j = all_pos[j], all_rad[j]
        p_k, r_k = all_pos[k], all_rad[k]

        # PBC 展开：将 j、k 的坐标平移到 i 的近邻像（最小像约定）
        p_j = p_i + pbc_diff_njit(p_j, p_i, self.L)
        p_k = p_i + pbc_diff_njit(p_k, p_i, self.L)

        face_type    = (self.particle_types[i] + self.particle_types[j]
                        + self.particle_types[k])
        triplet_cids = set()

        # new_ptype=0 → S 球，new_ptype=1 → L 球
        for new_ptype, r_new in enumerate(cfg.diameters / 2.0):
            valid, sol1, sol2 = solve_three_spheres_njit(p_i, r_i, p_j, r_j,
                                                          p_k, r_k, r_new)
            if not valid:
                continue
            # sol1/sol2 是三球平面两侧的两个解，均需检查碰撞
            for sol in (sol1, sol2):
                collision, coord = check_collision_njit(
                    sol, r_new, all_pos, all_rad, self.L, cfg.collision_tol)
                if not collision:
                    cid          = self._new_cand_id()
                    new_tet_type = face_type + new_ptype
                    self._candidate_set[cid]    = np.array([*(sol % self.L),
                                                             r_new, coord],
                                                            dtype=np.float32)
                    self._cand_graph[cid]       = {
                        'triplet':      (i, j, k),
                        'new_ptype':    new_ptype,
                        'face_type':    face_type,
                        'new_tet_type': new_tet_type,
                    }
                    self._cand_to_triplets[cid] = {(i, j, k)}
                    triplet_cids.add(cid)

        if triplet_cids:
            self._triplet_set[(i, j, k)] = triplet_cids

    def _init_sets(self):
        for i, j, k in combinations(range(self.n), 3):
            self._process_triplet(i, j, k)

    def _add_new_triplets(self, new_idx):
        for j, k in combinations(range(new_idx), 2):
            self._process_triplet(new_idx, j, k)

    def _filter_candidates(self, new_pos, new_rad):
        new_pos_np = np.array(new_pos, dtype=np.float64)
        to_delete  = []
        for cid, feat in self._candidate_set.items():
            sol       = feat[:3].astype(np.float64)
            r_new     = float(feat[3])
            collision, touching = check_single_collision_njit(
                sol, r_new, new_pos_np, new_rad, self.L, self.cfg.collision_tol)
            if collision:
                to_delete.append(cid)
            elif touching:
                feat[4] += 1
        for cid in to_delete:
            self._remove_candidate(cid)

    def _remove_candidate(self, cid):
        if cid not in self._candidate_set:
            return
        del self._candidate_set[cid]
        del self._cand_graph[cid]
        for triplet in self._cand_to_triplets.get(cid, set()):
            if triplet in self._triplet_set:
                self._triplet_set[triplet].discard(cid)
                if not self._triplet_set[triplet]:
                    del self._triplet_set[triplet]
        del self._cand_to_triplets[cid]

    def _update_current_candidates(self):
        """
        将候选集合整理为有序列表，构建 current_candidates。

        排序规则：配位数高的优先（更容易嵌入现有结构），同配位数时大球优先
        （大球占据空间多，先放有助于提升体积分数）。
        超过 max_candidates 时截断，避免动作空间过大。
        """
        cands = []
        for cid, feat in self._candidate_set.items():
            info = self._cand_graph[cid]
            cands.append({
                'cid':         cid,
                'pos':         feat[:3].tolist(),
                'r':           float(feat[3]),
                'coord':       float(feat[4]),
                'triplet':     info['triplet'],
                'new_ptype':   info['new_ptype'],
                'face_type':   info['face_type'],
                'new_tet_type': info['new_tet_type'],
            })
        cands.sort(key=lambda c: (-c['coord'], -c['r']))
        if len(cands) > self.cfg.max_candidates:
            cands = cands[:self.cfg.max_candidates]
        self.current_candidates = cands
