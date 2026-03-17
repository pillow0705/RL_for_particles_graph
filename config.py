import numpy as np
import torch


class Config:
    # ---- 设备 ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 物理 / 环境参数（双元球）----
    target_N      = 100
    target_phi    = 0.64        # 双元球密堆积目标体积分数
    d_small       = 0.8         # 小球直径
    d_large       = 1.2         # 大球直径
    diameters     = np.array([0.8, 1.2])   # 只有两种直径
    collision_tol = 0.05
    edge_tol      = 0.05
    max_particles = 150

    # ---- 图结构参数 ----
    # 是否将"巧合形成"的次级四面体加入图
    # True:  放置新粒子时，检测所有满足两两接触条件的四元组，全部加入图
    # False: 只加入由生成三元组直接构成的主四面体（更简洁，但图信息更少）
    include_secondary_tets = True
    # 四面体类型：SSSS=0, SSSL=1, SSLL=2, SLLL=3, LLLL=4（= 大球数量）
    # 面类型：SSS=0, SSL=1, SLL=2, LLL=3（= 面上大球数量）
    # 特殊：face_type=4 表示"无面"（step 0 初始选择时使用）
    n_tet_types    = 5
    n_face_types   = 4
    max_candidates = 500

    # ---- Graph Transformer ----
    d_model   = 128
    n_heads   = 4
    n_layers  = 3
    d_ff      = 256
    d_hidden  = 128   # 候选打分 MLP 隐层宽度

    # ---- 训练超参数 ----
    num_workers            = 8
    num_iterations         = 25
    samples_per_iter       = 50
    train_epochs           = 2
    advantage_filter_ratio = 0.25
    batch_size             = 64
    lr                     = 5e-4
    gamma                  = 0.99
    temperature            = 5.0

    # ---- 输出 ----
    log_file     = "graph_v1.0_train_log.csv"
    ckpt_prefix  = "graph_v1.0"
    save_data    = False
    rollback_tol = 0.01

    # ---- 评测 ----
    eval_episodes    = 20
    eval_temperature = 1.0
    eval_report_file = "graph_v1.0_eval_report.txt"
    eval_conf_file   = "graph_v1.0_best_packing.conf"
