import numpy as np
import torch


class Config:
    # ---- 设备 ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 物理 / 环境参数（双元球）----
    target_N      = 100
    target_phi    = 0.64
    d_small       = 0.8
    d_large       = 1.2
    diameters     = np.array([0.8, 1.2])
    collision_tol = 0.05
    edge_tol      = 0.05
    max_particles = 150

    # ---- 图结构参数 ----
    # 粒子类型：0=小球，1=大球
    n_ptypes     = 2
    # 接触边类型：0=SS，1=SL，2=LL（= 两端粒子类型之和）
    n_edge_types = 3
    max_candidates = 500

    # ---- Graph Neural Network ----
    d_model  = 128
    n_heads  = 4      # 保留字段，当前 GNN 不使用
    n_layers = 3
    d_ff     = 256
    d_hidden = 128

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
    log_file     = "graph_v2.0_train_log.csv"
    ckpt_prefix  = "graph_v2.0"
    save_data    = False
    rollback_tol = 0.01

    # ---- 评测 ----
    eval_episodes    = 20
    eval_temperature = 1.0
    eval_report_file = "graph_v2.0_eval_report.txt"
    eval_conf_file   = "graph_v2.0_best_packing.conf"
