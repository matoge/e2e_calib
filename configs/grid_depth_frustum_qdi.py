# Grid+Depth experiment config
# Edit here to change model architecture, checkpoint, and training params.
# Each run saves to experiments/{name}/

CFG = dict(
    # ── Experiment ─────────────────────────────────────────────────────────
    name             = "ladder4f1_frustum_qdi",
    why              = "段 4f-1: 段 4f (格子 8px + 局所エンコーダ、7.03px) から、局所エンコーダの Q だけ 点のトークン (絶対位置入り) → 自分の点の (d, 強度) に変える。",
    loss             = "nll",       # "nll" = gaussian2d_nll 1 本、"split" = μ は距離、σ は μ を止めた NLL
    sigma_start      = 20,            # split のとき、このエポックまでは μ だけ
    clearml          = True,
    scene            = "depth_rgb_rd_grid8",
    n_obj            = (1, 1),        # lidar シーンの物体数の範囲
    # scene: "lidar" = make_image_and_points_lidar, "grid" = 旧 (物体 2 個、間隔 4-8px)
    use_convnext     = False,
    use_frustum      = True,
    frustum_q_from   = "di",          # 局所エンコーダの Q を (d, 強度) だけから作る
    use_intensity    = True,      # FrustumLocalEncoder が 4 列固定。人工データの強度は 0
    model            = "depth",     # "depth" = CalibNetDepth (toy_1006), "cnd2" = CalibNet2 (実データで使っているもの)
    grid_n           = 8,           # cnd2 の frustum セル数。64px で 16px セル = 実データ (256px, 16 セル) と同じ大きさ

    # ── Model ──────────────────────────────────────────────────────────────
    n_layers         = 3,
    self_first       = False,
    kv_self_attn     = False,
    cross_temp       = 1.0,
    cross_temp_start = 1.0,
    img_size    = 128,
    in_channels = 3,

    # ── Training ───────────────────────────────────────────────────────────
    epochs      = 80,
    batch_size  = 64,
    lr          = 1e-3,
    lr_min      = 1e-6,
    train_size  = 8000,
    val_size    = 800,
    max_offset     = 15.0,
    random_depths  = False,   # False → BG fixed at 1.0
    pool_size      = 32000,   # >0: サンプルを 1 回作って GPU に置き、毎エポック train_size 個を引く (ベンチ)
)
