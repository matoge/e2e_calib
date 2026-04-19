# Grid+Depth experiment config
# Edit here to change model architecture, checkpoint, and training params.
# Each run saves to experiments/{name}/

CFG = dict(
    # ── Experiment ─────────────────────────────────────────────────────────
    name        = "v17_4layer_temp05",   # → experiments/v17_4layer_temp05/

    # ── Model ──────────────────────────────────────────────────────────────
    n_layers      = 4,        # 2/3/4
    self_first    = False,    # True = self-attn before cross-attn in each block
    kv_self_attn  = False,    # True = image tokens self-attn before cross-attn
    cross_temp    = 0.5,      # cross-attn temperature (1.0=default, lower=sharper)
    img_size    = 64,
    in_channels = 3,

    # ── Training ───────────────────────────────────────────────────────────
    epochs      = 100,
    batch_size  = 64,
    lr          = 1e-3,
    lr_min      = 1e-6,
    train_size  = 8000,
    val_size    = 800,
    max_offset     = 16.0,
    random_depths  = True,    # True → all 3 depths random (v9+)
)
