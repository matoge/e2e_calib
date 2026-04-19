# Grid+Depth experiment config
# Edit here to change model architecture, checkpoint, and training params.
# Each run saves to experiments/{name}/

CFG = dict(
    # ── Experiment ─────────────────────────────────────────────────────────
    name             = "v18_3layer_randdepth_fix",   # → experiments/v18_3layer_randdepth_fix/

    # ── Model ──────────────────────────────────────────────────────────────
    n_layers         = 3,
    self_first       = False,
    kv_self_attn     = False,
    cross_temp       = 1.0,
    cross_temp_start = 1.0,
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
