# Grid+Depth experiment config
# Edit here to change model architecture, checkpoint, and training params.
# Each run saves to experiments/{name}/

CFG = dict(
    # ── Experiment ─────────────────────────────────────────────────────────
    name        = "v10_3layer_tfdecoder",   # → experiments/v10_3layer_tfdecoder/

    # ── Model ──────────────────────────────────────────────────────────────
    n_layers    = 3,        # 2 or 3  (coarse→fine  or  coarse→coarse→fine)
    self_first  = True,     # True = self-attn before cross-attn in each block
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
