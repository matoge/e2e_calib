CFG = dict(
    name         = 'ps_v9_repro_check',
    n_layers     = 2,
    img_size     = 64,
    in_channels  = 3,
    use_convnext = False,
    use_frustum  = True,
    epochs       = 100,
    batch_size   = 64,
    lr           = 0.001,
    lr_min       = 1e-06,
    val_fraction = 0.1,
    split_seed   = 42,
)
