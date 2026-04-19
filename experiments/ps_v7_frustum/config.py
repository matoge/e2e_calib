CFG = dict(
    name         = "ps_v7_frustum",
    n_layers     = 3,
    img_size     = 64,
    in_channels  = 3,
    use_convnext = True,
    use_frustum  = True,
    epochs       = 100,
    batch_size   = 64,
    lr           = 1e-3,
    lr_min       = 1e-6,
)
