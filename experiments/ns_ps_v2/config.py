CFG = dict(
    name         = "ns_ps_v2",
    n_layers     = 4,
    img_size     = 64,
    in_channels  = 3,
    use_convnext = True,
    use_frustum  = True,
    epochs       = 100,
    batch_size   = 64,
    lr           = 1e-3,
    lr_min       = 1e-6,
)
