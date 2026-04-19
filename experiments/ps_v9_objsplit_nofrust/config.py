CFG = dict(
    name         = 'ps_v9_objsplit_nofrust',
    n_layers     = 4,
    img_size     = 64,
    in_channels  = 3,
    use_convnext = True,
    use_frustum  = False,
    epochs       = 200,
    batch_size   = 64,
    lr           = 0.001,
    lr_min       = 1e-06,
    val_fraction = 0.1,
    split_seed   = 42,
)
