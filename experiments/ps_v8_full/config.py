CFG = dict(
    name          = "ps_v8_full",
    n_layers      = 4,
    img_size      = 64,
    in_channels   = 3,
    use_convnext  = True,
    use_frustum   = True,
    epochs        = 200,
    batch_size    = 64,
    lr            = 1e-3,
    lr_min        = 1e-6,
)
