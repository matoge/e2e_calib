"""Launch deformable-attention variants back-to-back.

Reuses `train_all_v2_mc.main()` with two CFG overrides:
  • vdef_sl  — single-level deformable (drop-in per block)
  • vdef_ml  — multi-level deformable (each block sees coarse+fine, with
                learnable resolution embedding)

Identical to `all_v3_mc` baseline on everything else.

Run:  python -u train_deform_sweep.py  [sl|ml|both]
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import sys
from train_all_v2_mc import main as train_main, CFG as BASE_CFG


COMMON = dict(
    n_layers      = 4,
    img_size      = 64,
    in_channels   = 3,
    use_convnext  = True,
    use_frustum   = True,
    epochs        = 200,
    batch_size    = 48,   # conservative vs baseline 64 — GPU shared with vLLM
    lr            = 1e-3,
    lr_min        = 1e-7,
    n_train_per   = 20000,
    n_val_per     = 500,
    split_seed    = 42,
)

SL_CFG = dict(COMMON, name='vdef_sl',
              deform_mode='sl', deform_n_points=4)
ML_CFG = dict(COMMON, name='vdef_ml',
              deform_mode='ml', deform_n_points=4)


def run(mode: str):
    if mode == 'sl':
        train_main(SL_CFG)
    elif mode == 'ml':
        train_main(ML_CFG)
    else:
        raise ValueError(mode)


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'both'
    if mode == 'both':
        print('=== SL training ===', flush=True)
        run('sl')
        print('=== ML training ===', flush=True)
        run('ml')
    else:
        run(mode)
