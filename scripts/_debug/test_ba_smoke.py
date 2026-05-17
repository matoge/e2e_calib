"""Smoke test: run model on ONE PandaSet val sample, then aggregate
per-pt (du, dv, Σ_uv) → 6-DoF δ via solve_dofs("6dof_ext")."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from scripts.inference.infer_pipeline import make_ds
from scripts.inference.infer_calib import load_calib_model
from scripts.inference.infer_pipeline import infer_one
from scripts.ba.ba_multicam_corr import solve_dofs, _DOF_PRESETS


def main():
    exp = 'km_only_15deg_06m_n2_img128_fp16_dgx2'
    cache = '/cache/kamikado_v3_tiled'
    ds, c = make_ds(exp, cache, split='val', oversample=1)
    model = load_calib_model(exp)

    res = infer_one(model, ds, idx=0, seed=0)
    valid = res['valid']
    hyp_uv = res['hyp_uv'][valid]            # (N, 2)
    delta  = res['delta'][valid]              # (N, 2) = du, dv
    sx     = res['sigma_x'][valid]
    sy     = res['sigma_y'][valid]
    rho    = res['rho'][valid]
    z      = res['z'][valid]
    print(f'inferred: N={hyp_uv.shape[0]}  delta_mean={delta.mean(0)}  '
          f'sigma_x_mean={sx.mean():.2f}', flush=True)

    # par for solve_dofs is (du, dv, sx, sy, rho)
    par = np.stack([delta[:, 0], delta[:, 1], sx, sy, rho], axis=-1).astype(np.float32)

    # K from the dataset (full-image), then scale to crop-local px so it
    # matches hyp_uv's units.
    inst = ds._load_inst(0)
    K_full = np.asarray(inst['K_full'].numpy(), dtype=np.float32)
    box = ds._last_crop  # u0, v0, cs (in full image)
    S = c['img_size']
    scale = S / float(box['cs'])
    K = K_full.copy()
    K[0, 0] *= scale; K[1, 1] *= scale
    K[0, 2] = (K[0, 2] - box['u0']) * scale
    K[1, 2] = (K[1, 2] - box['v0']) * scale
    print(f'K (crop-local): fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}', flush=True)

    delta_6 = solve_dofs(hyp_uv, par, z, K,
                          _DOF_PRESETS['6dof_ext'], damping=1e-3)
    cov_6 = solve_dofs._last_cov
    print('=== 6-DoF δ (omega_x, omega_y, omega_z [deg], tx, ty, tz [m]) ===', flush=True)
    for name, val, std in zip(_DOF_PRESETS['6dof_ext'], delta_6, np.sqrt(np.diag(cov_6))):
        print(f'  {name:10s}  δ={val:+.4f}  σ={std:.4f}', flush=True)
    print(f'cov diag: {np.diag(cov_6)}', flush=True)
    print('OK', flush=True)


if __name__ == '__main__':
    main()
    # also dump GT pert_vec for δ comparison
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from scripts.inference.infer_pipeline import make_ds
    ds, c = make_ds('km_only_15deg_06m_n2_img128_fp16_dgx2', '/cache/kamikado_v3_tiled', split='val', oversample=1)
    sample = ds[0]
    if len(sample) >= 7:
        pert_vec = sample[6].numpy()
        print(f'\nGT pert_vec (tx,ty,tz,yaw,pitch,roll,dfx_pct,dfy_pct):')
        print(f'  {pert_vec}')
