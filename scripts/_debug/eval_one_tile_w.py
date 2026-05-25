"""Eval δ-MSE on a single val tile under W=I / W=σ / W=learned, with a
small grid of (ωx, ωy) perturbations so we get a meaningful average.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from datasets.pandaset_full import PandaSetCalibDatasetFull, collate_full
from models.model_depth import CalibNetDepth
from scripts.ba.ba_torch import solve_pinhole, make_info_from_sigma_rho

CACHE = Path.home() / 'cache' / 'kamikado_v3_tiled'
CKPT  = REPO / 'experiments' / 'km_wv_wm_dgx2_n4_img128_8gpu_HEAD' / 'best_model.pt'
EXP_CFG_PATH = REPO / 'experiments' / 'km_wv_wm_dgx2_n4_img128_8gpu_HEAD' / 'config.py'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _cfg():
    src = EXP_CFG_PATH.read_text(); ns = {}; exec(src, ns, ns); return ns['CFG']

def _build(cfg):
    return CalibNetDepth(
        img_size=cfg['img_size'], in_channels=cfg['in_channels'],
        n_layers=cfg['n_layers'], self_first=cfg.get('self_first', False),
        use_convnext=cfg.get('use_convnext', True),
        use_frustum=cfg.get('use_frustum', True),
        deform_mode=cfg.get('deform_mode', 'sl'),
        convnext_n_blocks=cfg.get('convnext_n_blocks', 2),
        convnext_fine_d=cfg.get('convnext_fine_d', None),
        convnext_stem_d=cfg.get('convnext_stem_d', None),
        use_info_head=True,
    )

def _K(cfg, vfp):
    S = float(cfg['img_size']); B = vfp.shape[0]
    K = torch.zeros(B, 3, 3, device=vfp.device, dtype=vfp.dtype)
    K[:, 0, 0] = vfp; K[:, 1, 1] = vfp
    K[:, 0, 2] = S / 2; K[:, 1, 2] = S / 2; K[:, 2, 2] = 1.0
    return K

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--idx', type=int, required=True)
    ap.add_argument('--head-pt', type=Path, required=True)
    ap.add_argument('--model-pt', type=Path, default=None,
                    help='full model.pt from an unlock run (overrides backbone). '
                         'If omitted uses frozen baseline ckpt + head-pt overlay.')
    ap.add_argument('--n', type=int, default=64)
    ap.add_argument('--rng', type=float, default=0.30)
    args = ap.parse_args()

    cfg = _cfg()
    ds = PandaSetCalibDatasetFull(
        cache_dir=CACHE, split='val', img_size=cfg['img_size'],
        min_crop_px=cfg['min_crop_px'], max_crop_px=cfg['max_crop_px'],
        max_offset_m=0.0, max_rot_deg=0.0,
        oversample=1, grid_n=cfg.get('grid_n', 16),
        center_band=0.0, preload=False)
    print(f"[ev1] tile {args.idx} ({ds.fnames[args.idx]})  N_pert={args.n}  rng=±{args.rng}deg")

    rng = np.random.RandomState(7)
    wins = []
    for _ in range(args.n):
        ox = float(rng.uniform(-args.rng, args.rng))
        oy = float(rng.uniform(-args.rng, args.rng))
        win = ds.apply_perturbation_explicit(args.idx,
            np.zeros(3), np.array([0.0, oy, ox], dtype=np.float64))
        if win is not None: wins.append(win)
    print(f"[ev1] sampled {len(wins)} valid perturbations")

    batch = collate_full(wins)
    imgs, true_uvd, dist_uvd, pad_mask, vfp, bucket_uvd, bucket_valid, _ = [
        t.to(DEVICE) if torch.is_tensor(t) else t for t in batch]

    model = _build(cfg).to(DEVICE)
    if args.model_pt is not None:
        sd = torch.load(args.model_pt, map_location='cpu', weights_only=False)
        if isinstance(sd, dict) and ('state_dict' in sd or 'model' in sd):
            sd = sd.get('state_dict') or sd.get('model')
        model.load_state_dict(sd, strict=False)
        print(f"[ev1] loaded FULL model from {args.model_pt}")
    else:
        sd = torch.load(CKPT, map_location='cpu', weights_only=False)
        if isinstance(sd, dict) and ('state_dict' in sd or 'model' in sd):
            sd = sd.get('state_dict') or sd.get('model')
        model.load_state_dict(sd, strict=False)
        head_sd = torch.load(args.head_pt, map_location=DEVICE, weights_only=False)
        model.info_head.load_state_dict(head_sd)
        print(f"[ev1] frozen baseline backbone + info_head from {args.head_pt}")
    model.eval()

    captured = {}
    def _hk(_m, _i, _o): captured['q'] = _i[0].detach()
    h = model.info_head.mlp[0].register_forward_hook(_hk)
    point_in = torch.cat([dist_uvd[..., :3], dist_uvd[..., 4:5]], dim=-1)
    img_norm = imgs.float().div(255.0)
    with torch.no_grad():
        per_pt, _ = model(img_norm, point_in, key_padding_mask=pad_mask, vfp=vfp,
                          bucket_uvd=bucket_uvd, bucket_valid=bucket_valid)
    h.remove()
    q = captured['q']

    valid = ~pad_mask
    duv_oracle = (true_uvd[..., :2] - dist_uvd[..., :2]).detach()
    uv_obs = dist_uvd[..., :2].detach().clone()
    z_pts = dist_uvd[..., 2].detach().clone() * 100.0
    pad = ~valid
    if pad.any():
        z_pts[pad] = 1.0; uv_obs[pad] = 0.5*float(cfg['img_size']); duv_oracle[pad] = 0.0
    K_in = _K(cfg, vfp)
    DOFS = ['omega_x', 'omega_y']

    B, N = uv_obs.shape[:2]
    W_I = torch.eye(2, device=DEVICE).expand(B, N, 2, 2)
    with torch.no_grad():
        d_target, _ = solve_pinhole(uv_obs, duv_oracle, W_I, z_pts, K_in, DOFS,
                                     valid=valid, n_iter=3, damping=1e-3)

    duv_pred = per_pt[..., :2].detach()
    if pad.any(): duv_pred = duv_pred.clone(); duv_pred[pad] = 0.0
    sx = per_pt[..., 2].exp(); sy = per_pt[..., 3].exp(); rho = per_pt[..., 4]
    W_sigma = make_info_from_sigma_rho(sx, sy, rho).detach()
    with torch.no_grad():
        W_learn = model.info_head(q)

    def _mse(W):
        with torch.no_grad():
            d, _ = solve_pinhole(uv_obs, duv_pred, W, z_pts, K_in, DOFS,
                                  valid=valid, n_iter=3, damping=1e-3)
        return (d - d_target).pow(2).mean(dim=-1).mean().item()

    mse_zero = (torch.zeros_like(d_target) - d_target).pow(2).mean(dim=-1).mean().item()
    mse_I = _mse(W_I); mse_s = _mse(W_sigma); mse_l = _mse(W_learn)

    print(f"\n[ev1] δ-MSE on tile {args.idx} averaged over {len(wins)} (ωx,ωy):")
    print(f"  do-nothing   {mse_zero:.6e}  (RMSE/axis = {np.sqrt(mse_zero):.4f} deg)")
    print(f"  W = I        {mse_I:.6e}  (RMSE/axis = {np.sqrt(mse_I):.4f} deg)")
    print(f"  W = σ-head   {mse_s:.6e}  (RMSE/axis = {np.sqrt(mse_s):.4f} deg)")
    print(f"  W = learned  {mse_l:.6e}  (RMSE/axis = {np.sqrt(mse_l):.4f} deg)")
    print(f"\n  learned vs σ      ratio = {mse_l/mse_s:.3f}   ({'BETTER' if mse_l<mse_s else 'WORSE'})")
    print(f"  learned vs do-nothing ratio = {mse_l/mse_zero:.3f}  ({'BETTER' if mse_l<mse_zero else 'WORSE'})")

if __name__ == '__main__':
    main()
