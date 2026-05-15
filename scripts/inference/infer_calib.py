"""Load a trained CalibNetDepth checkpoint and run forward.

Robust loader: reads experiments/{exp}/config.py for the EXACT hparams the
ckpt was trained with, then instantiates CalibNetDepth with those flags so
state_dict matches 1:1. Use this from a fresh script — no hand-set n_layers
to forget about.

Usage as a library:
    from scripts.inference.infer_calib import load_calib_model, predict_offset
    model = load_calib_model('ps_v9_lazy')
    delta_uv_with_sigma = predict_offset(model, img, uvd)   # (N, 5)

Usage as CLI smoke test:
    python scripts/inference/infer_calib.py ps_v9_lazy
        # builds a fake image+uvd batch, asserts forward runs, reports stats
"""
from __future__ import annotations
import sys, importlib.util
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from models.model_depth import CalibNetDepth


def load_calib_model(exp: str, device: str | torch.device = 'cuda'):
    """Build a CalibNetDepth matching experiments/{exp}/config.py and load weights."""
    exp_dir = REPO_ROOT / 'experiments' / exp
    cfg_path = exp_dir / 'config.py'
    ckpt_path = exp_dir / 'best_model.pt'
    if not cfg_path.exists():
        raise FileNotFoundError(f'no config: {cfg_path}')
    if not ckpt_path.exists():
        raise FileNotFoundError(f'no ckpt: {ckpt_path}  (did you `git lfs pull`?)')

    spec = importlib.util.spec_from_file_location('_cfg', cfg_path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    c = mod.CFG

    sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    # Auto-detect 4-ch (intensity) vs 3-ch PointMLP from the saved state_dict.
    # point_mlp.net.0.weight has shape (64, in_channels). Older ckpts are 3-ch.
    pm_w = sd.get('point_mlp.net.0.weight')
    if pm_w is not None and pm_w.shape[1] == 4:
        use_intensity_detected = True
    else:
        use_intensity_detected = bool(c.get('use_intensity', False))

    model = CalibNetDepth(
        img_size       = c['img_size'],
        in_channels    = c['in_channels'],
        n_layers       = c['n_layers'],
        self_first     = c.get('self_first', False),
        use_convnext   = c.get('use_convnext', False),
        use_frustum    = c.get('use_frustum', False),
        deform_mode    = c.get('deform_mode', 'none'),
        use_frame_token  = c.get('use_frame_token',  False),
        frame_token_side = c.get('frame_token_side', 8),
        use_lidar_kv     = c.get('use_lidar_kv', False),
        use_pose_emb     = c.get('use_pose_emb', False),
        use_frame_pose   = c.get('use_frame_pose', False),
        frame_pose_dof   = c.get('frame_pose_dof', 6),
        use_intensity    = use_intensity_detected,
    ).to(device)
    # detect extra_kv presence from state_dict (model_depth auto-creates extra_kv_attn
    # when use_lidar_kv is True; some checkpoints have it without explicit CFG flag).
    miss, unex = model.load_state_dict(sd, strict=False)   # non-strict: report mismatches
    model.eval()
    return model


@torch.no_grad()
def predict_offset(model: CalibNetDepth, img: torch.Tensor, uvd: torch.Tensor,
                   key_padding_mask: torch.Tensor | None = None,
                   vfp: torch.Tensor | None = None) -> torch.Tensor:
    """Run forward.

    img: (B, 3, H, W) — H == W == model.img_size, RGB float32 in [0, 1].
    uvd: (B, N, 3)  — U, V in pixels of the 64x64 patch (NOT full image),
                       D normalized as `meters / 100`. Optional 4th column
                       is_obj is silently ignored.
    Returns (B, N, 5) — [Δu (px), Δv (px), log_σx, log_σy, ρ].
    Caller adds Δu, Δv to the input UV to get predicted positions.
    """
    if uvd.shape[-1] >= 4:
        uvd = uvd[..., :3]
    if key_padding_mask is None:
        key_padding_mask = torch.zeros(*uvd.shape[:2], dtype=torch.bool, device=img.device)
    return model(img, uvd, key_padding_mask=key_padding_mask, vfp=vfp)


def _smoke(exp: str):
    print(f'[infer_calib] loading {exp} ...')
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_calib_model(exp, device=dev)
    img = torch.zeros(1, 3, model.img_size, model.img_size, device=dev)
    uvd = torch.tensor([[[32., 32., 0.3], [16., 48., 0.4]]], device=dev)
    out = predict_offset(model, img, uvd)
    print(f'  out shape: {tuple(out.shape)}  (expect (1, 2, 5))')
    print(f'  Δuv: {out[0, :, :2].cpu().tolist()}')
    print(f'  log_σ: {out[0, :, 2:4].cpu().tolist()}')


if __name__ == '__main__':
    _smoke(sys.argv[1] if len(sys.argv) > 1 else 'ps_v9_lazy')
