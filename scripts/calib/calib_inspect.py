"""calib_inspect — 1 file 3 step calibration inspection pipeline.

Steps:
  1. proj       : 1 frame LiDAR projection overlay (CND2 model 不要)
  2. arrows     : 1 frame sliding-window per-point Δuv quiver overlay
  3. aggregate  : 1 sequence (frame N 個) を集約した per-cell quiver map

すべて ClearML task として記録。画像は report_image() で ClearML file
server 経由の URL を持つ → ブラウザの Debug Samples タブから click 可。

Usage:
  python scripts/calib/calib_inspect.py \\
    --step proj \\
    --cache /raid/home/hfunaya/cache_v5/tss4_v3_full_iter1kb4_yaw3 \\
    --seq 1686529656324 --frame 0 \\
    --name calib_proto_1f \\
    --why "Step 1: 1 frame raw projection of seq 1686529656324." \\
    --clearml

  python scripts/calib/calib_inspect.py \\
    --step arrows \\
    --cache /raid/home/hfunaya/cache_v5/tss4_v3_full_iter1kb4_yaw3 \\
    --ckpt-run cnd2_calib_6ds_50ep_0609_1027 \\
    --seq 1686529656324 --frame 0 \\
    --name calib_proto_arrows \\
    --why "Step 2: arrows on the same frame." --clearml

  python scripts/calib/calib_inspect.py \\
    --step aggregate \\
    --cache /raid/home/hfunaya/cache_v5/tss4_v3_full_iter1kb4_yaw3 \\
    --ckpt-run cnd2_calib_6ds_50ep_0609_1027 \\
    --seq 1686529656324 \\
    --name calib_proto_aggregate \\
    --why "Step 3: aggregate over the sequence." --clearml
"""
from __future__ import annotations
import argparse
import io
import sys
import time
from pathlib import Path

import numpy as np
import torch
import lmdb
import pickle
import struct
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


# -----------------------------------------------------------------------------
# Cache reader (FULL LMDB only — 1 inst = 1 parent frame)
# -----------------------------------------------------------------------------
HDR = '<Q'; HDRSZ = struct.calcsize(HDR)


def _load_inst_raw(env, key: str) -> dict:
    """Read 1 inst from FULL LMDB. Returns dict with jpg array + uv_full + z_cam + scene/frame meta + K/R/t."""
    with env.begin() as t:
        v = t.get(key.encode())
    if v is None:
        raise KeyError(key)
    hl = struct.unpack(HDR, v[:HDRSZ])[0]
    hdr = pickle.loads(v[HDRSZ:HDRSZ+hl])
    body = v[HDRSZ+hl:]
    ofs = hdr['offsets']

    def get(name):
        if name not in ofs: return None
        o, n, dt, sh = ofs[name]
        if dt is None:  # raw bytes (jpg)
            return body[o:o+n]
        return np.frombuffer(body[o:o+n], dtype=np.dtype(dt)).reshape(sh)

    jpg = get('jpg')
    img = np.asarray(Image.open(io.BytesIO(jpg)).convert('RGB'))
    return dict(
        img=img, uv_full=get('uv_full'), z_cam=get('z_cam'),
        pts=get('pts'), is_obj=get('is_obj'),
        K_full=get('K_full'),
        IH=hdr['IH'], IW=hdr['IW'],
        scene=hdr.get('scene', ''), frame=hdr.get('frame', -1),
        is_fisheye=hdr.get('is_fisheye', False),
    )


def _filter_keys_by_seq(env, seq_substring: str | None) -> list[str]:
    """Return all keys whose scene contains seq_substring (or all keys if None)."""
    out = []
    with env.begin() as t:
        for k, v in t.cursor():
            if k.startswith(b'__cubs__/'): continue
            hl = struct.unpack(HDR, v[:HDRSZ])[0]
            hdr = pickle.loads(v[HDRSZ:HDRSZ+hl])
            scene = str(hdr.get('scene', ''))
            frame = int(hdr.get('frame', -1))
            if seq_substring is None or seq_substring in scene:
                out.append((frame, k.decode()))
    out.sort()
    return [k for _, k in out]


# -----------------------------------------------------------------------------
# Step 1: projection overlay (raw, no model)
# -----------------------------------------------------------------------------
def render_projection(inst, title: str = '') -> np.ndarray:
    img = inst['img']
    IH, IW = inst['IH'], inst['IW']
    uv = inst['uv_full']
    z = inst['z_cam']
    in_b = (uv[:, 0] >= 0) & (uv[:, 0] < IW) & (uv[:, 1] >= 0) & (uv[:, 1] < IH)
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.imshow(img)
    sc = ax.scatter(uv[in_b, 0], uv[in_b, 1], c=z[in_b], s=0.6,
                    cmap='turbo', vmin=0.5, vmax=80.0, alpha=0.7)
    ax.set_xlim(0, IW); ax.set_ylim(IH, 0)
    plt.colorbar(sc, ax=ax, fraction=0.04, label='z (m)')
    ax.set_title(title or
        f'projection  scene={inst["scene"]}  frame={inst["frame"]}  '
        f'pts_in={int(in_b.sum())}/{len(uv)}', fontsize=9)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return np.array(Image.open(buf).convert('RGB'))


# -----------------------------------------------------------------------------
# Step 2 / 3: model forward + arrows (need CND2 model + dataset for sub-crop)
# -----------------------------------------------------------------------------
def _build_model_and_ds(ckpt_run: str, cache: Path):
    """Build CND2 model from ckpt + dataset for sub-crop carving.
    Imports are inside this function so step=proj doesn't need them."""
    from datasets.pandaset_full import PandaSetCalibDatasetFull, collate_full
    import scripts.eval.eval_shared_256x800 as _ev
    from scripts.eval.eval_shared_256x800 import _build_subwin, DEVICE
    from scripts._debug._cnd2_full_frame_stats import _build_cnd2_model
    run_dir = REPO / 'experiments' / ckpt_run
    _ev.CKPT = run_dir / 'best_model.pt'
    _ev.EXP_CFG_PATH = run_dir / 'config.py'
    cfg = _ev._load_cfg()
    ds = PandaSetCalibDatasetFull(
        cache_dir=cache, split='train',
        img_size=int(cfg.get('img_size', 128)),
        min_crop_px=int(cfg.get('min_crop_px', 128)),
        max_crop_px=int(cfg.get('max_crop_px', 512)),
        max_offset_m=0.0, max_rot_deg=0.0,
        oversample=1, grid_n=int(cfg.get('grid_n', 16)),
        center_band=0.0, preload=False,
    )
    model = _build_cnd2_model(cfg).to(DEVICE)
    sd = torch.load(_ev.CKPT, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and ('state_dict' in sd or 'model' in sd):
        sd = sd.get('state_dict') or sd.get('model')
    sd = {k.removeprefix('module.'): v for k, v in sd.items()}
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f'[calib_inspect] ckpt loaded: missing={len(miss)} unexpected={len(unexp)}')
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return cfg, ds, model, _build_subwin, collate_full, DEVICE


def _forward_subcrops(ds, inst_dict_for_ds_bypass, model, build_subwin, collate_full, device,
                      cs: int, stride: int, IW: int, IH: int, batch_size: int = 16,
                      cfg=None):
    """Build sliding sub-crops for a single inst, run model in batches.
    Returns lists of per-point (u_par, v_par, du, dv, sx, sy) in parent px."""
    from scripts.ba.ba_torch import make_info_from_sigma_rho
    inst = inst_dict_for_ds_bypass

    u_offs = list(range(0, max(1, IW - cs + 1), stride))
    v_offs = list(range(0, max(1, IH - cs + 1), stride))
    if u_offs[-1] + cs < IW: u_offs.append(IW - cs)
    if v_offs[-1] + cs < IH: v_offs.append(IH - cs)
    u0v0_list = [(u, v) for v in v_offs for u in u_offs]

    wins, used_uv = [], []
    for (u0, v0) in u0v0_list:
        w = build_subwin(ds, inst, np.zeros(3), np.zeros(3), u0=u0, v0=v0, cs=cs)
        if w is not None:
            wins.append(w); used_uv.append((u0, v0))

    use_intensity = bool(getattr(model, 'use_intensity', True))
    S = float(cfg.get('img_size', 128))
    scale = float(cs) / S

    U, V, DU, DV, SX, SY, WW = [], [], [], [], [], [], []
    for i in range(0, len(wins), batch_size):
        chunk = wins[i:i+batch_size]
        u0v0_chunk = used_uv[i:i+batch_size]
        moved = [t.to(device) if torch.is_tensor(t) else t for t in collate_full(chunk)]
        (imgs, true_uvd, dist_uvd, pad_mask, vfp,
         bucket_uvd, bucket_valid, _) = moved[:8]
        valid = ~pad_mask
        if use_intensity:
            point_in = torch.cat([dist_uvd[..., :3], dist_uvd[..., 4:5]], dim=-1)
        else:
            point_in = dist_uvd[..., :3]
        img_norm = imgs.float().div(255.0)
        with torch.no_grad():
            out = model(img_norm, point_in, dpose_R=None, vfp=vfp,
                        bucket_uvd=bucket_uvd, bucket_valid=bucket_valid,
                        key_padding_mask=pad_mask)
        per_pt = out[0] if isinstance(out, tuple) else out
        duv = per_pt[..., :2].detach().cpu().numpy()
        sx_b = per_pt[..., 2].exp().detach().cpu().numpy()
        sy_b = per_pt[..., 3].exp().detach().cpu().numpy()
        rho = per_pt[..., 4].clamp(-0.99, 0.99)
        W = make_info_from_sigma_rho(per_pt[..., 2].exp(),
                                       per_pt[..., 3].exp(), rho).detach()
        W_par = W / (scale * scale)
        valid_np = valid.cpu().numpy()
        true_np = true_uvd[..., :2].detach().cpu().numpy()
        W_np = W_par.cpu().numpy()
        for b in range(true_np.shape[0]):
            u0, v0 = u0v0_chunk[b]
            m = valid_np[b]
            u_par = true_np[b, m, 0] * scale + u0
            v_par = true_np[b, m, 1] * scale + v0
            du = duv[b, m, 0] * scale
            dv = duv[b, m, 1] * scale
            sx = sx_b[b, m] * scale
            sy = sy_b[b, m] * scale
            in_b = (u_par >= 0) & (u_par < IW) & (v_par >= 0) & (v_par < IH)
            U.append(u_par[in_b]); V.append(v_par[in_b])
            DU.append(du[in_b]); DV.append(dv[in_b])
            SX.append(sx[in_b]); SY.append(sy[in_b])
            WW.append(W_np[b, m][in_b])
    return (np.concatenate(U), np.concatenate(V),
            np.concatenate(DU), np.concatenate(DV),
            np.concatenate(SX), np.concatenate(SY),
            np.concatenate(WW))


def render_arrows_one_frame(inst, U, V, DU, DV, SX, SY,
                             quiver_scale=0.1, max_arrows=8000) -> np.ndarray:
    img = inst['img']
    IH, IW = inst['IH'], inst['IW']
    sigma = np.sqrt(SX ** 2 + SY ** 2)
    if len(U) > max_arrows:
        order = np.argsort(sigma)[:max_arrows]
        U, V, DU, DV, sigma = U[order], V[order], DU[order], DV[order], sigma[order]
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.imshow(img)
    sc = ax.quiver(U, V, DU, DV, sigma, cmap='turbo_r', angles='xy',
                   scale_units='xy', scale=quiver_scale,
                   width=0.0012, headwidth=4, headlength=5, alpha=0.85)
    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.01)
    cb.set_label('sigma (px)', fontsize=8)
    ax.set_xlim(0, IW); ax.set_ylim(IH, 0)
    ax.set_title(
        f'arrows  scene={inst["scene"]}  frame={inst["frame"]}  n={len(U)}',
        fontsize=9)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return np.array(Image.open(buf).convert('RGB'))


def aggregate_arrows(insts, ds, model, build_subwin, collate_full, device,
                      cs, stride, cfg, IW, IH, cell_px=32):
    """Run model on all insts, accumulate per-cell stats."""
    nW = (IW + cell_px - 1) // cell_px
    nH = (IH + cell_px - 1) // cell_px
    sum_du = np.zeros((nH, nW), np.float64); sum_dv = np.zeros((nH, nW), np.float64)
    sum_n = np.zeros((nH, nW), np.int64)
    sum_W = np.zeros((nH, nW, 2, 2), np.float64); sum_Wd = np.zeros((nH, nW, 2), np.float64)
    t0 = time.time()
    for i, inst in enumerate(insts):
        U, V, DU, DV, SX, SY, W = _forward_subcrops(
            ds, inst, model, build_subwin, collate_full, device,
            cs=cs, stride=stride, IW=IW, IH=IH, cfg=cfg)
        cu = (U // cell_px).astype(np.int64)
        cv = (V // cell_px).astype(np.int64)
        ok = (cu >= 0) & (cu < nW) & (cv >= 0) & (cv < nH)
        cu = cu[ok]; cv = cv[ok]; DU = DU[ok]; DV = DV[ok]; W = W[ok]
        np.add.at(sum_du, (cv, cu), DU)
        np.add.at(sum_dv, (cv, cu), DV)
        np.add.at(sum_n, (cv, cu), 1)
        d = np.stack([DU, DV], -1)
        Wd = np.einsum('mij,mj->mi', W, d)
        np.add.at(sum_W, (cv, cu), W)
        np.add.at(sum_Wd, (cv, cu), Wd)
        if (i + 1) % 5 == 0 or i == len(insts) - 1:
            print(f'  agg [{i+1}/{len(insts)}] {time.time()-t0:.0f}s', flush=True)

    n = sum_n.astype(np.float64)
    n_safe = np.where(n > 0, n, 1.0)
    mean_du = sum_du / n_safe; mean_dv = sum_dv / n_safe
    info_mean_du = np.zeros_like(mean_du); info_mean_dv = np.zeros_like(mean_dv)
    info_post_std = np.zeros_like(mean_du)
    for cv in range(nH):
        for cu in range(nW):
            if sum_n[cv, cu] < 1: continue
            try:
                C = np.linalg.inv(sum_W[cv, cu])
            except np.linalg.LinAlgError: continue
            d = C @ sum_Wd[cv, cu]
            info_mean_du[cv, cu] = d[0]; info_mean_dv[cv, cu] = d[1]
            info_post_std[cv, cu] = np.sqrt(max(C[0,0] + C[1,1], 0.0))
    return dict(
        nW=nW, nH=nH, cell_px=cell_px,
        sum_du=sum_du, sum_dv=sum_dv, sum_n=sum_n,
        sum_W=sum_W, sum_Wd=sum_Wd,
        mean_du=mean_du, mean_dv=mean_dv,
        info_mean_du=info_mean_du, info_mean_dv=info_mean_dv,
        info_post_std=info_post_std,
    )


def render_aggregate_pngs(stats, IW, IH, title='') -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nW, nH, cell_px = stats['nW'], stats['nH'], stats['cell_px']
    cells_u = (np.arange(nW) + 0.5) * cell_px
    cells_v = (np.arange(nH) + 0.5) * cell_px
    UU, VV = np.meshgrid(cells_u, cells_v)
    ok = stats['sum_n'] >= 5

    out_pngs = []
    for label, du, dv, color, cmap, cb_label in [
        ('hard',  stats['mean_du'],      stats['mean_dv'],
            np.sqrt(stats['mean_du']**2 + stats['mean_dv']**2), 'turbo', '|mean| (px)'),
        ('info',  stats['info_mean_du'], stats['info_mean_dv'],
            np.sqrt(stats['info_mean_du']**2 + stats['info_mean_dv']**2),
            'turbo', '|info-mean| (px)'),
        ('heat',  None, None,
            np.sqrt(stats['info_mean_du']**2 + stats['info_mean_dv']**2),
            'magma', '|info-mean| (px)'),
    ]:
        fig, ax = plt.subplots(figsize=(20, 10), dpi=130)
        ax.set_xlim(0, IW); ax.set_ylim(IH, 0); ax.set_aspect('equal')
        ax.set_facecolor('#0d0d0d')
        if du is not None:
            sc = ax.quiver(UU[ok], VV[ok], du[ok], dv[ok], color[ok],
                            cmap=cmap, angles='xy', scale_units='xy', scale=1.0,
                            width=0.0015, headwidth=4, headlength=5)
            cb = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.01)
            cb.set_label(cb_label, fontsize=8)
        else:
            masked = np.ma.masked_where(~ok, color)
            im = ax.imshow(masked, origin='upper', cmap=cmap, extent=(0, IW, IH, 0))
            cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
            cb.set_label(cb_label, fontsize=8)
        ax.set_title(f'{title}  ({label})', fontsize=9)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        out_pngs.append(np.array(Image.open(buf).convert('RGB')))
    return tuple(out_pngs)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--step', choices=('proj', 'arrows', 'aggregate'), required=True)
    ap.add_argument('--cache', type=Path, required=True,
                    help='FULL LMDB cache root (contains data.lmdb + meta.pt)')
    ap.add_argument('--seq', default=None,
                    help='filter by substring of scene field (e.g. 1686529656324). '
                         'None = all frames.')
    ap.add_argument('--frame', type=int, default=0,
                    help='within-seq frame index for step=proj/arrows')
    ap.add_argument('--ckpt-run', default=None,
                    help='experiments/<run>/best_model.pt (required for arrows/aggregate)')
    ap.add_argument('--cs', type=int, default=256)
    ap.add_argument('--stride', type=int, default=128)
    ap.add_argument('--cell-px', type=int, default=32)
    ap.add_argument('--max-arrows', type=int, default=8000)
    ap.add_argument('--quiver-scale', type=float, default=0.1)
    ap.add_argument('--clearml', action='store_true')
    ap.add_argument('--clearml-project', default='e2e_calib/calib_pipeline')
    ap.add_argument('--name', required=True)
    ap.add_argument('--why', required=True,
                    help='Required full-context reason. Min 200 chars. Must include: '
                         'motivation (なぜ今これ), prior result (前 step の何を踏んで), '
                         'expected outcome (何が見えたら次).')
    ap.add_argument('--parent-task', default=None,
                    help='ClearML task id of the prior step (for lineage). Sets '
                         'task.set_parent() so the pipeline chain is queryable.')
    ap.add_argument('--silver-stone', default=None,
                    help='Optional silver_stone slug that motivates this task; '
                         'echoed in ClearML comment for backlink.')
    ap.add_argument('--out', type=Path,
                    default=REPO / 'scripts/calib/_outputs')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if len(args.why.strip()) < 200:
        ap.error(
            f'--why must be at least 200 chars (got {len(args.why.strip())}). '
            'Calib_inspect tasks are read weeks later — abbreviated WHY makes '
            'them un-reproducible. Include motivation, prior result, expected '
            'outcome.')
    if args.step in ('arrows', 'aggregate') and not args.ckpt_run:
        ap.error(f'--ckpt-run required for step={args.step}')

    cml_logger = None
    cml_task = None
    if args.clearml:
        from clearml import Task
        cml_task = Task.init(project_name=args.clearml_project, task_name=args.name)
        # connect every arg as a hyperparameter so the task page's
        # Configuration tab shows the full reproduction recipe.
        cml_task.connect({k: (str(v) if isinstance(v, Path) else v)
                          for k, v in vars(args).items()})
        comment = args.why
        if args.silver_stone:
            comment = f"[silver_stone:{args.silver_stone}]\n\n{comment}"
        if args.parent_task:
            comment = f"[parent_task:{args.parent_task}]\n\n{comment}"
            try:
                cml_task.set_parent(args.parent_task)
            except Exception as e:
                print(f'[calib_inspect] WARN set_parent({args.parent_task}) failed: {e}')
        cml_task.set_comment(comment)
        cml_logger = cml_task.get_logger()
        print(f'[calib_inspect] ClearML task id={cml_task.id} name={cml_task.name}')
        if args.parent_task:
            print(f'[calib_inspect] parent_task={args.parent_task}')
        if args.silver_stone:
            print(f'[calib_inspect] silver_stone={args.silver_stone}')

    env = lmdb.open(str(args.cache / 'data.lmdb'), readonly=True, lock=False,
                     subdir=True, max_dbs=0)
    keys = _filter_keys_by_seq(env, args.seq)
    print(f'[calib_inspect] cache={args.cache.name}  seq={args.seq}  matched_keys={len(keys)}')
    if not keys:
        env.close(); raise SystemExit('no matching keys')

    if args.step == 'proj':
        key = keys[args.frame]
        inst = _load_inst_raw(env, key)
        png = render_projection(inst,
            title=f'proj  scene={inst["scene"]}  frame={inst["frame"]}  '
                  f'key={key}  cache={args.cache.name}')
        out_path = args.out / f'{args.name}_proj.png'
        Image.fromarray(png).save(out_path)
        print(f'[calib_inspect] wrote {out_path}')
        if cml_logger is not None:
            cml_logger.report_image(title='proj', series='frame_00',
                                     iteration=0, image=png)
            cml_task.upload_artifact('proj_png', out_path)
    elif args.step == 'arrows':
        key = keys[args.frame]
        inst = _load_inst_raw(env, key)
        # _build_model_and_ds opens the same LMDB via PandaSetCalibDatasetFull;
        # close our standalone env first so they don't collide on the same
        # environment handle (lmdb refuses two opens of the same dir in-proc).
        env.close()
        cfg, ds, model, build_subwin, collate_full, device = _build_model_and_ds(
            args.ckpt_run, args.cache)
        # bypass ds._load_inst by injecting our raw inst into a fake idx:
        # simplest is to find idx in ds.fnames matching key.
        idx = ds.fnames.index(key) if key in ds.fnames else 0
        inst_for_subwin = ds._load_inst(idx)
        U, V, DU, DV, SX, SY, _W = _forward_subcrops(
            ds, inst_for_subwin, model, build_subwin, collate_full, device,
            cs=args.cs, stride=args.stride, IW=inst['IW'], IH=inst['IH'], cfg=cfg)
        png = render_arrows_one_frame(inst, U, V, DU, DV, SX, SY,
                                       quiver_scale=args.quiver_scale,
                                       max_arrows=args.max_arrows)
        out_path = args.out / f'{args.name}_arrows.png'
        Image.fromarray(png).save(out_path)
        print(f'[calib_inspect] wrote {out_path}  n_pred={len(U)}')
        if cml_logger is not None:
            cml_logger.report_image(title='arrows', series=f'frame_{args.frame:04d}',
                                     iteration=0, image=png)
            cml_task.upload_artifact('arrows_png', out_path)
    elif args.step == 'aggregate':
        env.close()
        cfg, ds, model, build_subwin, collate_full, device = _build_model_and_ds(
            args.ckpt_run, args.cache)
        # gather all insts in seq
        sub_keys = keys
        IW = IH = None
        insts = []
        for k in sub_keys:
            if k not in ds.fnames: continue
            idx = ds.fnames.index(k)
            inst = ds._load_inst(idx)
            if IW is None:
                IW = int(inst['IW']); IH = int(inst['IH'])
            insts.append(inst)
        print(f'[calib_inspect] aggregate over {len(insts)} insts  IW×IH={IW}×{IH}')
        stats = aggregate_arrows(insts, ds, model, build_subwin, collate_full, device,
                                  cs=args.cs, stride=args.stride, cfg=cfg,
                                  IW=IW, IH=IH, cell_px=args.cell_px)
        png_hard, png_info, png_heat = render_aggregate_pngs(
            stats, IW, IH,
            title=f'aggregate  cache={args.cache.name}  seq={args.seq}  '
                  f'n={len(insts)}  cs={args.cs}  cell={args.cell_px}px')
        npz_path = args.out / f'{args.name}_aggregate.npz'
        np.savez(npz_path, **{k: v for k, v in stats.items() if isinstance(v, np.ndarray)})
        for label, png in [('hard', png_hard), ('info', png_info), ('heat', png_heat)]:
            p = args.out / f'{args.name}_aggregate_{label}.png'
            Image.fromarray(png).save(p)
            print(f'[calib_inspect] wrote {p}')
            if cml_logger is not None:
                cml_logger.report_image(title='aggregate', series=label,
                                         iteration=0, image=png)
        if cml_logger is not None:
            cml_task.upload_artifact('aggregate_npz', npz_path)
            cml_task.upload_artifact('aggregate_hard', args.out / f'{args.name}_aggregate_hard.png')
            cml_task.upload_artifact('aggregate_info', args.out / f'{args.name}_aggregate_info.png')
            cml_task.upload_artifact('aggregate_heat', args.out / f'{args.name}_aggregate_heat.png')

    try:
        env.close()
    except lmdb.Error:
        pass  # already closed by step branch (arrows/aggregate)
    if cml_task is not None:
        cml_task.close()


if __name__ == '__main__':
    main()
