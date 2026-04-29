"""Train CalibNetDepth on PandaSet (jitter-fix verification, 500 epochs)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import math, time, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from datasets.pandaset import collate_pandaset
from datasets.pandaset_lazy import PandaSetCalibDatasetLazy
from models.model_depth import CalibNetDepth
from models.model_cov import gaussian2d_nll
from torch.utils.data import DataLoader, Subset

torch.set_float32_matmul_precision("high")
DEVICE = torch.device("cuda")

CFG = dict(
    name          = "ps_v9_lazy",
    n_layers      = 2,
    img_size      = 64,
    in_channels   = 3,
    use_convnext  = False,
    use_frustum   = True,
    epochs        = 100,
    batch_size    = 64,
    lr            = 1e-3,
    lr_min        = 1e-6,
    val_fraction  = 0.1,
    split_seed    = 42,
)


def epoch_loop(model, loader, optimizer, scaler, train):
    model.train(train)
    total_nll, total_mse, n = 0.0, 0.0, 0
    obj_nll_s, obj_n = 0.0, 0
    bg_nll_s,  bg_n  = 0.0, 0
    obj_errs, bg_errs = [], []  # collect per-point L2 to compute median/p95
    for imgs, true_uvd, dist_uvd, pad_mask in loader:
        imgs     = imgs.to(DEVICE)
        true_uvd = true_uvd.to(DEVICE)
        dist_uvd = dist_uvd.to(DEVICE)
        pad_mask = pad_mask.to(DEVICE)
        gt       = true_uvd[..., :2] - dist_uvd[..., :2]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            params = model(imgs, dist_uvd[..., :3], key_padding_mask=pad_mask)
            valid  = ~pad_mask
            loss   = gaussian2d_nll(params[valid], gt[valid])
        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
        with torch.no_grad():
            err_all = (params[valid][..., :2].float() - gt[valid]).norm(dim=-1)
            mse    = err_all.mean().item()
            is_obj = valid & (dist_uvd[..., 3] > 0.5)
            is_bg  = valid & (dist_uvd[..., 3] < 0.5)
            # keep err tensors on GPU; cat+cpu only at epoch end (avoid per-batch sync)
            if is_obj.any():
                obj_nll_s += gaussian2d_nll(params[is_obj], gt[is_obj]).item(); obj_n += 1
                obj_errs.append((params[is_obj][..., :2].float() - gt[is_obj]).norm(dim=-1).detach())
            if is_bg.any():
                bg_nll_s  += gaussian2d_nll(params[is_bg],  gt[is_bg]).item();  bg_n  += 1
                bg_errs.append((params[is_bg][..., :2].float() - gt[is_bg]).norm(dim=-1).detach())
        total_nll += loss.item(); total_mse += mse; n += 1
    obj_nll = obj_nll_s / max(obj_n, 1)
    bg_nll  = bg_nll_s  / max(bg_n,  1)
    # full per-point distribution stats — gives mean/median/p95 that line up
    # with eyeballed vis (median ≈ "typical"); mean inflated by long-tail.
    # torch.quantile errors at numel > 2^24; cat on GPU (= no per-batch sync),
    # then single .cpu().numpy() at epoch end.
    import numpy as _np
    def _stats(errs):
        if not errs:
            return float('nan'), float('nan'), float('nan')
        e = torch.cat(errs).cpu().numpy()
        return float(e.mean()), float(_np.median(e)), float(_np.percentile(e, 95))
    obj_mse, obj_med, obj_p95 = _stats(obj_errs)
    bg_mse,  bg_med,  bg_p95  = _stats(bg_errs)
    return (total_nll / max(n,1), total_mse / max(n,1),
            obj_nll, bg_nll, obj_mse, bg_mse,
            obj_med, obj_p95, bg_med, bg_p95)


def main(cfg=None):
    c = cfg if cfg is not None else CFG
    exp_dir = Path("experiments") / c["name"]
    exp_dir.mkdir(parents=True, exist_ok=True)
    log_path = exp_dir / "train.log"
    log_path.write_text("")

    # Persist config for WebUI
    (exp_dir / "config.py").write_text(
        "CFG = dict(\n" +
        "".join(f"    {k:<13}= {v!r},\n" for k, v in c.items()) + ")\n")

    def log(msg):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"{ts}  {msg}"
        print(line)
        with open(log_path, "a") as f: f.write(line+"\n")

    cache = c.get('cache', '/mnt/nvme6t/e2e_calib_cache/pandaset_mc_s64_lazy')
    nw = c.get('num_workers', 16)
    pf = c.get('prefetch_factor', 4)
    kw = dict(num_workers=nw, pin_memory=True,
              persistent_workers=(nw > 0),
              prefetch_factor=pf if nw > 0 else None,
              collate_fn=collate_pandaset)
    # Merge train+val scenes, then random object-level split
    import random as _r
    log(f"loading cache {cache} (lazy)")
    ds_kw = dict(max_offset_m=c.get('max_offset_m', 0.20),
                  max_rot_deg=c.get('max_rot_deg', 0.5),
                  min_sub_px=c.get('min_sub_px', None),
                  max_sub_px=c.get('max_sub_px', None))
    log(f"  perturbation: ±{ds_kw['max_rot_deg']} deg / ±{ds_kw['max_offset_m']} m"
        f"   min_sub_px={ds_kw['min_sub_px']}")
    tr_full = PandaSetCalibDatasetLazy(cache, split='train', **ds_kw)
    log(f"train cache loaded: {len(tr_full)} instances  (cache_img={tr_full.cache_img})")
    va_full = PandaSetCalibDatasetLazy(cache, split='val', **ds_kw)
    log(f"val cache loaded: {len(va_full)} instances")
    from torch.utils.data import ConcatDataset
    full_ds = ConcatDataset([tr_full, va_full])
    idxs = list(range(len(full_ds)))
    _r.Random(c["split_seed"]).shuffle(idxs)
    n_val_obj = int(len(idxs) * c["val_fraction"])
    val_idxs, train_idxs = idxs[:n_val_obj], idxs[n_val_obj:]
    train_ds = Subset(full_ds, train_idxs)
    val_ds   = Subset(full_ds, val_idxs)
    log(f"object-level split: train={len(train_ds)} val={len(val_ds)} (seed={c['split_seed']})")

    # Optional per-epoch random subsample (RandomSampler with num_samples).
    # Each epoch draws this many random indices (without replacement when
    # num_samples ≤ len(train_ds)) → caps wall-time per epoch independent of
    # cache size. Used for many-short-experiment workflows on the 384 cache.
    train_size = c.get('train_size', None)
    val_size   = c.get('val_size',   None)
    from torch.utils.data import RandomSampler, SequentialSampler
    if train_size and train_size < len(train_ds):
        tr_sampler = RandomSampler(train_ds, replacement=False, num_samples=train_size)
        log(f"  train subsample: {train_size}/{len(train_ds)} per epoch")
        train_loader = DataLoader(train_ds, batch_size=c["batch_size"], sampler=tr_sampler, **kw)
    else:
        train_loader = DataLoader(train_ds, batch_size=c["batch_size"], shuffle=True,  **kw)
    if val_size and val_size < len(val_ds):
        # deterministic first-N (sequential) so val NLL is comparable across runs
        val_subset = Subset(val_ds, list(range(val_size)))
        log(f"  val subsample: {val_size}/{len(val_ds)} (deterministic first-N)")
        val_loader = DataLoader(val_subset, batch_size=c["batch_size"], shuffle=False, **kw)
    else:
        val_loader   = DataLoader(val_ds,   batch_size=c["batch_size"], shuffle=False, **kw)

    model = CalibNetDepth(img_size=c["img_size"], in_channels=c["in_channels"],
                          n_layers=c["n_layers"], self_first=c.get("self_first", False),
                          use_convnext=c.get("use_convnext", False),
                          use_frustum=c.get("use_frustum", False)).to(DEVICE)
    log(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=1e-3)
    epochs    = c["epochs"]
    lr_min_r  = c["lr_min"] / c["lr"]
    def lr_lambda(e):
        if e < 5: return (e+1)/5
        t = (e-5)/max(1,epochs-5)
        return lr_min_r + (1-lr_min_r)*0.5*(1+math.cos(math.pi*t))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.GradScaler(device="cuda")
    best_val  = float("inf")
    ckpt      = exp_dir / "best_model.pt"
    t0        = time.time()

    history = {'ep': [], 'tr_nll': [], 'va_nll': [], 'tr_mse': [], 'va_mse': []}

    # Optional curriculum: schedule = list of (start_ep, end_ep, rot_deg, t_m).
    # If set, sigma is clamped to that schedule at each epoch start. The dataset's
    # max_rot_deg / max_offset_m are mutated in place — but workers cache the dataset
    # at fork time, so persistent_workers=False is required for the curriculum to
    # take effect mid-run. We restart workers per stage transition by toggling
    # train_loader.dataset attributes and recreating the loader.
    sigma_schedule = c.get('sigma_schedule', None)
    def _sigma_for_epoch(ep):
        if not sigma_schedule:
            return None
        for s in sigma_schedule:
            if s[0] <= ep <= s[1]:
                return s[2], s[3]
        return sigma_schedule[-1][2], sigma_schedule[-1][3]

    cur_sigma = None
    for epoch in range(1, epochs+1):
        if sigma_schedule is not None:
            new_sigma = _sigma_for_epoch(epoch)
            if new_sigma != cur_sigma:
                rot, tm = new_sigma
                tr_full.max_rot_deg = rot;  tr_full.max_offset_m = tm
                va_full.max_rot_deg = rot;  va_full.max_offset_m = tm
                # rebuild loaders with workers=0 OR fresh persistent workers
                # so the in-memory dataset state propagates.
                train_loader = DataLoader(train_ds, batch_size=c["batch_size"],
                                          shuffle=True, **kw)
                val_loader   = DataLoader(val_ds,   batch_size=c["batch_size"],
                                          shuffle=False, **kw)
                cur_sigma = new_sigma
                log(f"  curriculum: sigma → rot={rot} t={tm}")
        (tr_nll, tr_mse, tr_obj, tr_bg, tr_obj_mse, tr_bg_mse,
         tr_obj_med, tr_obj_p95, tr_bg_med, tr_bg_p95) = epoch_loop(
            model, train_loader, optimizer, scaler, True)
        with torch.no_grad():
            (va_nll, va_mse, va_obj, va_bg, va_obj_mse, va_bg_mse,
             va_obj_med, va_obj_p95, va_bg_med, va_bg_p95) = epoch_loop(
                model, val_loader, optimizer, scaler, False)
        scheduler.step()
        history['ep'].append(epoch)
        history['tr_nll'].append(tr_nll); history['va_nll'].append(va_nll)
        history['tr_mse'].append(tr_mse); history['va_mse'].append(va_mse)
        log(f"[{epoch:3d}/{epochs}]  "
            f"train nll={tr_nll:+.3f}(obj={tr_obj:+.3f} bg={tr_bg:+.3f}) "
            f"mse={tr_mse:.2f}(obj={tr_obj_mse:.2f}/m{tr_obj_med:.2f}/95p{tr_obj_p95:.1f} "
            f"bg={tr_bg_mse:.2f}/m{tr_bg_med:.2f}/95p{tr_bg_p95:.1f})  "
            f"val nll={va_nll:+.3f}(obj={va_obj:+.3f} bg={va_bg:+.3f}) "
            f"mse={va_mse:.2f}(obj={va_obj_mse:.2f}/m{va_obj_med:.2f}/95p{va_obj_p95:.1f} "
            f"bg={va_bg_mse:.2f}/m{va_bg_med:.2f}/95p{va_bg_p95:.1f})  "
            f"lr={scheduler.get_last_lr()[0]:.2e}  tot={(time.time()-t0)/60:.1f}min")
        if va_nll < best_val:
            best_val = va_nll
            torch.save(model.state_dict(), ckpt)
            log(f"  ↳ saved (val_nll={best_val:.4f})")

    log(f"Best val NLL: {best_val:.4f}  |  time: {(time.time()-t0)/60:.1f}min")

    # ── curves ──
    vis_dir = exp_dir / "vis"; vis_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history['ep'], history['tr_nll'], label='train'); axes[0].plot(history['ep'], history['va_nll'], label='val')
    axes[0].set_title('NLL'); axes[0].set_xlabel('epoch'); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(history['ep'], history['tr_mse'], label='train'); axes[1].plot(history['ep'], history['va_mse'], label='val')
    axes[1].set_title('MSE (px)'); axes[1].set_xlabel('epoch'); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(vis_dir / "curves.png", dpi=100); plt.close(fig)

    # ── vis (delegated to vis_ps so BB overlay stays in sync) ──
    from scripts.visualization.vis_ps import main as vis_main
    vis_main(c["name"], n_vis=48, cache=cache)
    n_vis = 48
    log(f"Saved {n_vis} vis → {vis_dir}")

    # ── report.html (overwritten by vis_ps; rebuild to include local best_nll) ──
    vis_imgs = "\n".join(f'      <img src="vis/val_{i:02d}.png" style="width:200px; margin:4px; border:1px solid #333;">' for i in range(n_vis))
    report = f"""<!doctype html><meta charset="utf-8">
<title>{c['name']} — report</title>
<style>body{{background:#0d1117;color:#eee;font-family:sans-serif;padding:20px;}} pre{{background:#161b22;padding:10px;border-radius:6px;}} h2{{border-bottom:1px solid #333;padding-bottom:4px;}}</style>
<h1>{c['name']}</h1>
<p>best val NLL: <b>{best_val:.4f}</b></p>
<pre>{chr(10).join(f'{k:<15}= {v!r}' for k,v in c.items())}</pre>
<h2>Curves</h2>
<img src="vis/curves.png" style="max-width:800px;">
<h2>Val samples</h2>
<div style="display:flex;flex-wrap:wrap;">
{vis_imgs}
</div>
"""
    (exp_dir / "report.html").write_text(report)
    log("Report saved → report.html")


if __name__ == "__main__":
    # Avoid fork-after-thread deadlock: ClearML's reporter thread holds locks
    # at fork time, and DataLoader workers inherit those locks but not the
    # holder thread → futex wait forever. forkserver spawns a clean server
    # process up-front; all workers are forked from that thread-less server.
    import multiprocessing as _mp
    try:
        _mp.set_start_method('forkserver', force=True)
    except RuntimeError:
        pass
    import argparse, copy
    ap = argparse.ArgumentParser()
    ap.add_argument('--name')
    ap.add_argument('--cache')
    ap.add_argument('--rot-deg', type=float, default=None,
                    help='extrinsic perturbation half-range (deg per axis)')
    ap.add_argument('--t-m',     type=float, default=None,
                    help='extrinsic perturbation half-range (m per axis)')
    ap.add_argument('--epochs',  type=int,   default=None)
    ap.add_argument('--workers', type=int,   default=None,
                    help='dataloader workers (default 16; sakurai2 may hang at 16, try 4)')
    ap.add_argument('--min-sub-px', type=int, default=None,
                    help='min sub-window side in cache px (v2 384-cache: 128 recommended)')
    ap.add_argument('--max-sub-px', type=int, default=None,
                    help='max sub-window side. v2 384-cache: pass 192 to match v1 difficulty '
                         '(cap zoom-out at 3x like v1 192-cache).')
    ap.add_argument('--batch-size', type=int, default=None,
                    help='train/val batch size (default 64; bump to 256 for v2 cache to feed GPU)')
    ap.add_argument('--prefetch-factor', type=int, default=None,
                    help='DataLoader prefetch factor (default 4)')
    ap.add_argument('--n-layers', type=int, default=None,
                    help='cross-attn layers (default 2; all_v1 used 4)')
    ap.add_argument('--convnext', action='store_true',
                    help='use ConvNeXt backbone (=all_v1 1.62M config)')
    ap.add_argument('--train-size', type=int, default=None,
                    help='per-epoch random subsample of train set (caps wall-time). '
                         'e.g. 20000 → 78 batches @ bs=256 → ~14s/ep')
    ap.add_argument('--val-size', type=int, default=None,
                    help='deterministic first-N val subsample for fast eval')
    ap.add_argument('--curriculum', default=None,
                    help='sigma curriculum spec, semicolon-separated stages '
                         'e.g. "1-25:0.5,0.05;26-60:1.0,0.10;61-100:2.0,0.20"')
    ap.add_argument('--clearml', action='store_true')
    ap.add_argument('--why',     default='')
    args = ap.parse_args()
    cfg = dict(CFG)
    if args.name:    cfg['name'] = args.name
    if args.cache:   cfg['cache'] = args.cache
    if args.rot_deg is not None: cfg['max_rot_deg']  = args.rot_deg
    if args.t_m     is not None: cfg['max_offset_m'] = args.t_m
    if args.epochs  is not None: cfg['epochs'] = args.epochs
    if args.workers is not None: cfg['num_workers'] = args.workers
    if args.min_sub_px is not None: cfg['min_sub_px'] = args.min_sub_px
    if args.max_sub_px is not None: cfg['max_sub_px'] = args.max_sub_px
    if args.batch_size is not None: cfg['batch_size'] = args.batch_size
    if args.prefetch_factor is not None: cfg['prefetch_factor'] = args.prefetch_factor
    if args.n_layers is not None: cfg['n_layers'] = args.n_layers
    if args.convnext: cfg['use_convnext'] = True
    if args.train_size is not None: cfg['train_size'] = args.train_size
    if args.val_size   is not None: cfg['val_size']   = args.val_size
    if args.curriculum:
        # parse "1-25:0.5,0.05;26-60:1.0,0.10;..."
        stages = []
        for part in args.curriculum.split(';'):
            rng, sigmas = part.strip().split(':')
            ep_lo, ep_hi = (int(x) for x in rng.split('-'))
            rot, tm = (float(x) for x in sigmas.split(','))
            stages.append((ep_lo, ep_hi, rot, tm))
        cfg['sigma_schedule'] = stages

    # optional ClearML reporting (with rich context if --clearml + --why)
    cml_task = None
    if args.clearml:
        from scripts.util.clearml_context import init_with_context, write_retrospective
        cml_task = init_with_context(
            project='e2e_calib/calib', name=cfg['name'], cfg=cfg,
            why=args.why, baseline={'name':'ps_v9_lazy', 'metric':'val_nll', 'value':1.8141})
    main(cfg=cfg)
