"""Post-process multi-frame pose fusion from a `pose_dump_ep*.pt` file.

Implements the 3 fusion rules from `docs/2026-08-31_nuscenes-calibration.md`:

  sum      δ̄ = (Σ Hᵢ)⁻¹ Σ Hᵢ δᵢ                  (fixed-effect inverse-variance pool)
  gate3    two-pass χ² validation gate:            c_i = (δᵢ − δ̄)ᵀ Hᵢ (δᵢ − δ̄) / 6
             drop c_i > 3, refit, repeat once.
  CI       covariance intersection, uniform 1/F    (baseline the report shows is
                                                     equivalent to `sum` for the
                                                     point estimate but under-
                                                     confident on covariance).

Also reports over-dispersion  k = χ²/6  measured on the fused residuals.

Input `pose_dump_ep050.pt` contains
    delta_pred (N, 6)  — per-frame 6-DOF estimate in whichever units train_cnd2_ddp
                          emitted (rot in RADIANS via so3 log-map, trans in metres).
    delta_gt   (N, 6)  — per-frame GT.
    H          (N,6,6) — per-frame information matrix (fp64).

Usage:
    python scripts/eval/frame_fusion.py \
        experiments/<exp>/pose_dump_ep050.pt \
        --out docs/assets/<slug>/frame_fusion.json \
        --plot docs/assets/<slug>/frame_fusion.png
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch


def _pool(H, dp):
    """Inverse-variance pool.  H: (F,6,6)  dp: (F,6)  → δ̄ (6,), H_sum (6,6)."""
    H_sum = H.sum(0)
    b_sum = np.einsum('fij,fj->i', H, dp)
    delta_bar = np.linalg.solve(H_sum, b_sum)
    return delta_bar, H_sum


def _chi2_per_frame(H, dp, delta_bar):
    e = dp - delta_bar[None, :]
    return np.einsum('fi,fij,fj->f', e, H, e) / 6.0


def fuse(H, dp, mode='gate3', gate_c=3.0, gate_iters=2):
    """Return (δ̄, H_sum, keep_mask). All fp64."""
    keep = np.ones(len(H), dtype=bool)
    if mode == 'sum':
        d, Hs = _pool(H, dp);  return d, Hs, keep
    if mode == 'CI':
        # uniform 1/F: δ̄ from H_sum stays the same (1/F cancels through H⁻¹b)
        # but H_sum becomes H_avg/F to be conservative.
        d, Hs = _pool(H, dp)
        return d, Hs / len(H), keep
    if mode == 'gate3':
        for _ in range(gate_iters):
            if keep.sum() < 2: break
            d, Hs = _pool(H[keep], dp[keep])
            c = _chi2_per_frame(H, dp, d)
            new_keep = keep & (c <= gate_c)
            if new_keep.sum() == keep.sum(): break
            if new_keep.sum() < 2: break
            keep = new_keep
        d, Hs = _pool(H[keep], dp[keep])
        return d, Hs, keep
    raise ValueError(f'unknown mode {mode!r}')


def _err_deg_mm(delta, gt):
    """delta / gt already come from _ba_pose_loss where rot is in DEG and
    translation in metres. Report L1-mean per component (matches train.log's
    `rot_err = (delta_pred - delta_gt).abs().mean()`), so single-frame errors
    line up with the training log."""
    rot = np.abs(delta[:3] - gt[:3]).mean()                  # deg
    t_mm = np.abs(delta[3:] - gt[3:]).mean() * 1000.0        # mm
    return float(rot), float(t_mm)


def sweep(H, dp, dg, F_list=(1, 2, 4, 8, 16, 32), mode='gate3', seed=0):
    """Non-overlapping windows of size F; return per-F medians / p90 / chi2/k.

    The trainer uses share_pert per WINDOW (a 40-tile crop-grid), so each
    "frame" in the dump has its own ε_gt. To fuse F frames as if they saw the
    same rig, we work in residual space:  r_i = δ_pred_i − δ_gt_i.  A perfect
    net gives r_i = 0; N frames pooled inverse-variance yield ‖r̄‖ that
    shrinks like 1/√N. This matches the nuScenes report's setup (its per-
    frame δ_gt was also generated fresh — the fusion is over per-frame
    residuals, not raw pose estimates)."""
    N = len(H)
    rng = np.random.default_rng(seed)
    # residual space (pose the net FAILED to recover)
    r = dp - dg
    rows = []
    for F in F_list:
        if F > N: break
        n_windows = N // F
        rots, ts, chi2s, F_effs = [], [], [], []
        perm = rng.permutation(N)
        Hp, rp = H[perm], r[perm]
        for w in range(n_windows):
            sl = slice(w * F, (w + 1) * F)
            d, Hs, keep = fuse(Hp[sl], rp[sl], mode=mode)
            # d is now the fused RESIDUAL (should be near zero); its magnitude
            # IS the error, no reference subtraction needed.
            rot = float(np.abs(d[:3]).mean())
            t_mm = float(np.abs(d[3:]).mean() * 1000.0)
            rots.append(rot); ts.append(t_mm); F_effs.append(int(keep.sum()))
            chi2s.append(d @ Hs @ d / 6.0)
        rots = np.asarray(rots); ts = np.asarray(ts); chi2s = np.asarray(chi2s)
        rows.append(dict(
            F=int(F), n=int(n_windows),
            rot_med=float(np.median(rots)),
            rot_p90=float(np.percentile(rots, 90)) if len(rots) >= 10 else float('nan'),
            rot_max=float(np.max(rots)),
            t_med=float(np.median(ts)),
            t_p90=float(np.percentile(ts, 90)) if len(ts) >= 10 else float('nan'),
            t_max=float(np.max(ts)),
            chi2r_med=float(np.median(chi2s)),
            F_eff=float(np.mean(F_effs)),
            k=float(np.median(chi2s)),
        ))
    return rows


def load_dump(path):
    d = torch.load(path, map_location='cpu', weights_only=False)
    dp = d['delta_pred'].reshape(-1, 6).double().numpy()
    dg = d['delta_gt'].reshape(-1, 6).double().numpy()
    H  = d['H'].reshape(-1, 6, 6).double().numpy()
    eig = np.linalg.eigvalsh(H).min(-1)
    mask = eig > 1e-8
    return dp[mask], dg[mask], H[mask]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dump', type=str, help='experiments/<exp>/pose_dump_epNNN.pt')
    ap.add_argument('--out', type=str, default='')
    ap.add_argument('--plot', type=str, default='')
    ap.add_argument('--F', type=str, default='1,2,4,8,16,32',
                    help='comma-separated frame counts')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    dp, dg, H = load_dump(args.dump)
    print(f'[frame_fusion] N={len(H)} valid frames  (pos-def H filter)', flush=True)

    F_list = [int(x) for x in args.F.split(',')]
    results = {mode: sweep(H, dp, dg, F_list=F_list, mode=mode, seed=args.seed)
               for mode in ['sum', 'gate3', 'CI']}

    # Print summary — gate3 is the recommended row
    print()
    print(f'{"mode":>6} {"F":>4} {"n":>4} {"rot_med°":>10} {"rot_p90°":>10} '
          f'{"rot_max°":>10} {"t_med mm":>9} {"t_p90 mm":>9} {"chi2/6":>7} {"F_eff":>6}')
    for mode in ['sum', 'gate3', 'CI']:
        for r in results[mode]:
            print(f'{mode:>6} {r["F"]:>4} {r["n"]:>4} '
                  f'{r["rot_med"]:>10.4f} {r["rot_p90"]:>10.4f} {r["rot_max"]:>10.4f} '
                  f'{r["t_med"]:>9.2f} {r["t_p90"]:>9.2f} '
                  f'{r["chi2r_med"]:>7.2f} {r["F_eff"]:>6.2f}')

    if args.out:
        p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(results, indent=2))
        print(f'\nwrote {p}')

    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        C = {'sum': '#5B9BD5', 'gate3': '#ED7D31', 'CI': '#70AD47'}
        plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                             'axes.grid': True, 'grid.alpha': 0.3,
                             'grid.linestyle': '--',
                             'axes.spines.top': False, 'axes.spines.right': False})
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), tight_layout=True)

        # (a) rot vs F
        ax = axes[0]
        for mode in ['sum', 'gate3', 'CI']:
            Fs = [r['F'] for r in results[mode]]
            ax.plot(Fs, [r['rot_med'] for r in results[mode]], 'o-',
                    color=C[mode], lw=2, ms=5, label=mode)
        # ideal 1/sqrt(F) scaling from F=1
        r1 = results['gate3'][0]['rot_med']
        Fs_all = [r['F'] for r in results['gate3']]
        ax.plot(Fs_all, [r1/np.sqrt(F) for F in Fs_all], 'k:', alpha=0.5,
                lw=1.5, label='ideal 1/√F')
        ax.set_xscale('log', base=2); ax.set_yscale('log')
        ax.set_xlabel('F (frames pooled)'); ax.set_ylabel('rot residual (deg)  [log]')
        ax.set_title('(a) rot vs F')
        ax.legend()

        # (b) trans vs F
        ax = axes[1]
        for mode in ['sum', 'gate3', 'CI']:
            Fs = [r['F'] for r in results[mode]]
            ax.plot(Fs, [r['t_med'] for r in results[mode]], 'o-',
                    color=C[mode], lw=2, ms=5, label=mode)
        t1 = results['gate3'][0]['t_med']
        ax.plot(Fs_all, [t1/np.sqrt(F) for F in Fs_all], 'k:', alpha=0.5,
                lw=1.5, label='ideal 1/√F')
        ax.set_xscale('log', base=2); ax.set_yscale('log')
        ax.set_xlabel('F (frames pooled)'); ax.set_ylabel('trans residual (mm)  [log]')
        ax.set_title('(b) trans vs F')
        ax.legend()

        # (c) chi2/6 vs F  (over-dispersion measurement)
        ax = axes[2]
        for mode in ['sum', 'gate3', 'CI']:
            Fs = [r['F'] for r in results[mode]]
            ax.plot(Fs, [r['chi2r_med'] for r in results[mode]], 'o-',
                    color=C[mode], lw=2, ms=5, label=mode)
        ax.axhline(1.0, ls='--', color='gray', alpha=0.6, label='calibrated (χ²/6 = 1)')
        ax.set_xscale('log', base=2)
        ax.set_xlabel('F (frames pooled)'); ax.set_ylabel('χ² / 6  (over-dispersion k)')
        ax.set_title('(c) covariance calibration vs F')
        ax.legend()

        fig.suptitle(f'Frame-fusion post-process · {Path(args.dump).parent.name} · N={len(H)}',
                     fontsize=11.5, y=1.02)
        plt.savefig(args.plot, dpi=140, bbox_inches='tight')
        print(f'wrote {args.plot}')


if __name__ == '__main__':
    main()
