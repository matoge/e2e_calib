"""End-to-end training-step profiler: forward+backward, per-module breakdown.

Use this BEFORE/AFTER an optimization to confirm wall-time impact. Single-process,
no DataLoader noise — synthetic data shaped like the real training inputs.

usage:
  python scripts/training/bench_step.py
  python scripts/training/bench_step.py --img-size 64 --no-frustum-dense

What it reports:
  TOTAL fwd+bwd ms/iter       — the user-visible step time (= 1 / SPS × batch)
  per-module FWD ms           — cnn, point_mlp, frustum_enc.forward_dense,
                                cross_coarse, cross_refine, cross_fine, cross_fine2
  per-component (block 0) FWD — cross_attn, self_attn, extra_kv_attn, ffn, norms
  peak GPU memory             — for OOM headroom check

Implicit BWD = TOTAL − sum(FWD modules). Gradient computation typically
~2x forward; bwd cost is "the rest" since hooks fire on forward only.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import argparse, collections, time
import torch
from models.model_depth import CalibNetDepth

torch.backends.cudnn.benchmark = True
DEV = torch.device('cuda')


class CudaTimer:
    def __init__(self):
        self.starts = {}
        self.tot = collections.defaultdict(lambda: [0.0, 0])

    def pre(self, name):
        ev = torch.cuda.Event(enable_timing=True); ev.record()
        self.starts[name] = ev

    def post(self, name):
        ev = torch.cuda.Event(enable_timing=True); ev.record()
        torch.cuda.synchronize()
        self.tot[name][0] += self.starts[name].elapsed_time(ev)
        self.tot[name][1] += 1

    def report(self, header=''):
        if header: print(f'\n{header}')
        if not self.tot:
            print('  (no events)'); return
        print(f'  {"name":<26}{"ms/call":>10} {"calls":>6}')
        items = sorted(self.tot.items(), key=lambda x: -x[1][0]/x[1][1])
        total = sum(s/n for s,n in self.tot.values())
        for nm,(s,n) in items:
            avg = s/n
            print(f'  {nm:<26}{avg:>9.3f} {n:>5}')
        print(f'  {"sum":<26}{total:>9.3f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--n-pivots', type=int, default=256, help='per-batch query points (= dist_uvd N)')
    ap.add_argument('--n-full', type=int, default=1024, help='full lidar pts (no longer used directly; bucket grid is G²·K)')
    ap.add_argument('--k-per-cell', type=int, default=8, help='lidar pts per cell in the bucket grid')
    ap.add_argument('--grid-n', type=int, default=16, help='cell grid size (G)')
    ap.add_argument('--img-size', type=int, default=128)
    ap.add_argument('--n-layers', type=int, default=4)
    ap.add_argument('--no-convnext', action='store_true')
    ap.add_argument('--no-frustum', action='store_true', help='disable frustum_enc entirely')
    ap.add_argument('--no-frustum-dense', action='store_true', help='use per-pivot frustum, not dense map')
    ap.add_argument('--deform-mode', default='sl', choices=['none', 'sl', 'ml'])
    ap.add_argument('--iters', type=int, default=20, help='timed iters (after 3 warmup)')
    ap.add_argument('--dtype', default='bfloat16', choices=['bfloat16', 'float16', 'float32'])
    args = ap.parse_args()

    amp_dtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}[args.dtype]

    m = CalibNetDepth(
        d=128, img_size=args.img_size, in_channels=3, n_layers=args.n_layers,
        use_convnext=not args.no_convnext,
        use_frustum=not args.no_frustum,
        deform_mode=args.deform_mode,
        frustum_dense=(not args.no_frustum_dense and not args.no_frustum),
        use_intensity=True,
    ).to(DEV).train()

    B, N = args.batch, args.n_pivots
    img = torch.randn(B, 3, args.img_size, args.img_size, device=DEV)
    # 4ch input: u, v, d, intensity (use_intensity=True)
    dist_uvd = torch.cat([
        torch.rand(B, N, 2, device=DEV) * args.img_size,
        torch.rand(B, N, 1, device=DEV) * 0.5,
        torch.rand(B, N, 1, device=DEV),
    ], dim=-1)
    pad = torch.zeros(B, N, dtype=torch.bool, device=DEV)
    vfp = torch.full((B,), 1500.0, device=DEV)
    G = args.grid_n
    K = args.k_per_cell
    bucket_uvd = torch.cat([
        torch.rand(B, G * G, K, 2, device=DEV) * args.img_size,
        torch.rand(B, G * G, K, 1, device=DEV) * 0.5,
        torch.rand(B, G * G, K, 1, device=DEV),
    ], dim=-1)
    # ~half the slots populated on average — realistic
    bucket_valid = torch.rand(B, G * G, K, device=DEV) < 0.5

    # Hook timers
    t = CudaTimer()
    hooks = []
    for nm in ['cnn', 'point_mlp', 'frustum_enc', 'cross_coarse', 'cross_refine',
               'cross_fine', 'cross_fine2']:
        if hasattr(m, nm):
            mod = getattr(m, nm)
            h1 = mod.register_forward_pre_hook(lambda mm, inp, n=nm: t.pre(n))
            h2 = mod.register_forward_hook(lambda mm, inp, out, n=nm: t.post(n))
            hooks.extend([h1, h2])

    # frustum_enc.forward_dense is not a __call__, so manually wrap
    if hasattr(m, 'frustum_enc') and hasattr(m.frustum_enc, 'forward_dense'):
        orig_fd = m.frustum_enc.forward_dense
        def timed_fd(*a, **kw):
            t.pre('frustum_enc.forward_dense'); r = orig_fd(*a, **kw); t.post('frustum_enc.forward_dense'); return r
        m.frustum_enc.forward_dense = timed_fd

    # cross_coarse sub-modules — represent the per-block breakdown
    if hasattr(m, 'cross_coarse'):
        cb = m.cross_coarse
        for sub in ['cross_attn', 'self_attn', 'extra_kv_attn', 'ffn', 'norm_q', 'norm_kv', 'norm_self', 'norm_ffn']:
            if hasattr(cb, sub):
                mod = getattr(cb, sub)
                key = f'cc.{sub}'
                h1 = mod.register_forward_pre_hook(lambda mm, inp, n=key: t.pre(n))
                h2 = mod.register_forward_hook(lambda mm, inp, out, n=key: t.post(n))
                hooks.extend([h1, h2])

    def step():
        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
            out = m(img, dist_uvd, key_padding_mask=pad, vfp=vfp,
                    bucket_uvd=bucket_uvd, bucket_valid=bucket_valid)
        out.sum().backward()
        m.zero_grad(set_to_none=True)

    # warmup
    for _ in range(3): step()
    torch.cuda.synchronize()

    # total fwd+bwd timing
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(args.iters): step()
    torch.cuda.synchronize()
    total_ms = (time.time() - t0) / args.iters * 1000
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    # clear timer + measure component times (separate iterations to avoid double-counting)
    t.tot.clear()
    for _ in range(args.iters): step()
    torch.cuda.synchronize()

    # Report
    n_params = sum(p.numel() for p in m.parameters()) / 1e6
    print(f'\nCalibNetDepth bench (B={B}, N_pivots={N}, N_full={args.n_full}, '
          f'img={args.img_size}, dtype={args.dtype})')
    print(f'  arch: convnext={not args.no_convnext} frustum={not args.no_frustum} '
          f'dense={not args.no_frustum_dense and not args.no_frustum} '
          f'deform={args.deform_mode} n_layers={args.n_layers}')
    print(f'  params: {n_params:.2f}M')
    print(f'\n  TOTAL fwd+bwd: {total_ms:.2f} ms/iter   peak GPU: {peak_gb:.2f} GB')
    print(f'  implied SPS @ B={B}: {B / total_ms * 1000:.0f}')

    # Top-level modules
    top_keys = [k for k in t.tot if not k.startswith('cc.')]
    if top_keys:
        sub = CudaTimer()
        for k in top_keys: sub.tot[k] = t.tot[k]
        sub.report('Top-level modules (FWD only):')

    # cross_coarse internal
    sub_keys = [k for k in t.tot if k.startswith('cc.')]
    if sub_keys:
        sub = CudaTimer()
        for k in sub_keys: sub.tot[k] = t.tot[k]
        sub.report('cross_coarse internals (one block; ×4 for total cross cost):')

    # Implied bwd
    fwd_sum = sum(s/n for k,(s,n) in t.tot.items() if not k.startswith('cc.'))
    print(f'\n  implied bwd ≈ total - top-level fwd = {total_ms - fwd_sum:.2f} ms ({(total_ms-fwd_sum)/total_ms*100:.0f}% of step)')

    for h in hooks: h.remove()


if __name__ == '__main__':
    main()
