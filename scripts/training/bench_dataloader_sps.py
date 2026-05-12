"""Measure DataLoader throughput in samples-per-second (no model, no GPU).

--profile: wraps the measure loop in cProfile (forces num_workers=0 so the
profile catches __getitem__). Throughput is artificially low under --profile —
use it for the per-call breakdown, not the sps headline.
"""
import argparse, sys, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from datasets.pandaset_full import PandaSetCalibDatasetFull, collate_full
from torch.utils.data import DataLoader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True, type=str)
    ap.add_argument('--split', default='train')
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--num-workers', type=int, default=12)
    ap.add_argument('--prefetch-factor', type=int, default=4)
    ap.add_argument('--img-size', type=int, default=128)
    ap.add_argument('--min-crop-px', type=int, default=128)
    ap.add_argument('--max-crop-px', type=int, default=384)
    ap.add_argument('--oversample', type=int, default=1)
    ap.add_argument('--preload', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--warmup', type=int, default=20)
    ap.add_argument('--profile', action='store_true',
                    help='cProfile the measure loop (forces num_workers=0)')
    ap.add_argument('--profile-rows', type=int, default=40)
    args = ap.parse_args()
    if args.profile and args.num_workers != 0:
        print(f'[profile] forcing num_workers=0 (was {args.num_workers})', flush=True)
        args.num_workers = 0

    print(f"creating dataset (preload={args.preload}, ~{70000*3.8/1000:.0f}s for PS front)...", flush=True)
    t_init = time.time()
    ds = PandaSetCalibDatasetFull(
        args.cache, split=args.split, img_size=args.img_size,
        min_crop_px=args.min_crop_px, max_crop_px=args.max_crop_px,
        oversample=args.oversample, preload=args.preload,
    )
    print(f"dataset: {len(ds)} samples preload={args.preload}  init={time.time()-t_init:.1f}s", flush=True)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers,
                    persistent_workers=(args.num_workers > 0),
                    prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
                    pin_memory=True, collate_fn=collate_full)
    it = iter(dl)
    print(f'warming up {args.warmup} steps...', flush=True)
    t0 = time.time()
    for _ in range(args.warmup):
        next(it)
    print(f'  warmup done ({time.time() - t0:.1f}s)', flush=True)
    print(f'measuring {args.steps} steps @ batch={args.batch_size} workers={args.num_workers}', flush=True)
    if args.profile:
        import cProfile, pstats, io as _io
        pr = cProfile.Profile()
        pr.enable()
    n = 0; t0 = time.time()
    for s in range(args.steps):
        next(it); n += args.batch_size
        if (s + 1) % 50 == 0:
            el = time.time() - t0
            print(f'  step {s+1}/{args.steps}: {n / el:.0f} sps  ({el:.1f}s)', flush=True)
    el = time.time() - t0
    print(f'\nFINAL: {n / el:.0f} sps  ({n} samples in {el:.1f}s, {el / args.steps * 1000:.1f} ms/step)')
    if args.profile:
        pr.disable()
        buf = _io.StringIO()
        pstats.Stats(pr, stream=buf).sort_stats('cumulative').print_stats(args.profile_rows)
        print('\n=== cProfile (cumulative) ===')
        print(buf.getvalue())


if __name__ == '__main__':
    main()
