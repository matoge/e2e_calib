"""Quick dataloader bench: time ds[idx] for the V3 jpg_bytes cache.
Single-thread baseline + multi-worker DataLoader throughput."""
import sys, pathlib, time, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import torch
from torch.utils.data import DataLoader
from datasets.pandaset_full import PandaSetCalibDatasetFull, collate_full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='/dev/shm/pandaset_v3_full')
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--prefetch', type=int, default=2)
    ap.add_argument('--max-crop-px', type=int, default=384)
    args = ap.parse_args()

    ds = PandaSetCalibDatasetFull(args.cache, split='train',
                                   img_size=64, min_crop_px=128,
                                   max_crop_px=args.max_crop_px, oversample=1)
    print(f'cache: {args.cache}  insts: {len(ds)}')

    # Single-thread per-sample
    print(f'\n=== single-thread (n={args.n}) ===')
    _ = ds[0]  # warmup
    t0 = time.time()
    for i in range(args.n):
        ds[i % len(ds)]
    dt = (time.time() - t0) * 1000
    print(f'  total {dt:.0f}ms  per-sample {dt/args.n:.2f}ms  ⇒ {1000*args.n/dt:.0f} sps')

    # Multi-worker DataLoader
    print(f'\n=== DataLoader workers={args.workers} batch={args.batch_size} prefetch={args.prefetch} ===')
    import multiprocessing as mp
    try: mp.set_start_method('forkserver')
    except RuntimeError: pass
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers,
                         prefetch_factor=args.prefetch, persistent_workers=True,
                         collate_fn=collate_full, pin_memory=True, shuffle=True)
    it = iter(loader)
    next(it)  # warmup
    n_batches = max(20, args.n // args.batch_size)
    t0 = time.time(); n_samp = 0
    for _ in range(n_batches):
        b = next(it); n_samp += b[0].shape[0]
    dt = time.time() - t0
    print(f'  {n_samp} samples in {dt:.2f}s  ⇒ {n_samp/dt:.0f} sps  ({dt*1000/n_samp:.2f}ms/sample)')


if __name__ == '__main__':
    main()
