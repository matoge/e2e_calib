"""Quick dataloader bench: time ds[idx] for the V3 jpg_bytes cache.

Matches the training config in train_ps_v3_ddp.py:
  start_method='spawn', batch_size=64, num_workers=16, prefetch_factor=4,
  pin_memory=True, persistent_workers=True.

Notes on start_method:
  - train_ps_v3_ddp.py uses `forkserver, force=True` (line 354).
  - Previous version of this script called `set_start_method('forkserver')`
    WITHOUT force=True, which silently got ignored if torch/accel had already
    set a context at import-time, and led to `_pickle.UnpicklingError:
    pickle data was truncated` on the forkserver side.
  - We default to 'spawn' + force=True here because it's the most robust
    across environments; pass --start-method forkserver to mirror training.
"""
import sys, pathlib, time, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
# Force start method BEFORE importing torch so no one else wins the race.
import multiprocessing as mp  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='/dev/shm/pandaset_v3_full')
    ap.add_argument('--n', type=int, default=200,
                    help='samples for single-thread loop (ignored if --skip-single)')
    ap.add_argument('--n-batches', type=int, default=50,
                    help='batches to time after the warmup batch')
    ap.add_argument('--workers', type=int, default=16,
                    help='matches train_ps_v3_ddp.py default')
    ap.add_argument('--batch-size', type=int, default=64,
                    help='per-rank batch_size; matches train default')
    ap.add_argument('--prefetch', type=int, default=4,
                    help='matches train default')
    ap.add_argument('--min-crop-px', type=int, default=128)
    ap.add_argument('--max-crop-px', type=int, default=384)
    ap.add_argument('--img-size', type=int, default=64)
    ap.add_argument('--max-rot-deg', type=float, default=0.5)
    ap.add_argument('--max-offset-m', type=float, default=0.20)
    ap.add_argument('--start-method', default='spawn',
                    choices=['spawn', 'forkserver', 'fork'])
    ap.add_argument('--skip-single', action='store_true',
                    help='skip single-thread baseline (avoids parent bloat)')
    ap.add_argument('--world-size', type=int, default=1,
                    help='informational only: multiply sps to get global estimate')
    ap.add_argument('--pin-memory', default='true', choices=['true', 'false'],
                    help='DataLoader pin_memory flag (true/false); default matches training')
    args = ap.parse_args()
    pin_memory = (args.pin_memory == 'true')

    # Must set start method before any fork/spawn happens.
    mp.set_start_method(args.start_method, force=True)

    # torch import AFTER start_method is locked in.
    import torch  # noqa: F401
    from torch.utils.data import DataLoader
    from datasets.pandaset_full import PandaSetCalibDatasetFull, collate_full

    ds = PandaSetCalibDatasetFull(args.cache, split='train',
                                   img_size=args.img_size,
                                   min_crop_px=args.min_crop_px,
                                   max_crop_px=args.max_crop_px,
                                   max_rot_deg=args.max_rot_deg,
                                   max_offset_m=args.max_offset_m,
                                   oversample=1)
    print(f'cache: {args.cache}  insts: {len(ds)}  start_method={args.start_method}')

    if not args.skip_single:
        # Single-thread per-sample (DO NOT run with large --n before DataLoader
        # if using forkserver: the parent's address space gets huge and the
        # spawn payload transfer becomes fragile.)
        print(f'\n=== single-thread (n={args.n}) ===')
        _ = ds[0]  # warmup
        t0 = time.time()
        for i in range(args.n):
            ds[i % len(ds)]
        dt = (time.time() - t0) * 1000
        print(f'  total {dt:.0f}ms  per-sample {dt/args.n:.2f}ms  ⇒ {1000*args.n/dt:.0f} sps')

    # Multi-worker DataLoader (matches training rank config)
    print(f'\n=== DataLoader workers={args.workers} batch={args.batch_size} '
          f'prefetch={args.prefetch} start={args.start_method} '
          f'pin_memory={pin_memory} ===')
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers,
                         prefetch_factor=args.prefetch, persistent_workers=True,
                         collate_fn=collate_full, pin_memory=pin_memory, shuffle=True)
    it = iter(loader)
    t_warmup = time.time()
    _ = next(it)  # warmup: workers spin up + first prefetch
    print(f'  [warmup] first batch in {time.time()-t_warmup:.2f}s')
    t0 = time.time(); n_samp = 0
    for _ in range(args.n_batches):
        b = next(it); n_samp += b[0].shape[0]
    dt = time.time() - t0
    sps = n_samp / dt
    print(f'  {n_samp} samples in {dt:.2f}s  ⇒ {sps:.0f} sps  '
          f'({dt*1000/n_samp:.2f}ms/sample)')
    if args.world_size > 1:
        print(f'  [estimated global with world_size={args.world_size}]: '
              f'{sps*args.world_size:.0f} sps')


if __name__ == '__main__':
    main()
