"""VAL phase smoke test."""
import sys, pathlib, time, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

def main():
    from datasets.pandaset_full import PandaSetCalibDatasetFull, collate_full
    from torch.utils.data import DataLoader, ConcatDataset, Subset
    from models.model_depth import CalibNetDepth
    from scripts.util.midtrain_vis import midtrain_vis
    from pathlib import Path

    caches = ['/cache/kamikado_v3_tiled', '/cache/woven_v3_tile', '/cache/waymo_v3_tiled_i']
    exp = Path('/tmp/test_val_exp'); exp.mkdir(exist_ok=True, parents=True)

    ds_kw = dict(img_size=128, min_crop_px=128, max_crop_px=384,
                 max_offset_m=0.2, max_rot_deg=0.5, oversample=1)

    print('=== build val datasets ===', flush=True)
    vas = []
    for c in caches:
        va = PandaSetCalibDatasetFull(c, split='val', **ds_kw)
        print(f'  {c}: val={len(va)}', flush=True)
        vas.append(va)
    val_ds = ConcatDataset(vas)
    val_sub = Subset(val_ds, list(range(0, len(val_ds), max(1, len(val_ds)//400))))
    print(f'val subset: {len(val_sub)}', flush=True)

    val_loader = DataLoader(val_sub, batch_size=16, num_workers=4, collate_fn=collate_full,
                             persistent_workers=True, pin_memory=False,
                             multiprocessing_context='spawn')

    print('=== iterate val_loader ===', flush=True)
    t0 = time.time()
    n = 0
    for i, batch in enumerate(val_loader):
        n = i
    print(f'val_loader OK ({n+1} batches in {time.time()-t0:.1f}s)', flush=True)

    print('=== model + midtrain_vis ===', flush=True)
    device = 'cuda'
    model = CalibNetDepth(img_size=128, in_channels=3, n_layers=2).to(device)
    for cp in caches:
        sub_exp = exp / Path(cp).name; sub_exp.mkdir(exist_ok=True)
        print(f'  vis on {cp}', flush=True)
        midtrain_vis(model, sub_exp, cp, epoch=10,
                     img_size=128, min_crop_px=128, max_crop_px=384,
                     device=device, n=3)

    print('=== iterate val_loader again ===', flush=True)
    t0 = time.time()
    for i, batch in enumerate(val_loader):
        if i >= 5: break
    print(f'OK after vis ({i+1} batches in {time.time()-t0:.1f}s)', flush=True)
    print('ALL PASSED', flush=True)


if __name__ == '__main__':
    main()
