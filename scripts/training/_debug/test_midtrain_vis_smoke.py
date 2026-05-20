"""1-shot midtrain_vis smoke for the 3 caches. No training, just call the
exact same midtrain_vis(...) the trainer fires every 10 epochs."""
import sys, pathlib, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

def main():
    from models.model_depth import CalibNetDepth
    from scripts.util.midtrain_vis import midtrain_vis
    from pathlib import Path

    caches = ['/cache/kamikado_v3_tiled', '/cache/woven_v3_tile', '/cache/waymo_v3_tiled_i']
    exp = Path('/tmp/vis_smoke'); exp.mkdir(exist_ok=True)
    device='cuda'
    model = CalibNetDepth(img_size=128, in_channels=3, n_layers=2).to(device)
    for cp in caches:
        sub_exp = exp / Path(cp).name; sub_exp.mkdir(exist_ok=True)
        print(f'=== {cp} ===', flush=True)
        midtrain_vis(model, sub_exp, cp, epoch=10,
                     img_size=128, min_crop_px=128, max_crop_px=384,
                     device=device, n=5)
    print('OK', flush=True)

if __name__ == '__main__':
    main()
