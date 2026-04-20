"""Qualitative + quantitative verification of experiments/all_v3_mc on NS/PS/WM val."""
import torch, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from dataset_nuscenes import NuScenesCalibDataset
from dataset_pandaset import PandaSetCalibDataset
from dataset_waymo     import WaymoCalibDataset
from model_depth       import CalibNetDepth

DEV    = torch.device("cuda")
EXPDIR = Path("experiments/all_v3_mc")
OUT    = EXPDIR / "verify.png"
CDIR   = '/mnt/nvme6t/e2e_calib_cache'

DATASETS = [
    ('NS', NuScenesCalibDataset, f'{CDIR}/nuscenes_mc_cache.pt'),
    ('PS', PandaSetCalibDataset, f'{CDIR}/pandaset_mc_cache.pt'),
    ('WM', WaymoCalibDataset,    f'{CDIR}/waymo_mc_cache.pt'),
]
N_EVAL = 200     # per dataset, for stats
N_VIS  = 4       # per dataset, for figure


def load_model():
    ns = {}; exec((EXPDIR / "config.py").read_text(), ns); c = ns['CFG']
    m = CalibNetDepth(img_size=c['img_size'], in_channels=c['in_channels'],
                      n_layers=c['n_layers'], self_first=False,
                      use_convnext=c.get('use_convnext', False),
                      use_frustum=c.get('use_frustum', False)).to(DEV)
    m.load_state_dict(torch.load(EXPDIR / "best_model.pt",
                                 map_location=DEV, weights_only=True))
    m.eval()
    return m


@torch.no_grad()
def run_sample(model, img, true_uvd, dist_uvd):
    img = img.unsqueeze(0).to(DEV)
    pad = torch.zeros(1, dist_uvd.shape[0], dtype=torch.bool, device=DEV)
    inp = dist_uvd[..., :3].unsqueeze(0).to(DEV)
    params = model(img, inp, key_padding_mask=pad)[0].cpu().float().numpy()
    dist_xy  = dist_uvd[..., :2].numpy()
    true_xy  = true_uvd[..., :2].numpy()
    is_obj   = dist_uvd[..., 3].numpy() > 0.5
    pred_xy  = dist_xy + params[:, :2]
    return dist_xy, true_xy, pred_xy, is_obj, img[0].cpu().permute(1,2,0).numpy()


def eval_all(model):
    stats = {}
    vis   = {}
    rng   = np.random.default_rng(0)
    for name, cls, cache in DATASETS:
        ds = cls(cache, split='val')
        idxs = rng.choice(len(ds), size=min(N_EVAL, len(ds)), replace=False)
        eb_o, ea_o, eb_b, ea_b = [], [], [], []
        vis_picks = []
        for i, idx in enumerate(idxs):
            img, t, d = ds[int(idx)]
            dist, true, pred, is_obj, img_np = run_sample(model, img, t, d)
            err_b = np.linalg.norm(dist - true, axis=1)
            err_a = np.linalg.norm(pred - true, axis=1)
            if is_obj.any():
                eb_o.append(err_b[is_obj].mean()); ea_o.append(err_a[is_obj].mean())
            if (~is_obj).any():
                eb_b.append(err_b[~is_obj].mean()); ea_b.append(err_a[~is_obj].mean())
            if len(vis_picks) < N_VIS and is_obj.sum() >= 8 and err_b[is_obj].mean() > 5.0:
                vis_picks.append(dict(img=img_np, dist=dist, true=true, pred=pred,
                                       is_obj=is_obj,
                                       eb=err_b[is_obj].mean() if is_obj.any() else 0,
                                       ea=err_a[is_obj].mean() if is_obj.any() else 0))
        stats[name] = dict(
            obj_b=np.mean(eb_o), obj_a=np.mean(ea_o),
            bg_b =np.mean(eb_b), bg_a =np.mean(ea_b),
            n=len(idxs),
        )
        # fallback fill if not enough visually-shifted picks
        while len(vis_picks) < N_VIS and len(idxs) > 0:
            idx = int(rng.choice(idxs))
            img, t, d = ds[idx]
            dist, true, pred, is_obj, img_np = run_sample(model, img, t, d)
            if not is_obj.any(): continue
            err_b = np.linalg.norm(dist[is_obj] - true[is_obj], axis=1).mean()
            err_a = np.linalg.norm(pred[is_obj] - true[is_obj], axis=1).mean()
            vis_picks.append(dict(img=img_np, dist=dist, true=true, pred=pred,
                                   is_obj=is_obj, eb=err_b, ea=err_a))
        vis[name] = vis_picks
    return stats, vis


def render_figure(vis):
    S = 64
    fig, axes = plt.subplots(len(DATASETS), N_VIS, figsize=(N_VIS*3.2, len(DATASETS)*3.4),
                             dpi=120)
    for r, (name, _, _) in enumerate(DATASETS):
        for c in range(N_VIS):
            ax = axes[r][c]
            p  = vis[name][c] if c < len(vis[name]) else None
            ax.set_xticks([]); ax.set_yticks([])
            if p is None: ax.axis('off'); continue
            ax.imshow(p['img'], extent=[0, S, S, 0])
            d, t, pr, io = p['dist'], p['true'], p['pred'], p['is_obj']
            # before (red) and after (green) arrows — obj only
            if io.any():
                ax.quiver(d[io,0], d[io,1], t[io,0]-d[io,0], t[io,1]-d[io,1],
                          angles='xy', scale_units='xy', scale=1,
                          color='#ff3b30', width=0.006, alpha=0.85,
                          headwidth=3.5, headlength=4.5, zorder=4,
                          label='GT offset')
                ax.quiver(d[io,0], d[io,1], pr[io,0]-d[io,0], pr[io,1]-d[io,1],
                          angles='xy', scale_units='xy', scale=1,
                          color='#30d158', width=0.006, alpha=0.95,
                          headwidth=3.5, headlength=4.5, zorder=5,
                          label='pred')
            ax.text(2, 6, f"{name}", color='white', fontsize=11, family='monospace',
                    fontweight='700',
                    bbox=dict(facecolor='#111', edgecolor='none', pad=3))
            ax.text(S-2, S-2, f" {p['eb']:.1f}→{p['ea']:.2f}px ",
                    color='white', fontsize=10, family='monospace', ha='right', va='bottom',
                    bbox=dict(facecolor='#c13c14', edgecolor='none', pad=3))
            ax.set_xlim(0, S); ax.set_ylim(S, 0)
    plt.tight_layout()
    plt.savefig(OUT, dpi=120, bbox_inches='tight')
    plt.close(fig)


def main():
    model = load_model()
    stats, vis = eval_all(model)
    print()
    print(f"{'dataset':<4}  {'n':>4}  {'obj err px':>20}   {'bg err px':>20}")
    for name, s in stats.items():
        print(f"{name:<4}  {s['n']:>4}  "
              f"{s['obj_b']:6.2f} → {s['obj_a']:6.2f} ({(1-s['obj_a']/s['obj_b'])*100:+.0f}%)  "
              f"{s['bg_b']:6.2f} → {s['bg_a']:6.2f} ({(1-s['bg_a']/s['bg_b'])*100:+.0f}%)")
    render_figure(vis)
    print(f"\nsaved: {OUT}")


if __name__ == '__main__':
    main()
