"""Run iterative single-frame BA on 10 PandaSet val frames with all_v3_mc.

v2's batch used a different val split (old /tmp/pandaset_cache.pt, now gone), so
an exact head-to-head per-frame is impossible. Instead we sample 10 frames from
the new mc cache and report the distribution — comparable to v2's 10-frame
batch at the distribution level (same perturbation sigmas, same frame-sampling
policy: pick 10 frames spanning the cache, one per evenly-spaced pose key).

Outputs:
  experiments/all_v3_mc/ba_batch/batch_metrics.json
  experiments/all_v3_mc/ba_batch/<scene>_f<frame>/metrics.json
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import json, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from datasets.pandaset import PandaSetCalibDataset
from models.model_depth import CalibNetDepth
import ba_singleframe as bas


CACHE_PATH = '/mnt/nvme6t/e2e_calib_cache/pandaset_mc_cache.pt'
CKPT       = Path('experiments/all_v3_mc/best_model.pt')
OUT_ROOT   = Path('experiments/all_v3_mc/ba_batch')
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SIGMA_YPR  = 0.5
SIGMA_TR   = 0.20
N_ITERS    = 3
N_FRAMES   = 10
SEED       = 42


def find_scene_fi(cam_pos, root='/mnt/mininas/datasets/pandaset'):
    from pathlib import Path
    cp = np.asarray(cam_pos)
    root = Path(root)
    for scene_dir in sorted(root.iterdir()):
        pose_f = scene_dir / 'camera' / 'front_camera' / 'poses.json'
        if not pose_f.exists():
            continue
        poses = json.loads(pose_f.read_text())
        for fi, pose in enumerate(poses):
            p = np.array([pose['position']['x'], pose['position']['y'], pose['position']['z']])
            if np.linalg.norm(p - cp) < 0.01:
                return scene_dir.name, fi
    return None, None


def run_one(model, instances, inst_idxs, seed, scene, fi):
    first    = instances[inst_idxs[0]]
    K        = first['K_full'].numpy()
    R_gt_c2w = first['R_gt'].numpy()
    cp       = first['cam_pos'].numpy()
    pts_w    = first['pts'].numpy()

    rng = np.random.default_rng(seed)
    ypr_gt     = (rng.random(3)*2 - 1) * SIGMA_YPR
    t_gt_world = (rng.random(3)*2 - 1) * SIGMA_TR

    ypr_est, t_est_w, history = bas.iterate(
        model, instances, inst_idxs,
        ypr_gt, t_gt_world, R_gt_c2w, K, n_iters=N_ITERS)

    ypr_err = ypr_est - ypr_gt
    t_err_w = t_est_w - t_gt_world
    R_est  = bas.ypr_to_R(ypr_est)
    R_gt_d = bas.ypr_to_R(ypr_gt)
    rot_err_deg = np.rad2deg(np.arccos(
        np.clip((np.trace(R_est.T @ R_gt_d) - 1) / 2, -1, 1)))

    uv_gt, _ = bas.project((R_gt_c2w.T @ (pts_w - cp).T).T, K)
    R_off = R_gt_c2w @ bas.ypr_to_R(ypr_gt)
    cp_off = cp + t_gt_world
    uv_dist, z_dist = bas.project((R_off.T @ (pts_w - cp_off).T).T, K)
    R_res = R_gt_c2w @ bas.ypr_to_R(ypr_gt - ypr_est)
    cp_res = cp + (t_gt_world - t_est_w)
    uv_ba, z_ba = bas.project((R_res.T @ (pts_w - cp_res).T).T, K)
    vis = (z_dist > 0.5)
    err_before = float(np.linalg.norm(uv_dist[vis] - uv_gt[vis], axis=1).mean())
    err_after  = float(np.linalg.norm(uv_ba[vis]   - uv_gt[vis], axis=1).mean())

    return dict(
        scene=scene, frame=int(fi), n_obj_insts=len(inst_idxs),
        ypr_gt_deg=ypr_gt.tolist(),
        t_gt_world_cm=(t_gt_world*100).tolist(),
        ypr_est_deg=ypr_est.tolist(),
        t_est_world_cm=(t_est_w*100).tolist(),
        rot_err_deg=float(rot_err_deg),
        ypr_err_deg=ypr_err.tolist(),
        t_err_cm=float(np.linalg.norm(t_err_w) * 100),
        t_err_vec_cm=(t_err_w*100).tolist(),
        reproj_before_px=err_before,
        reproj_after_px=err_after,
        history=history,
    )


def main():
    print('[1] loading dataset & model')
    ds = PandaSetCalibDataset(CACHE_PATH, split='val')
    model = CalibNetDepth(img_size=64, in_channels=3, n_layers=4,
                          use_convnext=True, use_frustum=True).to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=True))
    model.eval()

    groups = defaultdict(list)
    for i, inst in enumerate(ds.instances):
        if 'obj_pos' not in inst:
            continue
        key = tuple(inst['cam_pos'].numpy().round(3).tolist())
        groups[key].append(i)
    keys_sorted = sorted(groups.keys(), key=lambda k: -len(groups[k]))  # richest first
    print(f'  val cache: {len(groups)} unique frames, richest has {len(groups[keys_sorted[0]])} obj insts')

    # pick 10 frames with decent obj count, evenly strided through the ranked list
    candidates = [k for k in keys_sorted if len(groups[k]) >= 8]
    print(f'  frames with >=8 obj insts: {len(candidates)}')
    stride = max(1, len(candidates) // N_FRAMES)
    targets = candidates[::stride][:N_FRAMES]

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    t0 = time.time()
    for i, key in enumerate(targets):
        inst_idxs = groups[key]
        scene, fi = find_scene_fi(np.array(key))
        if scene is None:
            print(f'[{i+1}/{N_FRAMES}] pose_key={key} (scene unknown, skip)')
            continue
        print(f'\n--- [{i+1}/{N_FRAMES}] scene={scene} fi={fi} n_obj={len(inst_idxs)} ---')
        r = run_one(model, ds.instances, inst_idxs, seed=SEED + i, scene=scene, fi=fi)
        print(f'  rot_err={r["rot_err_deg"]:.3f}° t_err={r["t_err_cm"]:.2f}cm  '
              f'reproj {r["reproj_before_px"]:.2f}→{r["reproj_after_px"]:.2f} px')
        results.append(r)
        out_d = OUT_ROOT / f'{scene}_f{fi:02d}'
        out_d.mkdir(parents=True, exist_ok=True)
        (out_d / 'metrics.json').write_text(json.dumps(r, indent=2))

    (OUT_ROOT / 'batch_metrics.json').write_text(json.dumps(results, indent=2))
    dt = time.time() - t0

    # summary
    if results:
        rot = np.array([r['rot_err_deg'] for r in results])
        tcm = np.array([r['t_err_cm'] for r in results])
        before = np.array([r['reproj_before_px'] for r in results])
        after = np.array([r['reproj_after_px'] for r in results])
        print(f'\n=== SUMMARY ({len(results)} frames, {dt:.1f}s) ===')
        print(f'  rot_err_deg   median={np.median(rot):.3f}  mean={rot.mean():.3f}  max={rot.max():.3f}')
        print(f'  t_err_cm      median={np.median(tcm):.2f}   mean={tcm.mean():.2f}   max={tcm.max():.2f}')
        print(f'  reproj_before median={np.median(before):.2f}  mean={before.mean():.2f}')
        print(f'  reproj_after  median={np.median(after):.2f}   mean={after.mean():.2f}   max={after.max():.2f}')
        print(f'  converged (<10 px): {int((after < 10).sum())}/{len(results)}')


if __name__ == '__main__':
    main()
