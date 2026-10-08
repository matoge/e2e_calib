"""Train CalibNetDepth on grid+depth dataset at 64x64.
Config: edit config_grid_depth.py to change model / checkpoint / training params.
Results saved to experiments/{name}/
"""
import math, time, shutil, logging, sys, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import importlib, os
# 設定ファイルは GRID_CFG で選ぶ (既定 configs/grid_depth.py)。2 つの run を同時に回すため
_CFG_MOD = os.environ.get("GRID_CFG", "configs.grid_depth")
CFG = importlib.import_module(_CFG_MOD).CFG
from datasets.synthetic import (GridDepthDataset, collate_grid_depth,
                                make_image_and_points_grid_depth, make_image_and_points_lidar,
                                make_image_and_points_depth)
from models.model_depth import CalibNetDepth
from models.calibnet2 import CalibNet2
from models.model_cov import gaussian2d_nll
from torch.utils.data import DataLoader

torch.set_float32_matmul_precision("high")

DEVICE = torch.device("cuda")


class Logger:
    """Write to stdout and a log file. File opened/closed per line to avoid fork issues."""
    def __init__(self, log_path: Path):
        self.log_path = log_path
        log_path.write_text("")  # truncate

    def info(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"{ts}  {msg}"
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line, flush=True)   # ClearML のコンソールは標準出力しか拾わない


def bench_tags(c):
    no = tuple(c.get("n_obj", (1, 4)))
    return ["toy_bench", f"scene={c.get('scene', 'grid')}",
            (f"obj={no[0]}-{no[1]}" if c.get("scene") == "lidar" else "obj=2"),
            f"cnn={'convnext' if c.get('use_convnext') else 'plain'}",
            f"frustum={int(bool(c.get('use_frustum')))}", f"loss={c.get('loss', 'nll')}"]


# ── ベンチ用: サンプルを 1 回だけ作って GPU に置く ──────────────────────────
POOL_DIR = Path("/mnt/ssd2t/work/e2e_calib/cache/toy_pool")


def _gen_one(args):
    seed, img_size, max_offset, rd, scene = args[:5]
    n_obj = tuple(args[5]) if len(args) > 5 else (1, 4)
    if scene in ("lidar", "lidar_bgfix", "lidar_objonly"):
        # lidar_bgfix = 背景はずらさず物体だけずらす。lidar_objonly = 背景の点を出さない
        return make_image_and_points_lidar(img_size=img_size, max_offset=max_offset, seed=seed,
                                           n_obj=n_obj, shift_bg=(scene == "lidar"),
                                           keep_bg=(scene != "lidar_objonly"))
    if scene in ("depth", "depth_rgb", "depth_rgb_rd", "depth_rgb_rd_grid8", "depth_rgb_rd_area",
                 "depth_rgb_rd_grid4", "depth_rgb_rd_lidar", "depth_rgb_rd_lidar_canvas"):
        # train_depth.py の旧データ: 物体 2 個 + 背景、255 点 (1:1:1)、深度固定。
        # depth = グレー (白い物体と黒い背景)、depth_rgb = 同じ配置で色だけランダムな RGB
        return make_image_and_points_depth(img_size=img_size, max_offset=max_offset, seed=seed,
                                           color=scene.startswith("depth_rgb"),
                                           random_depth="_rd" in scene,
                                           grid_spacing=(4.0 if (scene.endswith("_grid4") or "_lidar" in scene) else
                                                         8.0 if scene.endswith(("_grid8", "_area")) else None),
                                           area_uniform=scene.endswith("_area"),
                                           lidar_spacing=((4.0, 24.0) if ("_lidar" in scene) else None),
                                           canvas=scene.endswith("_canvas"))
    return make_image_and_points_grid_depth(img_size=img_size, max_offset=max_offset,
                                            seed=seed, random_depths=rd)


def make_pool(n, base_seed, c):
    """seed = base_seed + i の n サンプルを作り、点数を最大値で 0 埋めして返す。
    同じ (n, base_seed, img_size, max_offset, random_depths) ならディスクから読む。"""
    rd = c.get("random_depths", False)
    scene = c.get("scene", "grid")
    no = tuple(c.get("n_obj", (1, 4)))
    f = POOL_DIR / (f"{scene}_n{n}_s{base_seed}_S{c['img_size']}_o{c['max_offset']}_rd{int(rd)}"
                    + (f"_obj{no[0]}-{no[1]}" if scene.startswith("lidar") else "") + ".pt")
    if f.exists():
        return torch.load(f)
    jobs = [(base_seed + i, c["img_size"], c["max_offset"], rd, scene, no) for i in range(n)]
    if n <= 4000:
        # 小さいものはこのプロセスで作る。torch を使った後に fork すると、子の
        # OpenMP が 100% CPU のまま戻らなかった (val 800 個で 2.5 分止まった)。
        xs = [_gen_one(j) for j in jobs]
    else:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(16) as pl:
            xs = pl.map(_gen_one, jobs, chunksize=64)
    imgs, true_p, dist_p, pad = collate_grid_depth(xs)
    out = dict(imgs=imgs.half(), true=true_p, dist=dist_p, pad=pad)
    POOL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(out, f)
    return out


class GpuBatches:
    """GPU 上のプールから (imgs, true_uvd, dist_uvd, pad_mask) を切り出す。
    n_draw 個を毎回ランダムに引く (shuffle=True) か、先頭から順に全部 (val)。
    点数はバッチ内の最大有効点数で切り詰める。"""
    def __init__(self, pool, batch_size, n_draw=None, shuffle=False):
        self.p = {k: v.to(DEVICE) for k, v in pool.items()}
        self.bs, self.shuffle = batch_size, shuffle
        self.n = self.p["imgs"].shape[0]
        self.n_draw = n_draw or self.n

    def __iter__(self):
        idx = (torch.randperm(self.n, device=DEVICE)[:self.n_draw] if self.shuffle
               else torch.arange(self.n_draw, device=DEVICE))
        for i in range(0, len(idx), self.bs):
            j = idx[i:i + self.bs]
            pad = self.p["pad"][j]
            m = int((~pad).sum(1).max())
            yield (self.p["imgs"][j].float(), self.p["true"][j, :m],
                   self.p["dist"][j, :m], pad[:, :m])


K_PER_CELL = 8   # main() で CFG["k_per_cell"] に置き換える


def _bucket(uvd, pad_mask, S, G, K_per_cell=None):
    """ずらした後の点を G×G セルに K_per_cell 個ずつ詰める (pandaset_full と同じ配置)。"""
    K_per_cell = K_PER_CELL if K_per_cell is None else K_per_cell
    B, N, C = uvd.shape
    cu = (uvd[..., 0] / (S / G)).long().clamp(0, G - 1)
    cv = (uvd[..., 1] / (S / G)).long().clamp(0, G - 1)
    cid = torch.where(pad_mask, torch.full_like(cu, G * G), cv * G + cu)
    bu = torch.zeros(B, G * G + 1, K_per_cell, C, device=uvd.device, dtype=uvd.dtype)
    bv = torch.zeros(B, G * G + 1, K_per_cell, dtype=torch.bool, device=uvd.device)
    for b in range(B):
        order = torch.argsort(cid[b], stable=True)
        sc = cid[b, order]
        cnt = torch.bincount(sc, minlength=G * G + 1)
        start = torch.zeros(G * G + 2, dtype=torch.long, device=uvd.device)
        start[1:] = cnt.cumsum(0)
        intra = torch.arange(N, device=uvd.device) - start[sc]
        keep = intra < K_per_cell
        bu[b, sc[keep], intra[keep]] = uvd[b, order][keep]
        bv[b, sc[keep], intra[keep]] = True
    return bu[:, :G * G], bv[:, :G * G]


def _nll_pt(params, target):
    """点ごとの 2D ガウス NLL (gaussian2d_nll の平均前)。"""
    tx, ty, lsx, lsy, rho = params.float().unbind(-1)
    dx, dy = target[..., 0].float() - tx, target[..., 1].float() - ty
    zx, zy = dx / lsx.exp(), dy / lsy.exp()
    r2 = (1 - rho * rho).clamp(min=1e-6)
    return 0.5 * ((zx * zx - 2 * rho * zx * zy + zy * zy) / r2 + 2 * lsx + 2 * lsy + torch.log(r2))


def mu_sigma_loss(params, gt, with_sigma, w=None):
    """μ は距離 ‖μ−正解‖ だけで学習する。with_sigma のときだけ、μ を止めた NLL で
    σ・ρ を足す。NLL 1 本だと μ の勾配が 1/σ² で縮み、σ を広げたまま μ が
    動かなかった (1 バッチ 400 回で物体の点 9.1〜15.4 px、距離だけなら 1.9〜2.6 px)。"""
    # w: 点ごとの重み (合計 1)。None なら全点平均。
    e = (params[..., :2].float() - gt.float()).norm(dim=-1)
    l_mu = e.mean() if w is None else (e * w).sum()
    if not with_sigma:
        return l_mu
    q = torch.cat([params[..., :2].detach(), params[..., 2:]], -1)
    nl = _nll_pt(q, gt)
    return l_mu + (nl.mean() if w is None else (nl * w).sum())


def group_weights(dist_uvd, pad_mask):
    """物体の点と背景の点が損失に半分ずつ効くように重みを作る (有効点の上で合計 1)。
    物体の点は全点の 6.8% しかなく、全点平均だと損失のほとんどが背景だった。"""
    d = dist_uvd[..., 2]
    v = ~pad_mask
    isbg = v & (d >= d.masked_fill(~v, -1).max(1, keepdim=True).values - 1e-3)
    isobj = v & ~isbg
    w = torch.zeros_like(d)
    n_obj, n_bg = isobj.sum().clamp(min=1), isbg.sum().clamp(min=1)
    has_obj = bool(isobj.any())
    w[isobj] = 0.5 / n_obj
    w[isbg] = (0.5 if has_obj else 1.0) / n_bg
    return w[v]


def epoch_loop(model, loader, optimizer, scaler, train, with_sigma=True, loss_mode="nll"):
    model.train(train)
    total_nll, total_mse, n = 0.0, 0.0, 0
    obj_nll_sum, obj_mse_sum, obj_n = 0.0, 0.0, 0
    bg_nll_sum,  bg_mse_sum,  bg_n  = 0.0, 0.0, 0
    for imgs, true_uvd, dist_uvd, pad_mask in loader:
        imgs     = imgs.to(DEVICE)
        true_uvd = true_uvd.to(DEVICE)
        dist_uvd = dist_uvd.to(DEVICE)
        pad_mask = pad_mask.to(DEVICE)
        gt       = true_uvd[..., :2] - dist_uvd[..., :2]

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if isinstance(model, CalibNet2):
                # 実データ側 (forward_calib) と同じ入力: [u,v,d,intensity] のクエリと、
                # ずらした後の点をセルごとに詰めた frustum バケット。人工データに
                # 強度は無いので 0。
                pin = torch.cat([dist_uvd, torch.zeros_like(dist_uvd[..., :1])], -1)
                pin = pin.masked_fill(pad_mask.unsqueeze(-1), 0.0)
                bu, bv = _bucket(pin, pad_mask, imgs.shape[-1], model.frustum_enc.grid_n)
                out = model(imgs, pin, dpose_R=None, vfp=None,
                            bucket_uvd=bu, bucket_valid=bv, key_padding_mask=pad_mask)
                params = out[0] if isinstance(out, tuple) else out
            elif getattr(model, "frustum_enc", None) is not None:
                # 局所 LiDAR エンコーダ: ずらした後の点をセルに詰めたバケットを渡す。
                # FrustumLocalEncoder は (u,v,d,強度) の 4 列固定なので強度 0 を足す
                # (モデルは use_intensity=True)。
                pin = torch.cat([dist_uvd, torch.zeros_like(dist_uvd[..., :1])], -1)
                pin = pin.masked_fill(pad_mask.unsqueeze(-1), 0.0)
                bu, bv = _bucket(pin, pad_mask, imgs.shape[-1], model.frustum_enc.grid_n)
                params = model(imgs, pin, key_padding_mask=pad_mask,
                               bucket_uvd=bu, bucket_valid=bv)
            else:
                params = model(imgs, dist_uvd, key_padding_mask=pad_mask)
            valid  = ~pad_mask
            if loss_mode == "split":
                loss = mu_sigma_loss(params[valid], gt[valid], with_sigma)
            elif loss_mode == "split_obj":
                # 物体の点だけで損失をとる (背景の点は入力・KV には残す)。人工データの
                # 背景は物体の形の穴でしかずれが分からず、当てにいく意味がない。
                w = group_weights(dist_uvd, pad_mask)
                d_ = dist_uvd[..., 2]
                isbg_ = d_ >= d_.masked_fill(pad_mask, -1).max(1, keepdim=True).values - 1e-3
                wo = (~isbg_)[valid].float()
                if wo.sum() == 0:
                    wo = torch.ones_like(wo)
                loss = mu_sigma_loss(params[valid], gt[valid], with_sigma, w=wo / wo.sum())
            elif loss_mode == "split_grp":
                loss = mu_sigma_loss(params[valid], gt[valid], with_sigma,
                                     w=group_weights(dist_uvd, pad_mask))
            else:
                loss = gaussian2d_nll(params[valid], gt[valid])

        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()

        with torch.no_grad():
            mse   = (params[valid][..., :2] - gt[valid]).norm(dim=-1).mean().item()
            depth = dist_uvd[..., 2]
            max_d = depth.max(dim=1, keepdim=True).values
            is_bg  = valid & (depth >= max_d - 1e-3)
            is_obj = valid & (depth <  max_d - 1e-3)
            if is_obj.any():
                obj_nll_sum += gaussian2d_nll(params[is_obj], gt[is_obj]).item()
                obj_mse_sum += (params[is_obj][..., :2] - gt[is_obj]).norm(dim=-1).mean().item()
                obj_n += 1
            if is_bg.any():
                bg_nll_sum += gaussian2d_nll(params[is_bg], gt[is_bg]).item()
                bg_mse_sum += (params[is_bg][..., :2] - gt[is_bg]).norm(dim=-1).mean().item()
                bg_n += 1
        total_nll += loss.item(); total_mse += mse; n += 1

    obj_nll = obj_nll_sum / max(obj_n, 1)
    bg_nll  = bg_nll_sum  / max(bg_n,  1)
    obj_mse = obj_mse_sum / max(obj_n, 1)
    bg_mse  = bg_mse_sum  / max(bg_n,  1)
    return total_nll / max(n, 1), total_mse / max(n, 1), obj_nll, bg_nll, obj_mse, bg_mse


@torch.no_grad()
def vis_sample(model, c, seed, path, title=""):
    """val の 1 サンプルを描く。× 真の位置、○ ずらした入力位置、＋ 予測位置、
    線 = 入力→予測。水色 = 物体の点、白 = 背景の点 (どちらも黒縁)。"""
    import matplotlib.patheffects as pe
    from matplotlib.collections import LineCollection
    was_training = model.training
    model.eval()
    S = c["img_size"]
    img, tu, du = _gen_one((seed, S, c["max_offset"], c.get("random_depths", False),
                            c.get("scene", "grid"), tuple(c.get("n_obj", (1, 4)))))
    dv = du.unsqueeze(0).to(DEVICE)
    pm = torch.zeros(dv.shape[:2], dtype=torch.bool, device=DEVICE)
    if getattr(model, "frustum_enc", None) is not None:
        dv = torch.cat([dv, torch.zeros_like(dv[..., :1])], -1)
        bu, bv = _bucket(dv, pm, S, model.frustum_enc.grid_n)
        out = model(img.unsqueeze(0).to(DEVICE), dv, key_padding_mask=pm, bucket_uvd=bu, bucket_valid=bv)
    else:
        out = model(img.unsqueeze(0).to(DEVICE), dv, key_padding_mask=pm)
    params = (out[0] if isinstance(out, tuple) else out)[0].cpu().float()
    model.train(was_training)
    pred = du[:, :2] + params[:, :2]
    sig = (params[:, 2].exp() * params[:, 3].exp()).sqrt()
    d = du[:, 2]; isbg = d >= d.max() - 1e-3
    OUT = [pe.withStroke(linewidth=2.0, foreground="black")]
    fig, ax = plt.subplots(figsize=(4.6, 5.0), dpi=110)
    if img.shape[0] == 1:
        ax.imshow(img[0].numpy(), cmap="gray", vmin=0, vmax=1, extent=[0, S, S, 0])
    else:
        ax.imshow(img.permute(1, 2, 0).numpy(), extent=[0, S, S, 0])
    lines = []
    for m, col, nm in [(isbg, "white", "背景"), (~isbg, "#00e5ff", "物体")]:
        if not m.any():
            lines.append(f"{nm} 0 点"); continue
        lc = LineCollection(torch.stack([du[m, :2], pred[m]], 1).numpy(), colors=col, linewidths=0.6)
        lc.set_path_effects([pe.withStroke(linewidth=1.5, foreground="black")]); ax.add_collection(lc)
        ax.scatter(tu[m, 0], tu[m, 1], s=16, marker="x", color=col, linewidths=1.0, path_effects=OUT)
        ax.scatter(du[m, 0], du[m, 1], s=12, marker="o", facecolors="none", edgecolors=col,
                   linewidths=0.8, path_effects=OUT)
        ax.scatter(pred[m, 0], pred[m, 1], s=22, marker="+", color=col, linewidths=1.2, path_effects=OUT)
        eb = (du[m, :2] - tu[m, :2]).norm(dim=1).mean().item()
        ea = (pred[m] - tu[m, :2]).norm(dim=1).mean().item()
        dvals = sorted(set(round(x, 3) for x in d[m].tolist()))
        lines.append(f"{nm} d={','.join(f'{x:.2f}' for x in dvals)}  {int(m.sum())} 点: {eb:.2f} → {ea:.2f} px  σ中央 {sig[m].median().item():.2f}")
    ax.set_xlim(0, S); ax.set_ylim(S, 0); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"seed {seed}  {title}\n" + "\n".join(lines), fontsize=8, fontname="Noto Sans CJK JP")
    ax.text(0.01, -0.02, "× 真の位置  ○ 入力  ＋ 予測  線 = 入力→予測", transform=ax.transAxes,
            fontsize=7, va="top", fontname="Noto Sans CJK JP")
    plt.savefig(path, dpi=110, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return path


def main():
    c       = CFG
    name    = c["name"]
    exp_dir = Path("experiments") / name
    exp_dir.mkdir(parents=True, exist_ok=True)

    log  = Logger(exp_dir / "train.log")
    ckpt = exp_dir / "best_model.pt"

    shutil.copy(_CFG_MOD.replace(".", "/") + ".py", exp_dir / "config.py")   # CFG は configs/ へ移った。旧 config_grid_depth.py は無い

    global K_PER_CELL
    K_PER_CELL = int(c.get("k_per_cell", 8))

    cml = None
    if c.get("clearml"):
        from scripts.util.clearml_context import init_with_context
        _task = init_with_context(project="e2e_calib/calib", name=name, cfg=dict(c),
                                  why=c.get("why", ""))
        _task.add_tags(bench_tags(c))   # 比較画面で構成が読めるように
        cml = _task.get_logger()
        # 起動直後に iteration 1 で 1 つ送る (ClearML の監視が秒数を iteration にしないように)
        cml.report_scalar(title="lr", series="lr", value=float(c["lr"]), iteration=1)

    log.info(f"name={name}  n_layers={c['n_layers']} self_first={c['self_first']} "
         f"max_offset={c['max_offset']}  batch={c['batch_size']}  "
         f"epochs={c['epochs']}  lr={c['lr']}→{c['lr_min']}")

    kw = dict(num_workers=4, pin_memory=True, persistent_workers=True,
              collate_fn=collate_grid_depth)
    rd = c.get("random_depths", False)
    if c.get("pool_size"):
        # ベンチ: 学習プールから毎エポック train_size 個を引く。val は従来と同じ
        # seed 700000〜 の val_size 個 (GridDepthDataset(base_seed=700000) と同じ中身)。
        tp = time.time()
        train_loader = GpuBatches(make_pool(c["pool_size"], 0, c), c["batch_size"],
                                  n_draw=c["train_size"], shuffle=True)
        val_loader   = GpuBatches(make_pool(c["val_size"], 700_000, c), c["batch_size"])
        log.info(f"pool: train {c['pool_size']} / val {c['val_size']}  ({time.time()-tp:.1f}s)")
    else:
      train_loader = DataLoader(
        GridDepthDataset(c["train_size"], c["img_size"],
                         max_offset=c["max_offset"], random_each_epoch=True,
                         random_depths=rd),
        batch_size=c["batch_size"], shuffle=True, **kw)
      val_loader = DataLoader(
        GridDepthDataset(c["val_size"], c["img_size"],
                         max_offset=c["max_offset"], base_seed=700_000,
                         random_depths=rd),
        batch_size=c["batch_size"], shuffle=False, **kw)

    if c.get("model", "depth") == "cnd2":
        # 実データ (train_cnd2_ddp.py) と同じ既定のコンストラクタ。img_size だけ人工データに合わせる。
        model = CalibNet2(d=128, img_size=c["img_size"], in_channels=c["in_channels"],
                          use_intensity=True, frustum_grid_n=c.get("grid_n", 8),
                          n_iter=c.get("n_iter", 4), n_heads=4, d_scalar=8, n_type1=40,
                          use_info_head=False).to(DEVICE)
    else:
      model = CalibNetDepth(
        img_size     = c["img_size"],
        in_channels  = c["in_channels"],
        n_layers     = c["n_layers"],
        self_first   = c["self_first"],
        kv_self_attn = c.get("kv_self_attn", False),
        cross_temp   = c.get("cross_temp", 1.0),
        # GridDepthDataset は uvd の 3 列しか出さない。実データ側 (PandaSet/nuScenes) に
        # 合わせて use_intensity の既定が True になったので、ここで明示的に切らないと
        # PointMLP3 が in_channels=4 で作られ、forward が
        # "mat1 and mat2 shapes cannot be multiplied (18880x3 and 4x64)" で落ちる。
        use_intensity = c.get("use_intensity", False),
        use_convnext  = c.get("use_convnext", False),
        use_frustum   = c.get("use_frustum", False),
        frustum_grid_n = c.get("grid_n", 16),
        frustum_q_from = c.get("frustum_q_from", "token"),
        deform_mode    = c.get("deform_mode", "none"),
        deform_first_global = c.get("deform_first_global", False),
        pe_mode        = c.get("pe_mode", "lin"),
        frustum_nb     = c.get("frustum_nb", "3x3"),
        point_rel      = c.get("point_rel", False),
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"params: {total_params/1e6:.2f}M")

    optimizer    = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=1e-3)
    epochs       = c["epochs"]
    lr_min_ratio = c.get("lr_min", 1e-5) / c["lr"]

    def lr_lambda(e):
        if e < 5: return (e + 1) / 5
        t = (e - 5) / max(1, epochs - 5)
        return lr_min_ratio + (1 - lr_min_ratio) * 0.5 * (1 + math.cos(math.pi * t))

    scheduler  = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler     = torch.GradScaler(device="cuda")
    best_val   = float("inf")
    t0         = time.time()
    temp_start = c.get("cross_temp_start", 1.0)
    reached    = {}
    temp_end   = c.get("cross_temp",       1.0)

    for epoch in range(1, epochs + 1):
        # cosine anneal temperature: temp_start → temp_end
        frac  = (epoch - 1) / max(1, epochs - 1)
        temp  = temp_end + (temp_start - temp_end) * 0.5 * (1 + math.cos(math.pi * frac))
        if hasattr(model, 'set_cross_temp'):
            model.set_cross_temp(temp)

        lm = c.get("loss", "nll")
        ws = epoch > c.get("sigma_start", 0)
        tr_nll, tr_mse, tr_obj, tr_bg, tr_obj_mse, tr_bg_mse = epoch_loop(
            model, train_loader, optimizer, scaler, True, with_sigma=ws, loss_mode=lm)
        with torch.no_grad():
            va_nll, va_mse, va_obj, va_bg, va_obj_mse, va_bg_mse = epoch_loop(model, val_loader, optimizer, scaler, False)
        scheduler.step()

        log.info(f"[{epoch:3d}/{epochs}]  "
             f"train nll={tr_nll:+.3f}(obj={tr_obj:+.3f} bg={tr_bg:+.3f}) "
             f"mse={tr_mse:.3f}(obj={tr_obj_mse:.3f} bg={tr_bg_mse:.3f})  "
             f"val nll={va_nll:+.3f}(obj={va_obj:+.3f} bg={va_bg:+.3f}) "
             f"mse={va_mse:.3f}(obj={va_obj_mse:.3f} bg={va_bg_mse:.3f})  "
             f"lr={scheduler.get_last_lr()[0]:.2e}  temp={temp:.3f}  "
             f"tot={(time.time()-t0)/60:.1f}min")

        if cml is not None:
            for t_, s_, v_ in [("mse_px/obj", "val", va_obj_mse), ("mse_px/bg", "val", va_bg_mse),
                               ("mse_px/obj", "train", tr_obj_mse), ("mse_px/bg", "train", tr_bg_mse),
                               ("nll/obj", "val", va_obj), ("nll/bg", "val", va_bg),
                               ("nll/all", "train", tr_nll), ("nll/all", "val", va_nll),
                               ("lr", "lr", scheduler.get_last_lr()[0])]:
                cml.report_scalar(title=t_, series=s_, value=float(v_), iteration=epoch)

        if cml is not None and (epoch % c.get("vis_every", 10) == 0 or epoch == 1):
            dbg = exp_dir / "debug"; dbg.mkdir(exist_ok=True)
            for vi in range(4):
                f = vis_sample(model, c, 700_000 + vi, dbg / f"ep{epoch:03d}_val{vi}.png", title=f"ep{epoch}")
                cml.report_image(title="val_samples", series=f"seed{700_000 + vi}",
                                 iteration=epoch, local_path=str(f))

        for th in (4.0, 2.0, 1.0):
            if th not in reached and va_obj_mse < th:
                reached[th] = (epoch, time.time() - t0)
                log.info(f"  ↳ BENCH obj<{th:g}px  ep{epoch}  {time.time()-t0:.0f}s")

        if va_nll < best_val:
            best_val = va_nll
            torch.save(model.state_dict(), ckpt)
            log.info(f"  ↳ saved (val_nll={best_val:.4f})")

    log.info(f"Best val NLL: {best_val:.4f}  |  time: {(time.time()-t0)/60:.1f}min")
    log.info("BENCH " + "  ".join(
        f"obj<{th:g}px: " + (f"ep{reached[th][0]} {reached[th][1]:.0f}s" if th in reached else "未到達")
        for th in (4.0, 2.0, 1.0)))

    # ── Val visualization ───────────────────────────────────────────────────
    vis_dir = exp_dir / "vis"
    vis_dir.mkdir(exist_ok=True)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    for vi in range(12):
        vis_sample(model, c, 700_000 + vi, vis_dir / f"val_{vi:02d}.png", title=f"best ckpt")
    log.info(f"Saved 12 val visualizations → {vis_dir}")


if __name__ == "__main__":
    main()
