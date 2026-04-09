"""
vis.py  –  Visualise inference results on test data.

Loads best_model.pt, runs inference on 8 test samples, and saves result_128.png.
Red  dots = distorted_uv  (input, displaced from shape)
Blue dots = corrected_uv  (distorted_uv + predicted_offset)
"""
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from dataset import CalibDataset
from model import CalibNet

CKPT_PATH  = "best_model.pt"
OUT_PATH   = "result_128.png"
N_SAMPLES  = 8
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model() -> torch.nn.Module:
    model = CalibNet().to(DEVICE)
    state = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def run_inference(model, batch):
    imgs, true_uv, dist_uv = batch
    imgs    = imgs.to(DEVICE)
    dist_uv = dist_uv.to(DEVICE)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        pred_offset = model(imgs, dist_uv)        # (B,N,2)

    corrected_uv = (dist_uv + pred_offset).cpu()
    return imgs.cpu(), true_uv, dist_uv.cpu(), corrected_uv


def main():
    # test dataset: base_seed chosen to differ from train/val
    test_ds = CalibDataset(length=N_SAMPLES, base_seed=200_000)

    batch = [
        torch.stack([test_ds[i][k] for i in range(N_SAMPLES)])
        for k in range(3)
    ]

    model = load_model()
    imgs, true_uv, dist_uv, corrected_uv = run_inference(model, batch)

    # compute mean error before / after
    err_before = (dist_uv - true_uv).norm(dim=-1).mean().item()
    err_after  = (corrected_uv - true_uv).norm(dim=-1).mean().item()
    print(f"Mean point error  BEFORE correction: {err_before:.2f} px")
    print(f"Mean point error  AFTER  correction: {err_after:.2f} px")
    print(f"Improvement: {(1 - err_after/err_before)*100:.1f}%")

    # ── Plot ────────────────────────────────────────────────────────────────
    cols = 4
    rows = math.ceil(N_SAMPLES / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5), dpi=100)
    axes = axes.flatten()

    for i in range(N_SAMPLES):
        ax = axes[i]
        ax.imshow(imgs[i, 0].numpy(), cmap="gray", vmin=0, vmax=1,
                  origin="upper", extent=[0, 128, 128, 0])

        # distorted (input) — red
        ax.scatter(dist_uv[i, :, 0].numpy(), dist_uv[i, :, 1].numpy(),
                   c="red", s=4, alpha=0.6, linewidths=0)
        # corrected — blue
        ax.scatter(corrected_uv[i, :, 0].numpy(), corrected_uv[i, :, 1].numpy(),
                   c="deepskyblue", s=4, alpha=0.8, linewidths=0)

        ax.set_xlim(0, 127); ax.set_ylim(127, 0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"sample {i}", fontsize=8)

    # hide unused axes
    for j in range(N_SAMPLES, len(axes)):
        axes[j].set_visible(False)

    # legend
    legend_handles = [
        mpatches.Patch(color="red",         label=f"distorted input  (err={err_before:.1f}px)"),
        mpatches.Patch(color="deepskyblue", label=f"corrected output (err={err_after:.1f}px)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(
        "UV Correction:  red = distorted,  blue = corrected\n"
        f"(128×128, N=256 pts per sample, max offset ±15 px)",
        fontsize=10, y=1.01,
    )
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(OUT_PATH, bbox_inches="tight", dpi=120)
    print(f"Saved → {OUT_PATH}")


import math
if __name__ == "__main__":
    main()
