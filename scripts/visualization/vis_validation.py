#!/usr/bin/env python3
"""Generate validation visualizations for experiments."""

import sys
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from dataset import GridDepthDataset, collate_grid_depth
from model_depth import CalibNetDepth

def visualize_experiment(exp_name, num_samples=5):
    """Generate validation visualizations for an experiment."""
    exp_dir = Path("experiments") / exp_name
    config_path = exp_dir / "config.py"

    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return

    # Load config
    import importlib.util
    spec = importlib.util.spec_from_file_location("cfg", config_path)
    cfg_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_module)
    cfg = cfg_module.CFG

    # Load model
    ckpt_path = exp_dir / "best_model.pt"
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return

    model = CalibNetDepth(
        img_size=cfg.get('img_size', 64),
        in_channels=cfg.get('in_channels', 3),
        n_layers=cfg.get('n_layers', 3),
        self_first=cfg.get('self_first', False),
        kv_self_attn=cfg.get('kv_self_attn', False),
    )
    ckpt = torch.load(ckpt_path, map_location='cpu')
    # Handle both wrapped and unwrapped formats
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # Create validation dataset
    val_ds = GridDepthDataset(
        length=cfg.get('val_size', 800),
        img_size=cfg.get('img_size', 64),
        max_offset=cfg.get('max_offset', 16.0),
        base_seed=10000,
        random_depths=cfg.get('random_depths', False)
    )

    # Generate visualizations
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        img, pts_true, pts_dist = val_ds[i]

        # Run model
        img_batch = img.unsqueeze(0)
        pts_batch = pts_dist.unsqueeze(0)
        pad_mask = torch.zeros(1, pts_dist.shape[0], dtype=torch.bool)

        with torch.no_grad():
            pred = model(img_batch, pts_batch, pad_mask)

        # Compute corrected points
        tx, ty = pred[0, :, 0], pred[0, :, 1]
        pts_corrected = pts_dist.clone()
        pts_corrected[:, 0] -= tx
        pts_corrected[:, 1] -= ty

        # Compute errors
        err_before = (pts_dist[:, :2] - pts_true[:, :2]).norm(dim=1).mean().item()
        err_after = (pts_corrected[:, :2] - pts_true[:, :2]).norm(dim=1).mean().item()

        # Plot
        ax_img, ax_before, ax_after = axes[i]

        # Image
        ax_img.imshow(img.permute(1, 2, 0))
        ax_img.set_title(f"Sample {i}")
        ax_img.axis('off')

        # Before calibration
        ax_before.imshow(img.permute(1, 2, 0))
        ax_before.scatter(pts_dist[:, 0], pts_dist[:, 1], c='r', s=10, alpha=0.7, label='Distorted')
        ax_before.scatter(pts_true[:, 0], pts_true[:, 1], c='g', s=10, alpha=0.7, label='True')
        ax_before.set_title(f"Before: {err_before:.2f}px")
        ax_before.legend()
        ax_before.axis('off')

        # After calibration
        ax_after.imshow(img.permute(1, 2, 0))
        ax_after.scatter(pts_corrected[:, 0], pts_corrected[:, 1], c='b', s=10, alpha=0.7, label='Corrected')
        ax_after.scatter(pts_true[:, 0], pts_true[:, 1], c='g', s=10, alpha=0.7, label='True')
        ax_after.set_title(f"After: {err_after:.2f}px")
        ax_after.legend()
        ax_after.axis('off')

    plt.tight_layout()
    out_path = exp_dir / "validation_samples.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python vis_validation.py <experiment_name> [num_samples]")
        sys.exit(1)

    exp_name = sys.argv[1]
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    visualize_experiment(exp_name, num_samples)
