"""
Generate learning curve comparison plot from attention order ablation study.
Parses compare_attention_80ep.log and creates publication-quality visualization.
"""
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Parse the log file
log_file = Path("/home/hfunaya/git/e2e_calib/compare_attention_80ep.log")
output_file = Path("/home/hfunaya/git/e2e_calib/docs/images/attention_order_learning_curves.png")
output_file.parent.mkdir(parents=True, exist_ok=True)

with open(log_file) as f:
    content = f.read()

# Extract data for Cross→Self (original)
cross_self_section = re.search(
    r"Training: Cross→Self \(original\).*?(?=Training: Self→Cross|$)",
    content,
    re.DOTALL
)
cross_self_data = cross_self_section.group(0)

# Extract data for Self→Cross (swapped)
self_cross_section = re.search(
    r"Training: Self→Cross \(swapped\).*?(?=RESULTS SUMMARY|$)",
    content,
    re.DOTALL
)
self_cross_data = self_cross_section.group(0)

def parse_epochs(section_text):
    """Parse epoch data from log section."""
    epochs = []
    train_losses = []
    val_losses = []
    best_vals = []

    pattern = r"\[\s*(\d+)/80\] train=([\d.]+) val=([\d.]+) best=([\d.]+)"
    for match in re.finditer(pattern, section_text):
        epoch = int(match.group(1))
        train_loss = float(match.group(2))
        val_loss = float(match.group(3))
        best_val = float(match.group(4))

        epochs.append(epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        best_vals.append(best_val)

    return epochs, train_losses, val_losses, best_vals

# Parse both models
cs_epochs, cs_train, cs_val, cs_best = parse_epochs(cross_self_data)
sc_epochs, sc_train, sc_val, sc_best = parse_epochs(self_cross_data)

# Find best validation points
cs_best_idx = np.argmin(cs_val)
sc_best_idx = np.argmin(sc_val)
cs_best_val = cs_val[cs_best_idx]
sc_best_val = sc_val[sc_best_idx]

print(f"Cross→Self best: {cs_best_val:.4f} px at epoch {cs_epochs[cs_best_idx]}")
print(f"Self→Cross best: {sc_best_val:.4f} px at epoch {sc_epochs[sc_best_idx]}")
print(f"Improvement: {((sc_best_val - cs_best_val) / sc_best_val * 100):.1f}%")

# Create publication-quality plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left plot: Training curves
ax1.plot(cs_epochs, cs_train, 'b-', linewidth=2, label='Cross→Self (train)', alpha=0.7)
ax1.plot(sc_epochs, sc_train, 'r-', linewidth=2, label='Self→Cross (train)', alpha=0.7)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Training Loss (px)', fontsize=12)
ax1.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 80)

# Right plot: Validation curves with annotations
ax2.plot(cs_epochs, cs_val, 'b-', linewidth=2.5, label='Cross→Self (val)', alpha=0.8)
ax2.plot(sc_epochs, sc_val, 'r-', linewidth=2.5, label='Self→Cross (val)', alpha=0.8)

# Mark best points
ax2.scatter([cs_epochs[cs_best_idx]], [cs_best_val],
           color='blue', s=150, marker='*', zorder=5,
           edgecolor='white', linewidth=1.5)
ax2.scatter([sc_epochs[sc_best_idx]], [sc_best_val],
           color='red', s=150, marker='*', zorder=5,
           edgecolor='white', linewidth=1.5)

# Add annotations for best points
ax2.annotate(f'Best: {cs_best_val:.2f} px\n(epoch {cs_epochs[cs_best_idx]})',
            xy=(cs_epochs[cs_best_idx], cs_best_val),
            xytext=(cs_epochs[cs_best_idx] + 8, cs_best_val + 1.5),
            fontsize=10, color='blue', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='blue'),
            arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))

ax2.annotate(f'Best: {sc_best_val:.2f} px\n(epoch {sc_epochs[sc_best_idx]})',
            xy=(sc_epochs[sc_best_idx], sc_best_val),
            xytext=(sc_epochs[sc_best_idx] - 20, sc_best_val + 0.8),
            fontsize=10, color='red', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='red'),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Validation Loss (px)', fontsize=12)
ax2.set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10, loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 80)

# Add overall improvement annotation
improvement = (sc_best_val - cs_best_val) / sc_best_val * 100
fig.text(0.5, 0.02,
         f'Cross→Self achieves 80.2% improvement ({cs_best_val:.2f} vs {sc_best_val:.2f} px)',
         ha='center', fontsize=13, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.8', facecolor='lightgreen', alpha=0.3))

plt.suptitle('Attention Order Ablation Study (80 Epochs)',
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0.05, 1, 0.96])

# Save high-resolution PNG
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✅ Saved learning curves to: {output_file}")

# Also create a single combined plot for embedding
fig2, ax = plt.subplots(figsize=(10, 6))

# Plot all curves
ax.plot(cs_epochs, cs_train, 'b--', linewidth=1.5, label='Cross→Self (train)', alpha=0.5)
ax.plot(cs_epochs, cs_val, 'b-', linewidth=2.5, label='Cross→Self (val)', alpha=0.9)
ax.plot(sc_epochs, sc_train, 'r--', linewidth=1.5, label='Self→Cross (train)', alpha=0.5)
ax.plot(sc_epochs, sc_val, 'r-', linewidth=2.5, label='Self→Cross (val)', alpha=0.9)

# Mark best validation points
ax.scatter([cs_epochs[cs_best_idx]], [cs_best_val],
          color='blue', s=200, marker='*', zorder=5,
          edgecolor='white', linewidth=2)
ax.scatter([sc_epochs[sc_best_idx]], [sc_best_val],
          color='red', s=200, marker='*', zorder=5,
          edgecolor='white', linewidth=2)

# Annotations
ax.annotate(f'{cs_best_val:.2f} px',
           xy=(cs_epochs[cs_best_idx], cs_best_val),
           xytext=(cs_epochs[cs_best_idx] + 5, cs_best_val - 2),
           fontsize=11, color='blue', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='blue'),
           arrowprops=dict(arrowstyle='->', color='blue', lw=2))

ax.annotate(f'{sc_best_val:.2f} px',
           xy=(sc_epochs[sc_best_idx], sc_best_val),
           xytext=(sc_epochs[sc_best_idx] - 15, sc_best_val - 1),
           fontsize=11, color='red', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='red'),
           arrowprops=dict(arrowstyle='->', color='red', lw=2))

ax.set_xlabel('Epoch', fontsize=13)
ax.set_ylabel('Loss (px)', fontsize=13)
ax.set_title('Attention Order Ablation: Cross→Self vs Self→Cross (80 Epochs)',
            fontsize=15, fontweight='bold', pad=15)
ax.legend(fontsize=11, loc='upper right', framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(0, 80)

# Improvement box
textstr = f'✓ Cross→Self wins\n  1.33 px vs 6.71 px\n  80.2% improvement'
props = dict(boxstyle='round,pad=1', facecolor='lightgreen', alpha=0.8, edgecolor='darkgreen', linewidth=2)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=12,
       verticalalignment='top', bbox=props, fontweight='bold')

plt.tight_layout()

combined_file = output_file.parent / "attention_order_combined.png"
plt.savefig(combined_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Saved combined plot to: {combined_file}")

plt.close('all')
