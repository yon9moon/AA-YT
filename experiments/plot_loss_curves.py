#!/usr/bin/env python3
"""Parse info.log from single-dataset training runs and plot loss curves."""

import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

LOG_ROOT = Path("/root/AA-YT/saved/log")
SUMMARY_DIR = Path("/root/AA-YT/experiments/summary")

DATASETS = ["0409", "0509", "0523", "0623", "0625", "New", "New2"]
COLORS = plt.cm.tab10(np.linspace(0, 1, len(DATASETS)))


def parse_info_log(log_dir):
    """Parse info.log, return (epochs, train_loss, val_loss)."""
    candidates = sorted(log_dir.glob("*/info.log"))
    if not candidates:
        return None, None, None
    log_path = max(candidates, key=lambda p: p.stat().st_mtime)

    with open(log_path) as f:
        text = f.read()

    # Parse all epoch blocks
    pattern = re.compile(
        r"epoch\s*:\s*(\d+).*?"
        r"loss\s*:\s*([\d.e+-]+).*?"
        r"val_loss\s*:\s*([\d.e+-]+)",
        re.DOTALL
    )

    epochs, train_loss, val_loss = [], [], []
    for m in pattern.finditer(text):
        epochs.append(int(m.group(1)))
        train_loss.append(float(m.group(2)))
        val_loss.append(float(m.group(3)))

    return epochs, train_loss, val_loss


def main():
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes = axes.flatten()

    all_data = {}

    for idx, ds in enumerate(DATASETS):
        log_dir = LOG_ROOT / f"GenExp_single_{ds}"
        epochs, train_loss, val_loss = parse_info_log(log_dir)

        if epochs is None or len(epochs) == 0:
            print(f"  {ds}: no data")
            continue

        # Clip extreme val_loss spikes for cleaner visualization
        val_clipped = np.array(val_loss, dtype=float)
        if len(val_clipped) > 0:
            q95 = np.percentile(val_clipped, 95)
            val_clipped = np.clip(val_clipped, 0, q95 * 1.5)

        print(f"  {ds}: {len(epochs)} epochs, "
              f"train loss {train_loss[0]:.4f}→{train_loss[-1]:.4f}, "
              f"val loss {val_loss[0]:.3f}→{val_loss[-1]:.3f}")

        all_data[ds] = (epochs, train_loss, val_loss, val_clipped)

        # Individual subplot
        ax = axes[idx]
        ax.plot(epochs, train_loss, color=COLORS[idx], alpha=0.35, linewidth=1, label="train loss")
        ax.plot(epochs, val_clipped, color=COLORS[idx], linewidth=2, label="val loss")
        best_epoch = np.argmin(val_loss)
        ax.axvline(x=epochs[best_epoch], color=COLORS[idx], linestyle=":", alpha=0.5)
        ax.set_title(f"single_{ds}  (best val@{epochs[best_epoch]}ep)", fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Combined validation loss
    ax = axes[7]
    for ds, (epochs, _, val_loss, val_clipped) in all_data.items():
        idx = DATASETS.index(ds)
        best_ep = epochs[int(np.argmin(val_loss))]
        ax.plot(epochs, val_clipped, color=COLORS[idx], linewidth=1.5, label=f"{ds} (best@{best_ep}ep)", alpha=0.85)
    ax.set_title("Validation Loss — All Single Datasets", fontsize=11)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (clipped)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Single-Dataset Training & Validation Loss Curves\n(test on 0407 has same pattern as val — in-domain validation)", fontsize=13)
    plt.tight_layout()
    out_path = SUMMARY_DIR / "single_dataset_loss_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
