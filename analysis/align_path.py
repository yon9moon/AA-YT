#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib import rcParams

rcParams.update({
    "font.family": "Arial"
})


# 与 dataloader 中的缩放保持一致：训练/测试时坐标都除以 20
LABEL_SCALE = 1


def load_json(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_label(label_str: str) -> np.ndarray:
    """从 JSON 中的标签字符串解析为物理坐标 (单位与 dataloader 一致，乘以 20)。

    classify_by_truth_output.json 里的 key 形如 "0 0"、"1 -1"，
    这些数值对应的是训练/测试时的归一化坐标（除以 20 之后再取整）。
    这里将其视作归一化坐标，再乘以 LABEL_SCALE 恢复到物理量纲，
    从而与 dataloader / single_sample_test 中的处理保持一致。
    """

    parts = [p for p in label_str.replace(',', ' ').split() if p]
    if len(parts) != 2:
        raise ValueError(f"Label '{label_str}' does not contain exactly 2 components")
    base = np.array([float(parts[0]), float(parts[1])], dtype=np.float32)
    return base * LABEL_SCALE


def build_vectors(lens_data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    starts = []
    deltas = []
    adjusted = []
    highlight_mask = []
    # 高亮的物理坐标（单位与 LABEL_SCALE 后的一致）
    HIGHLIGHT_LABELS = {
        (0.0, 0.0),
        (-18.0, -4.0),
        (30.0, 2.0),
    }
    for label_str, entry in lens_data.items():
        predict = entry.get("predict")
        if not predict or len(predict) != 2:
            continue
        # 物理起始坐标（例如 0, 20 等单位），由 JSON 中归一化坐标 * LABEL_SCALE 得到
        start = parse_label(label_str)

        # 只保留坐标为整数的点，避免非栅格噪声点
        rounded_ints = np.round(start).astype(int)
        if not np.allclose(start, rounded_ints, atol=1e-6):
            continue
        if not np.all(rounded_ints % 2 == 0):
            continue
        # 预测也是在归一化域内，这里同样乘以 LABEL_SCALE 还原到物理量纲
        predict_arr = np.array(predict, dtype=np.float32) * LABEL_SCALE
        # 与原脚本保持语义一致：使用 -predict 作为位移向量
        delta = -predict_arr
        dest = start + delta
        dest = np.round(dest, 2)
        starts.append(start)
        deltas.append(dest - start)
        adjusted.append(dest)
        highlight_mask.append(tuple(start.tolist()) in HIGHLIGHT_LABELS)
    if not starts:
        raise ValueError("No valid entries with predictions were found for this lens")
    return np.stack(starts), np.stack(deltas), np.stack(adjusted), np.array(highlight_mask, dtype=bool)


def plot_vectors(starts: np.ndarray, deltas: np.ndarray, adjusted: np.ndarray, highlight_mask: np.ndarray, lens_id: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal', adjustable='box')

    base_mask = ~highlight_mask

    if np.any(base_mask):
        ax.quiver(
            starts[base_mask, 0],
            starts[base_mask, 1],
            deltas[base_mask, 0],
            deltas[base_mask, 1],
            angles='xy',
            scale_units='xy',
            scale=1,
            width=0.003,
            color='tab:blue',
            alpha=0.6,
            zorder=2
        )

    if np.any(highlight_mask):
        ax.quiver(
            starts[highlight_mask, 0],
            starts[highlight_mask, 1],
            deltas[highlight_mask, 0],
            deltas[highlight_mask, 1],
            angles='xy',
            scale_units='xy',
            scale=1,
            width=0.003,
            color='tab:red',
            alpha=0.9,
            zorder=4
        )

    ax.scatter(starts[:, 0], starts[:, 1], s=5, c='tab:gray', zorder=1)
    if np.any(base_mask):
        ax.scatter(adjusted[base_mask, 0], adjusted[base_mask, 1], s=5, c='gold', zorder=3)
    if np.any(highlight_mask):
        ax.scatter(adjusted[highlight_mask, 0], adjusted[highlight_mask, 1], s=20, c='tab:red', edgecolor='white', linewidth=0.5, zorder=5)

    min_x = min(starts[:, 0].min(), adjusted[:, 0].min())
    max_x = max(starts[:, 0].max(), adjusted[:, 0].max())
    min_y = min(starts[:, 1].min(), adjusted[:, 1].min())
    max_y = max(starts[:, 1].max(), adjusted[:, 1].max())

    margin = 2
    x_low, x_high = min_x - margin, max_x + margin
    y_low, y_high = min_y - margin, max_y + margin
    ax.set_xlim(x_low, x_high)
    ax.set_ylim(y_low, y_high)
    ax.set_xlabel(r"Decenter X ($\mathrm{\mu}$m)", fontsize=20)
    ax.set_ylabel(r"Decenter Y ($\mathrm{\mu}$m)", fontsize=20)
    ax.tick_params(axis='both', labelsize=20)
    ax.grid(True, linestyle='--', alpha=0.2)


    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:gray', markersize=6, label='Initial Position')
    ]
    if np.any(base_mask):
        legend_handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='gold', markersize=6, label='Adjusted Position'))
    if np.any(highlight_mask):
        legend_handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:red', markersize=6, label='Highlighted Position'))
    ax.legend(handles=legend_handles, loc='upper right', fontsize=20)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path.with_suffix('.png'), dpi=300)
    fig.savefig(output_path.with_suffix('.pdf'), dpi=300)
    fig.savefig(output_path.with_suffix('.svg'))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot adjustment paths for a single lens")
    parser.add_argument("json_path", type=Path, help="Path to classify_by_truth_output.json")
    parser.add_argument("lens_id", type=str, help="Lens identifier (e.g., '25')")
    parser.add_argument("output_path", type=Path, help="Output file path without extension")
    args = parser.parse_args()

    data = load_json(args.json_path)
    lens_data = data.get(str(args.lens_id))
    if not lens_data:
        raise KeyError(f"Lens '{args.lens_id}' not found in JSON")

    starts, deltas, adjusted, highlight_mask = build_vectors(lens_data)
    plot_vectors(starts, deltas, adjusted, highlight_mask, str(args.lens_id), args.output_path)


if __name__ == "__main__":
    main()
