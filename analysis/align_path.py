#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

rcParams.update({
    "font.family": "Arial",
    "font.size": 14
})


def load_json(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_key_to_array(key_str: str) -> np.ndarray:
    """从 JSON Key 解析 Truth (x, y, u, v)"""
    parts = [float(p) for p in key_str.replace(',', ' ').split() if p]
    if len(parts) != 4:
        raise ValueError(f"Invalid label format: {key_str}")
    return np.array(parts, dtype=np.float32)


def extract_data_from_json(json_data: dict):
    truth_list = []
    predict_list = []
    errors_list = []

    for group_key, entries in json_data.items():
        if not isinstance(entries, dict):
            continue
            
        for label_str, val in entries.items():
            if "predict" not in val or "absolute_errors" not in val:
                continue
            
            try:
                t = parse_key_to_array(label_str)
                p = np.array(val["predict"], dtype=np.float32)
                e = np.array(val["absolute_errors"], dtype=np.float32)
                
                if len(t) == 4 and len(p) == 4 and len(e) == 4:
                    truth_list.append(t)
                    predict_list.append(p)
                    errors_list.append(e)
                    
            except ValueError:
                continue

    if not truth_list:
        return None, None, None
        
    return np.stack(truth_list), np.stack(predict_list), np.stack(errors_list)


def get_alpha_from_errors(error_vectors: np.ndarray, min_alpha=0.05, max_alpha=1.0):
    # 计算 L2 范数 (模长)
    residuals = np.linalg.norm(error_vectors, axis=1)
    
    if len(residuals) == 0:
        return np.array([]), 0.0
        
    r_min = residuals.min()
    r_max = residuals.max()
    
    if r_max - r_min < 1e-9:
        return np.full_like(residuals, max_alpha), r_max
        
    norm_r = (residuals - r_min) / (r_max - r_min)
    alphas = min_alpha + norm_r * (max_alpha - min_alpha)
    
    return alphas, r_max


def plot_weighted_quiver(ax, starts, predicts, alphas, title, xlabel, ylabel):
    # Delta (位移) = -Predict (假设预测的是偏差量，需反向调整)
    deltas = -predicts
    adjusted = starts + deltas
    
    # 1. 绘制起点 (Initial) - 灰色
    grey_rgba = np.zeros((len(starts), 4))
    grey_rgba[:, :3] = [0.5, 0.5, 0.5]
    grey_rgba[:, 3] = np.clip(alphas * 0.5, 0.05, 0.5)
    ax.scatter(starts[:, 0], starts[:, 1], s=10, c=grey_rgba, label='Initial (Truth)', zorder=1)
    
    # 2. 绘制终点 (Adjusted) - 金色
    gold_rgba = np.zeros((len(starts), 4))
    gold_rgba[:, :3] = [1.0, 0.75, 0.0]
    gold_rgba[:, 3] = alphas
    ax.scatter(adjusted[:, 0], adjusted[:, 1], s=15, c=gold_rgba, label='Adjusted', zorder=3)
    
    # 3. 绘制箭头 (Path) - 蓝色
    blue_rgba = np.zeros((len(starts), 4))
    blue_rgba[:, :3] = [0.12, 0.47, 0.71]
    blue_rgba[:, 3] = alphas
    
    ax.quiver(
        starts[:, 0], starts[:, 1],
        deltas[:, 0], deltas[:, 1],
        angles='xy', scale_units='xy', scale=1,
        color=blue_rgba,
        width=0.0025, headwidth=4, headlength=5,
        zorder=2
    )
    
    ax.set_title(f"{title}\n(High Opacity = High Error)", fontsize=16)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.axis('equal')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # === 修复点：兼容不同版本的 matplotlib ===
    leg = ax.legend(loc='upper right', fontsize=12)
    # 尝试获取 legend_handles (新版) 或 legendHandles (旧版)
    handles = getattr(leg, 'legend_handles', None) or getattr(leg, 'legendHandles', [])
    for lh in handles:
        lh.set_alpha(1.0)


def main():
    parser = argparse.ArgumentParser(description="Plot adjustment paths weighted by absolute errors")
    parser.add_argument("json_path", type=Path, help="Path to classify_by_truth_output.json")
    parser.add_argument("output_dir", type=Path, help="Directory to save output plots")
    args = parser.parse_args()

    data = load_json(args.json_path)
    truth_all, pred_all, errors_all = extract_data_from_json(data)
    
    if truth_all is None:
        print("No valid data found.")
        return

    print(f"Loaded {len(truth_all)} samples.")
    
    # XY
    xy_starts = truth_all[:, :2]
    xy_preds  = pred_all[:, :2]
    xy_errors = errors_all[:, :2]
    xy_alphas, xy_max_err = get_alpha_from_errors(xy_errors)
    print(f"[XY Plane] Max Residual Error: {xy_max_err:.6f}")

    # UV
    uv_starts = truth_all[:, 2:]
    uv_preds  = pred_all[:, 2:]
    uv_errors = errors_all[:, 2:]
    uv_alphas, uv_max_err = get_alpha_from_errors(uv_errors)
    print(f"[UV Plane] Max Residual Error: {uv_max_err:.6f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot XY
    fig_xy, ax_xy = plt.subplots(figsize=(10, 10))
    plot_weighted_quiver(
        ax_xy, 
        xy_starts, xy_preds, xy_alphas,
        title="Adjustment Paths - XY Plane",
        xlabel="X (mm)", ylabel="Y (mm)"
    )
    xy_out = args.output_dir / "path_xy_weighted.png"
    fig_xy.savefig(xy_out, dpi=300, bbox_inches='tight')
    print(f"Saved: {xy_out}")
    plt.close(fig_xy)

    # Plot UV
    fig_uv, ax_uv = plt.subplots(figsize=(10, 10))
    plot_weighted_quiver(
        ax_uv, 
        uv_starts, uv_preds, uv_alphas,
        title="Adjustment Paths - UV Plane",
        xlabel="U (deg)", ylabel="V (deg)"
    )
    uv_out = args.output_dir / "path_uv_weighted.png"
    fig_uv.savefig(uv_out, dpi=300, bbox_inches='tight')
    print(f"Saved: {uv_out}")
    plt.close(fig_uv)


if __name__ == "__main__":
    main()