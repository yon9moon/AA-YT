#!/usr/bin/env python3
"""
Aggregate experiment results and generate:
  - summary/all_results.csv      — per-experiment metrics
  - summary/aggregate_by_N.csv   — mean/std per N for cumulative and single
  - summary/generalization_plot.png — MAE vs N plot

Usage:
  python experiments/collect_results.py
"""

import json
import csv
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT / "experiments" / "results"
SUMMARY_DIR = PROJECT / "experiments" / "summary"

# Metrics to include in CSV / plots
METRIC_KEYS = ["loss", "mae_xs", "mae_ys", "mae_xt", "mae_yt",
               "MAE_x", "MAE_y", "MAE_u", "MAE_v", "MAE_avg",
               "MAE_x_std", "MAE_y_std", "MAE_u_std", "MAE_v_std", "MAE_avg_std"]


def load_all_results():
    """Load all result JSONs from results/ directory."""
    results = []
    for rp in sorted(RESULTS_DIR.glob("*.json")):
        with open(rp) as f:
            data = json.load(f)
        if "error" not in data:
            results.append(data)
        else:
            print(f"  Skipping {rp.name} — error: {data['error']}")
    print(f"Loaded {len(results)} successful results")
    return results


def write_all_results_csv(results):
    """Write one-row-per-experiment CSV."""
    rows = []
    for r in results:
        row = {
            "exp_id": r["exp_id"],
            "exp_type": r["exp_type"],
            "N": r["N"],
            "datasets": r["datasets"],
            "train_time_min": r.get("train_time_min", ""),
        }
        metrics = r.get("metrics", {})
        for k in METRIC_KEYS:
            row[k] = metrics.get(k, "")
        rows.append(row)

    # Sort: single first (by dataset), then cumulative (by N)
    rows.sort(key=lambda r: (
        0 if r["exp_type"] == "single" else 1,
        r["N"],
        r["datasets"]
    ))

    csv_path = SUMMARY_DIR / "all_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {csv_path}")
    return rows


def write_aggregate_csv(results):
    """Aggregate by N (for cumulative) and also report individual stats."""
    # ── Cumulative: group by N ──
    cumul = [r for r in results if r["exp_type"] == "cumulative"]
    singles = [r for r in results if r["exp_type"] == "single"]

    agg_rows = []

    # Aggregate singles by N=1
    if singles:
        agg = _aggregate_group(singles, "single (all)")
        agg_rows.append(agg)

    # Aggregate cumulative by N
    for n in sorted(set(r["N"] for r in cumul)):
        group = [r for r in cumul if r["N"] == n]
        agg = _aggregate_group(group, f"cumul_N={n}")
        agg_rows.append(agg)

    csv_path = SUMMARY_DIR / "aggregate_by_N.csv"
    if agg_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
            writer.writeheader()
            writer.writerows(agg_rows)
        print(f"Wrote {len(agg_rows)} aggregate rows to {csv_path}")
    return agg_rows, singles, cumul


def _aggregate_group(group, label):
    """Compute mean and std of metrics across a group of experiments."""
    row = {"group": label, "count": len(group)}
    if not group:
        return row

    for k in METRIC_KEYS:
        vals = [r["metrics"].get(k) for r in group if r["metrics"].get(k) is not None]
        if vals:
            row[f"{k}_mean"] = round(np.mean(vals), 6)
            row[f"{k}_std"] = round(np.std(vals), 6)
        else:
            row[f"{k}_mean"] = ""
            row[f"{k}_std"] = ""
    return row


def plot_results(singles, cumul):
    """Generate MAE vs N plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    # ── Prepare data ──
    # Single baselines: scatter with per-dataset labels
    single_data = [(r["datasets"], r["metrics"].get("MAE_avg")) for r in singles
                   if r["metrics"].get("MAE_avg") is not None]

    # Cumulative: one point per N (or mean if multiple)
    cumul_by_n = defaultdict(list)
    for r in cumul:
        mae = r["metrics"].get("MAE_avg")
        if mae is not None:
            cumul_by_n[r["N"]].append(mae)

    cumul_ns = sorted(cumul_by_n.keys())
    cumul_means = [np.mean(cumul_by_n[n]) for n in cumul_ns]
    cumul_stds = [np.std(cumul_by_n[n]) for n in cumul_ns]

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # --- Panel 1: MAE_avg vs N ---
    ax = axes[0]
    # Single baselines as scatter at x=1 with dataset name labels
    if single_data:
        rng = np.random.default_rng(42)
        jitter = rng.uniform(-0.18, 0.18, len(single_data))
        colors = plt.cm.tab10(np.linspace(0, 1, len(single_data)))
        for (ds_name, mae_val), jit, c in zip(single_data, jitter, colors):
            ax.scatter(1 + jit, mae_val, s=60, color=c, alpha=0.7, zorder=5)
            ax.annotate(ds_name, (1 + jit, mae_val),
                        textcoords="offset points", xytext=(8, 4),
                        fontsize=7, color=c, alpha=0.9)
        # Dummy legend entry
        ax.scatter([], [], s=30, color="gray", alpha=0.5, label=f"Single datasets (n={len(single_data)})")
    # Cumulative line with error bars
    ax.errorbar(cumul_ns, cumul_means, yerr=cumul_stds,
                marker="o", linewidth=2, markersize=8, capsize=5,
                color="C0", label="Cumulative stacking")
    ax.set_xlabel("Number of training datasets (N)")
    ax.set_ylabel("MAE_avg")
    ax.set_title("Generalization: MAE_avg vs N")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Panel 2: XY error ---
    ax = axes[1]
    for dim, color, label in [("MAE_x", "C0", "X"), ("MAE_y", "C1", "Y")]:
        dim_vals = [np.mean([r["metrics"].get(dim) for r in cumul if r["N"] == n
                             and r["metrics"].get(dim) is not None])
                    for n in cumul_ns]
        ax.plot(cumul_ns, dim_vals, marker="o", color=color, label=label)
    ax.set_xlabel("Number of training datasets (N)")
    ax.set_ylabel("MAE")
    ax.set_title("XY Error vs N")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Panel 3: UV error ---
    ax = axes[2]
    for dim, color, label in [("MAE_u", "C2", "U"), ("MAE_v", "C3", "V")]:
        dim_vals = [np.mean([r["metrics"].get(dim) for r in cumul if r["N"] == n
                             and r["metrics"].get(dim) is not None])
                    for n in cumul_ns]
        ax.plot(cumul_ns, dim_vals, marker="s", color=color, label=label)
    ax.set_xlabel("Number of training datasets (N)")
    ax.set_ylabel("MAE")
    ax.set_title("UV Error vs N")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = SUMMARY_DIR / "generalization_plot.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {plot_path}")


def print_summary_table(agg_rows, singles, cumul):
    """Print a readable summary table to console."""
    print("\n" + "=" * 80)
    print("GENERALIZATION EXPERIMENT SUMMARY")
    print("=" * 80)

    # ── Single baselines ──
    print("\n── Single-dataset baselines (N=1) ──")
    print(f"{'Dataset':<12} {'MAE_avg':>10} {'MAE_x':>10} {'MAE_y':>10} {'MAE_u':>10} {'MAE_v':>10} {'Train(min)':>10}")
    print("-" * 72)
    for r in sorted(singles, key=lambda r: r["datasets"]):
        m = r.get("metrics", {})
        print(f"{r['datasets']:<12} {m.get('MAE_avg', 'N/A'):>10.4f} {m.get('MAE_x', 'N/A'):>10.4f} "
              f"{m.get('MAE_y', 'N/A'):>10.4f} {m.get('MAE_u', 'N/A'):>10.4f} "
              f"{m.get('MAE_v', 'N/A'):>10.4f} {r.get('train_time_min', 'N/A'):>10}")

    if singles:
        mae_avgs = [r["metrics"].get("MAE_avg") for r in singles if r["metrics"].get("MAE_avg") is not None]
        print(f"{'mean±std':<12} {np.mean(mae_avgs):>10.4f} ± {np.std(mae_avgs):.4f}")

    # ── Cumulative stacking ──
    print("\n── Cumulative stacking ──")
    print(f"{'N':<6} {'Datasets':<40} {'MAE_avg':>10} {'MAE_x':>10} {'MAE_y':>10} {'MAE_u':>10} {'MAE_v':>10} {'Δ from N=1':>12}")
    print("-" * 100)

    cumul_sorted = sorted(cumul, key=lambda r: r["N"])
    if cumul_sorted:
        baseline_mae = cumul_sorted[0]["metrics"].get("MAE_avg", None)
    else:
        baseline_mae = None

    for r in cumul_sorted:
        m = r.get("metrics", {})
        cur_mae = m.get("MAE_avg")
        delta = ""
        if baseline_mae is not None and cur_mae is not None:
            pct = (baseline_mae - cur_mae) / baseline_mae * 100
            delta = f"{pct:+.1f}%"
        print(f"N={r['N']:<4} {r['datasets']:<40} {cur_mae or 'N/A':>10.4f} "
              f"{m.get('MAE_x', 'N/A'):>10.4f} {m.get('MAE_y', 'N/A'):>10.4f} "
              f"{m.get('MAE_u', 'N/A'):>10.4f} {m.get('MAE_v', 'N/A'):>10.4f} {delta:>12}")

    print("\n" + "=" * 80)


def main():
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    results = load_all_results()
    if not results:
        print("No results found. Run experiments first.")
        return

    write_all_results_csv(results)
    agg_rows, singles, cumul = write_aggregate_csv(results)
    plot_results(singles, cumul)
    print_summary_table(agg_rows, singles, cumul)


if __name__ == "__main__":
    main()
