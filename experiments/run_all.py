#!/usr/bin/env python3
"""
Run all cross-dataset generalization experiments sequentially.

For each experiment in manifest.csv:
  1. Train model with the train config
  2. Find the saved model_best.pth
  3. Evaluate on the fixed test set
  4. Save results as JSON

Usage:
  python experiments/run_all.py              # run all
  python experiments/run_all.py --start 5    # resume from experiment index 5
  python experiments/run_all.py --dry-run    # list what would run without executing
"""

import os
import sys
import json
import csv
import subprocess
import glob
import time
import argparse
from pathlib import Path
from datetime import datetime

# Ensure project root is on path for imports
PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import torch
import numpy as np
from tqdm import tqdm

import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from utils import read_json
from train import SEED

# ── Paths ──────────────────────────────────────────────────────────────
MANIFEST_PATH = PROJECT / "experiments" / "manifest.csv"
RESULTS_DIR = PROJECT / "experiments" / "results"
DEFAULT_TEST_CONFIG = PROJECT / "experiments" / "configs" / "test_config.json"
TRAIN_SCRIPT = PROJECT / "train.py"


def find_best_model(config_name):
    """Find model_best.pth under saved/models/{config_name}/ ."""
    model_dir = PROJECT / "saved" / "models" / config_name
    candidates = sorted(glob.glob(str(model_dir / "**" / "model_best.pth"), recursive=True))
    if not candidates:
        return None
    # Return the most recently modified
    return max(candidates, key=os.path.getmtime)


def evaluate_model(model_path, test_config_path, device_str="cuda:0"):
    """
    Load a trained model and evaluate on the test set.
    Returns a dict of metrics.
    """
    config_json = read_json(str(test_config_path))
    config = ConfigParser(config_json)

    g = torch.Generator()
    g.manual_seed(SEED)

    # Data loader
    data_loader = config.init_obj("data_loader", module_data, generator=g)

    # Model
    model = config.init_obj("arch", module_arch)
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Loss & metrics
    loss_fn = getattr(module_loss, config["loss"])
    metric_fns = [getattr(module_metric, met) for met in config["metrics"]]

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for data, target, _ in tqdm(data_loader, desc="  Evaluating", leave=False):
            data, target = data.to(device), target.to(device)
            output = model(data)

            all_outputs.append(output.cpu())
            all_targets.append(target.cpu())

            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                mv = metric(output, target)
                total_metrics[i] += (mv.item() if torch.is_tensor(mv) else mv) * batch_size

    n_samples = len(data_loader.sampler)
    results = {"loss": total_loss / n_samples}
    for i, met in enumerate(metric_fns):
        results[met.__name__] = total_metrics[i].item() / n_samples

    # Add MAE with std (all 4 dims)
    if all_outputs:
        outputs = torch.cat(all_outputs, dim=0)
        targets = torch.cat(all_targets, dim=0)
        std_stats = module_metric.mae_with_std(outputs, targets)
        results.update({k: float(v) for k, v in std_stats.items()})

    return results


def run_one_experiment(row, device="0"):
    """Train + evaluate one experiment. Returns results dict."""
    exp_id = row["exp_id"]
    train_config = row["train_config"]
    test_config = row.get("test_config", "").strip()
    if not test_config:
        test_config = str(DEFAULT_TEST_CONFIG)
    config_name = Path(train_config).stem

    print(f"\n{'='*70}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Experiment: {exp_id}")
    print(f"  Type: {row['exp_type']}, N={row['N']}, Datasets: {row['datasets']}")
    print(f"{'='*70}")

    # ── Step 1: Train ──────────────────────────────────────────────
    print(f"  Training... (config: {train_config})")
    t_start = time.time()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device

    result = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT), "-c", train_config],
        env=env,
        capture_output=True,
        text=True,
        cwd=str(PROJECT),
    )

    train_time = time.time() - t_start
    print(f"  Training finished in {train_time/60:.1f} min (returncode={result.returncode})")

    # Print last few lines of training output for debugging
    train_output_lines = result.stdout.strip().split("\n")
    for line in train_output_lines[-10:]:
        if line.strip():
            print(f"    [train] {line.strip()[:200]}")

    if result.returncode != 0:
        print(f"  ERROR during training:")
        print(result.stderr[-2000:])
        return {"exp_id": exp_id, "error": "training_failed", "stderr": result.stderr[-1000:]}

    # ── Step 2: Find best model ────────────────────────────────────
    # The config name is embedded in the train config JSON's "name" field
    with open(train_config) as f:
        tc = json.load(f)
    config_name_for_model = tc["name"]  # e.g. "GenExp_single_0409"

    model_path = find_best_model(config_name_for_model)
    if not model_path:
        print(f"  ERROR: model_best.pth not found under saved/models/{config_name_for_model}/")
        return {"exp_id": exp_id, "error": "model_not_found"}

    print(f"  Best model: {model_path}")

    # ── Step 3: Evaluate ───────────────────────────────────────────
    print(f"  Evaluating on test set...")
    try:
        metrics = evaluate_model(model_path, test_config)
    except Exception as e:
        print(f"  ERROR during evaluation: {e}")
        return {"exp_id": exp_id, "error": f"evaluation_failed: {e}"}

    # ── Step 4: Save results ───────────────────────────────────────
    results = {
        "exp_id": exp_id,
        "exp_type": row["exp_type"],
        "N": int(row["N"]),
        "datasets": row["datasets"],
        "model_path": model_path,
        "train_time_min": round(train_time / 60, 1),
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
    }

    result_path = RESULTS_DIR / f"{exp_id}.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"  Results saved to {result_path}")

    # Print key metrics
    print(f"  Metrics: loss={metrics.get('loss', 'N/A'):.6f}, "
          f"MAE_x={metrics.get('MAE_x', 'N/A'):.4f}, "
          f"MAE_y={metrics.get('MAE_y', 'N/A'):.4f}, "
          f"MAE_u={metrics.get('MAE_u', 'N/A'):.4f}, "
          f"MAE_v={metrics.get('MAE_v', 'N/A'):.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run generalization experiments")
    parser.add_argument("--start", type=int, default=0,
                        help="Start from this experiment index (0-based)")
    parser.add_argument("--end", type=int, default=None,
                        help="Stop after this experiment index (inclusive)")
    parser.add_argument("--device", type=str, default="0",
                        help="CUDA device (default: 0)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List experiments without running")
    args = parser.parse_args()

    # Load manifest
    with open(MANIFEST_PATH, "r") as f:
        reader = csv.DictReader(f)
        manifest = list(reader)

    print(f"Loaded {len(manifest)} experiments from manifest")
    total = len(manifest)

    if args.dry_run:
        print("\n── Dry run ──")
        for i, row in enumerate(manifest):
            marker = ">>>" if i >= args.start and (args.end is None or i <= args.end) else "   "
            print(f"  [{i:2d}] {marker} {row['exp_id']:20s}  N={row['N']}  type={row['exp_type']}  "
                  f"datasets={row['datasets']}")
        print(f"\nWould run {min(total, total - args.start)} experiments "
              f"(from index {args.start} to {args.end if args.end is not None else total-1})")
        return

    # Run experiments
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for i, row in enumerate(manifest):
        if i < args.start:
            continue
        if args.end is not None and i > args.end:
            break

        # Skip if already completed
        result_path = RESULTS_DIR / f"{row['exp_id']}.json"
        if result_path.exists():
            print(f"\n[{i}/{total-1}] {row['exp_id']} — already done, skipping")
            with open(result_path) as f:
                all_results.append(json.load(f))
            continue

        print(f"\n[{i}/{total-1}] Running...")
        result = run_one_experiment(row, device=args.device)
        all_results.append(result)

        if "error" in result:
            print(f"  ⚠ Experiment {row['exp_id']} had error: {result['error']}")
            # Continue with next experiment

    # ── Final summary ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"All experiments complete. {len([r for r in all_results if 'error' not in r])}/{len(all_results)} succeeded.")
    if any("error" in r for r in all_results):
        print("Failures:")
        for r in all_results:
            if "error" in r:
                print(f"  - {r['exp_id']}: {r['error']}")


if __name__ == "__main__":
    main()
