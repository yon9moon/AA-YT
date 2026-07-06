#!/usr/bin/env python3
"""
Generate training configs and test config for the cross-dataset generalization experiment.

Design:
  - Test set: Dataset_Splitted_0407 (both train/ and valid/)
  - Training pool (ordered by date): 0409, 0509, 0523, 0623, 0625, New, New2
  - Excluded: NewNorm
  - 7 individual baselines (train on each dataset alone)
  - 7 cumulative stacking runs (train on first K datasets)

Output:
  experiments/configs/train_single_{ds}.json   — individual baseline configs
  experiments/configs/train_cumul_N{K}.json    — cumulative stacking configs
  experiments/configs/test_config.json         — shared test config
  experiments/manifest.csv                     — experiment manifest
"""

import os
import json
import csv
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = Path("/root")
PROJECT = Path("/root/AA-YT")
CONFIG_DIR = PROJECT / "experiments" / "configs"
MANIFEST_PATH = PROJECT / "experiments" / "manifest.csv"

# ── Dataset definitions ────────────────────────────────────────────────
TEST_DATASET = "Dataset_Splitted_0407"
TEST_DIRS = [
    str(ROOT / TEST_DATASET / "train"),
    str(ROOT / TEST_DATASET / "valid"),
]

# Training datasets in stacking order (by date). NewNorm excluded.
TRAIN_DATASETS_ORDERED = [
    "Dataset_Splitted_0409",
    "Dataset_Splitted_0509",
    "Dataset_Splitted_0523",
    "Dataset_Splitted_0623",
    "Dataset_Splitted_0625",
    "Dataset_Splitted_New",
    "Dataset_Splitted_New2",
]

# ── Base config template (shared hyperparams) ─────────────────────────
BASE_TRAIN_CONFIG = {
    "n_gpu": 1,
    "arch": {
        "type": "AAmodel",
        "args": {
            "in_channels": 1,
            "num_classes": 4,
            "img_num": 5
        }
    },
    "optimizer": {
        "type": "Adam",
        "args": {
            "lr": 0.001,
            "weight_decay": 0,
            "amsgrad": True
        }
    },
    "loss": "mse_loss_means",
    "metrics": ["mae_xs", "mae_ys", "mae_xt", "mae_yt"],
    "lr_scheduler": {
        "type": "StepLR",
        "args": {
            "step_size": 10,
            "gamma": 0.1
        }
    },
    "trainer": {
        "epochs": 200,
        "save_dir": "saved/",
        "save_period": 1,
        "verbosity": 2,
        "monitor": "min val_loss",
        "early_stop": 30,
        "tensorboard": True
    }
}

BASE_TEST_CONFIG = {
    "name": "GenExp_Test_on_0407",
    "n_gpu": 1,
    "arch": {
        "type": "AAmodel",
        "args": {
            "in_channels": 1,
            "num_classes": 4,
            "img_num": 5
        }
    },
    "data_loader": {
        "type": "AADataLoader",
        "args": {
            "data_dir": TEST_DIRS,
            "batch_size": 64,
            "shuffle": False,
            "num_workers": 8,
            "img_num": 5
        }
    },
    "loss": "mse_loss_means",
    "metrics": ["mae_xs", "mae_ys", "mae_xt", "mae_yt"],
    "trainer": {
        "save_dir": "saved/",
        "verbosity": 2
    }
}


def make_train_config(name, train_datasets, valid_datasets):
    """Build a training config dict for the given datasets."""
    config = BASE_TRAIN_CONFIG.copy()
    config["name"] = name

    config["data_loader"] = {
        "type": "AADataLoader",
        "args": {
            "data_dir": [str(ROOT / ds / "train") for ds in train_datasets],
            "batch_size": 64,
            "shuffle": True,
            "num_workers": 8,
            "img_num": 5
        }
    }

    config["valid_data_loader"] = {
        "type": "AADataLoader",
        "args": {
            "data_dir": [str(ROOT / ds / "valid") for ds in valid_datasets],
            "batch_size": 64,
            "shuffle": False,
            "num_workers": 8,
            "img_num": 5
        }
    }

    return config


def main():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    # ── 1. Individual baselines (N=1 each) ──────────────────────────
    for ds in TRAIN_DATASETS_ORDERED:
        ds_short = ds.replace("Dataset_Splitted_", "")
        exp_id = f"single_{ds_short}"
        name = f"GenExp_single_{ds_short}"

        config = make_train_config(name, [ds], [ds])
        config_path = CONFIG_DIR / f"train_{exp_id}.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

        manifest_rows.append({
            "exp_id": exp_id,
            "exp_type": "single",
            "N": 1,
            "datasets": ds_short,
            "num_datasets": 1,
            "train_config": str(config_path),
        })
        print(f"  [single] {exp_id} -> {config_path}")

    # ── 2. Cumulative stacking (N=1 through N=7) ───────────────────
    for k in range(1, len(TRAIN_DATASETS_ORDERED) + 1):
        selected = TRAIN_DATASETS_ORDERED[:k]
        ds_short_list = [ds.replace("Dataset_Splitted_", "") for ds in selected]
        exp_id = f"cumul_N{k}"
        name = f"GenExp_cumul_N{k}"

        config = make_train_config(name, selected, selected)
        config_path = CONFIG_DIR / f"train_{exp_id}.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

        manifest_rows.append({
            "exp_id": exp_id,
            "exp_type": "cumulative",
            "N": k,
            "datasets": "+".join(ds_short_list),
            "num_datasets": k,
            "train_config": str(config_path),
        })
        print(f"  [cumul]  {exp_id} -> {config_path}")

    # ── 3. Shared test config ──────────────────────────────────────
    test_config_path = CONFIG_DIR / "test_config.json"
    with open(test_config_path, "w") as f:
        json.dump(BASE_TEST_CONFIG, f, indent=4)
    print(f"  [test]   config -> {test_config_path}")

    # ── 4. Write manifest CSV ──────────────────────────────────────
    fieldnames = ["exp_id", "exp_type", "N", "datasets", "num_datasets", "train_config"]
    with open(MANIFEST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"\nManifest written to {MANIFEST_PATH}")
    print(f"Total experiments: {len(manifest_rows)} "
          f"({len([r for r in manifest_rows if r['exp_type'] == 'single'])} single + "
          f"{len([r for r in manifest_rows if r['exp_type'] == 'cumulative'])} cumulative)")


if __name__ == "__main__":
    main()
