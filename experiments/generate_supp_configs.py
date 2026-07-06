#!/usr/bin/env python3
"""Generate supplementary experiment configs and append to manifest."""

import json
import csv
from pathlib import Path

ROOT = Path("/root")
PROJECT = Path("/root/AA-YT")
CONFIG_DIR = PROJECT / "experiments" / "configs"
MANIFEST_PATH = PROJECT / "experiments" / "manifest.csv"

BASE_TRAIN_CONFIG = {
    "n_gpu": 1,
    "arch": {"type": "AAmodel", "args": {"in_channels": 1, "num_classes": 4, "img_num": 5}},
    "optimizer": {"type": "Adam", "args": {"lr": 0.001, "weight_decay": 0, "amsgrad": True}},
    "loss": "mse_loss_means",
    "metrics": ["mae_xs", "mae_ys", "mae_xt", "mae_yt"],
    "lr_scheduler": {"type": "StepLR", "args": {"step_size": 10, "gamma": 0.1}},
    "trainer": {"epochs": 200, "save_dir": "saved/", "save_period": 1,
                "verbosity": 2, "monitor": "min val_loss", "early_stop": 30, "tensorboard": True}
}


def make_train_config(name, train_datasets):
    config = BASE_TRAIN_CONFIG.copy()
    config["name"] = name
    config["data_loader"] = {
        "type": "AADataLoader",
        "args": {"data_dir": [str(ROOT / ds / "train") for ds in train_datasets],
                 "batch_size": 64, "shuffle": True, "num_workers": 8, "img_num": 5}
    }
    config["valid_data_loader"] = {
        "type": "AADataLoader",
        "args": {"data_dir": [str(ROOT / ds / "valid") for ds in train_datasets],
                 "batch_size": 64, "shuffle": False, "num_workers": 8, "img_num": 5}
    }
    return config


def make_test_config(name, test_dirs):
    return {
        "name": name, "n_gpu": 1,
        "arch": {"type": "AAmodel", "args": {"in_channels": 1, "num_classes": 4, "img_num": 5}},
        "data_loader": {"type": "AADataLoader",
                        "args": {"data_dir": test_dirs, "batch_size": 64,
                                 "shuffle": False, "num_workers": 8, "img_num": 5}},
        "loss": "mse_loss_means", "metrics": ["mae_xs", "mae_ys", "mae_xt", "mae_yt"],
        "trainer": {"save_dir": "saved/", "verbosity": 2}
    }


# ── Set 1: New2 swap (replace 0625 with New2) ─────────────────
ds_pool_no_0625 = ["Dataset_Splitted_0409", "Dataset_Splitted_0509", "Dataset_Splitted_0523",
                   "Dataset_Splitted_0623", "Dataset_Splitted_New2", "Dataset_Splitted_New"]
exp1_id = "supp_swap_New2_no0625"
config1 = make_train_config(f"GenExp_{exp1_id}", ds_pool_no_0625)
with open(CONFIG_DIR / f"train_{exp1_id}.json", "w") as f:
    json.dump(config1, f, indent=4)
print(f"  [{exp1_id}] train on {[d.split('_')[-1] for d in ds_pool_no_0625]}")

# ── Set 2: 0623 as test set ──────────────────────────────────
test_0623_dirs = [str(ROOT / "Dataset_Splitted_0623" / s) for s in ["train", "valid"]]
test_config_0623 = make_test_config("GenExp_Test_on_0623", test_0623_dirs)
with open(CONFIG_DIR / "test_config_0623.json", "w") as f:
    json.dump(test_config_0623, f, indent=4)
print(f"  [test 0623] config created")

# Training pool: all except 0623, ordered by date
ds_pool_0623 = ["Dataset_Splitted_0409", "Dataset_Splitted_0509", "Dataset_Splitted_0523",
                "Dataset_Splitted_0625", "Dataset_Splitted_New", "Dataset_Splitted_New2"]

experiments_0623 = [
    ("supp_0623test_single_0409", ds_pool_0623[:1], 1),
    ("supp_0623test_cumul_N3", ds_pool_0623[:3], 3),
    ("supp_0623test_cumul_N6", ds_pool_0623[:6], 6),
]

for exp_id, datasets, n in experiments_0623:
    config = make_train_config(f"GenExp_{exp_id}", datasets)
    with open(CONFIG_DIR / f"train_{exp_id}.json", "w") as f:
        json.dump(config, f, indent=4)
    print(f"  [{exp_id}] N={n}: {[d.split('_')[-1] for d in datasets]}")

# ── Append to manifest ────────────────────────────────────────
new_rows = [
    {"exp_id": exp1_id, "exp_type": "supp_swap", "N": 6,
     "datasets": "+".join(d.split("_")[-1] for d in ds_pool_no_0625),
     "num_datasets": 6, "train_config": str(CONFIG_DIR / f"train_{exp1_id}.json"),
     "test_config": str(CONFIG_DIR / "test_config.json"), "note": "New2 replaces 0625"},
]
for exp_id, datasets, n in experiments_0623:
    new_rows.append({
        "exp_id": exp_id, "exp_type": "supp_0623test", "N": n,
        "datasets": "+".join(d.split("_")[-1] for d in datasets),
        "num_datasets": n, "train_config": str(CONFIG_DIR / f"train_{exp_id}.json"),
        "test_config": str(CONFIG_DIR / "test_config_0623.json"),
        "note": "test on 0623"
    })

# Read existing manifest
with open(MANIFEST_PATH, "r") as f:
    reader = csv.DictReader(f)
    existing_fieldnames = reader.fieldnames
    existing = list(reader)

# Merge fieldnames
all_fieldnames = list(existing_fieldnames) + ["test_config", "note"]
for field in all_fieldnames:
    if field not in existing_fieldnames:
        existing_fieldnames = all_fieldnames

with open(MANIFEST_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=existing_fieldnames)
    writer.writeheader()
    for row in existing:
        # fill missing fields for old rows
        for fn in existing_fieldnames:
            row.setdefault(fn, "")
        writer.writerow(row)
    for row in new_rows:
        for fn in existing_fieldnames:
            row.setdefault(fn, "")
        writer.writerow(row)

print(f"\nAppended {len(new_rows)} rows to manifest")
print(f"Total experiments: {len(existing) + len(new_rows)}")
