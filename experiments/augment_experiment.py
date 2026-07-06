#!/usr/bin/env python3
"""
Data augmentation experiment: train base models with different group-consistent
augmentations, then evaluate zero-shot and fine-tuning performance.

Augmentation variants:
  1. None      — baseline (cumul_N4, already done)
  2. Rot3      — random rotation ±3°
  3. Bright    — random brightness [0.8, 1.2]
  4. Blur      — random Gaussian blur sigma [0.0, 1.5]
  5. Rot+Bright+Blur — all three combined

Each variant trains on 0409+0509+0523+0623, then evaluates on 0407.
"""

import os, sys, json, copy, random, argparse
from pathlib import Path

import torch
from torch.utils.data import Subset, DataLoader

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import data_loader.data_loaders as module_data
from data_loader.data_loaders import GroupAugmentation
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from utils import read_json, write_json
from train import SEED

# ── Config ─────────────────────────────────────────────────────────
ROOT = Path("/root")
BASE_DATASETS = ["Dataset_Splitted_0409", "Dataset_Splitted_0509",
                 "Dataset_Splitted_0523", "Dataset_Splitted_0623"]
TARGET_DATASET = "Dataset_Splitted_0407"

TRAIN_EPOCHS = 200
TRAIN_LR = 1e-3
EARLY_STOP = 30
BATCH_SIZE = 64

FINETUNE_LR = 1e-4
FINETUNE_EPOCHS = 80
FINETUNE_EARLY_STOP = 15

# Augmentation variants
AUG_VARIANTS = {
    "none":    GroupAugmentation(rotation_deg=0, brightness_range=None, blur_sigma_range=(0, 0)),
    "rot3":    GroupAugmentation(rotation_deg=3, brightness_range=None, blur_sigma_range=(0, 0)),
    "bright":  GroupAugmentation(rotation_deg=0, brightness_range=(0.8, 1.2), blur_sigma_range=(0, 0)),
    "blur":    GroupAugmentation(rotation_deg=0, brightness_range=None, blur_sigma_range=(0.1, 1.5)),
    "combo":   GroupAugmentation(rotation_deg=3, brightness_range=(0.8, 1.2), blur_sigma_range=(0.1, 1.5)),
}

SAVE_DIR = PROJECT / "saved" / "models"
RESULTS_DIR = PROJECT / "experiments" / "results"


def create_model():
    return module_arch.AAmodel(img_num=5, in_channels=1, num_classes=4)


def build_data_loader(data_dirs, group_aug, shuffle=True):
    g = torch.Generator(); g.manual_seed(SEED)
    train_dirs = [str(ROOT / d / "train") for d in data_dirs]
    return module_data.AADataLoader(
        data_dir=train_dirs, img_num=5, batch_size=BATCH_SIZE,
        shuffle=shuffle, num_workers=8, training=True,
        generator=g, group_aug=group_aug)


def build_valid_loader(data_dirs):
    g = torch.Generator(); g.manual_seed(SEED)
    valid_dirs = [str(ROOT / d / "valid") for d in data_dirs]
    return module_data.AADataLoader(
        data_dir=valid_dirs, img_num=5, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=8, training=True,
        generator=g, group_aug=None)  # No augmentation for validation


def build_target_valid_loader(target_ds):
    g = torch.Generator(); g.manual_seed(SEED)
    return module_data.AADataLoader(
        data_dir=[str(ROOT / target_ds / "valid")],
        img_num=5, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=4, training=True,
        generator=g, group_aug=None)


def build_lens_mapping(target_ds):
    """Build lens→indices mapping for fine-tuning sampling."""
    loader = module_data.AADataLoader(
        data_dir=[str(ROOT / target_ds / "train")],
        img_num=5, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, training=True, group_aug=None)
    dataset = loader.dataset
    from collections import defaultdict
    lens_map = defaultdict(list)
    for idx, folder in enumerate(dataset.folder_list):
        lens_name = os.path.basename(os.path.dirname(folder))
        lens_map[lens_name].append(idx)
    return dataset, dict(lens_map)


def sample_lenses(all_lenses, k, seed=42):
    rng = random.Random(seed)
    if k >= len(all_lenses):
        return sorted(all_lenses)
    return sorted(rng.sample(all_lenses, k))


# ── Training ───────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss, n = 0.0, 0
    for data, target, _ in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(data), target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.size(0); n += data.size(0)
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, loss_fn, metric_fns, device):
    model.eval()
    total_loss, n = 0.0, 0
    total_metrics = torch.zeros(len(metric_fns))
    all_out, all_tgt = [], []
    for data, target, _ in loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        all_out.append(output.cpu()); all_tgt.append(target.cpu())
        loss = loss_fn(output, target)
        total_loss += loss.item() * data.size(0); n += data.size(0)
        for i, met in enumerate(metric_fns):
            mv = met(output, target)
            total_metrics[i] += (mv.item() if torch.is_tensor(mv) else mv) * data.size(0)
    results = {"loss": total_loss / n}
    for i, met in enumerate(metric_fns):
        results[met.__name__] = total_metrics[i].item() / n
    if all_out:
        std = module_metric.mae_with_std(torch.cat(all_out, 0), torch.cat(all_tgt, 0))
        results.update({k: float(v) for k, v in std.items()})
    return results


def train_full(model, train_loader, valid_loader, device, lr, epochs, early_stop,
               save_name=None):
    """Full training loop, returns best metrics and model state."""
    loss_fn = getattr(module_loss, "mse_loss_means")
    metric_fns = [getattr(module_metric, m) for m in
                  ["mae_xs", "mae_ys", "mae_xt", "mae_yt"]]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_val_loss = float("inf")
    best_metrics = None
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        scheduler.step()
        val_metrics = evaluate(model, valid_loader, loss_fn, metric_fns, device)
        val_loss = val_metrics["loss"]

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = val_metrics
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
            if save_name:
                save_dir = SAVE_DIR / save_name
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": best_state, "metrics": best_metrics},
                           save_dir / "model_best.pth")
        else:
            no_improve += 1
        if no_improve >= early_stop:
            break

    if best_state:
        model.load_state_dict(best_state)
    return best_metrics, epoch + 1


def finetune_on_target(base_model, target_ds, k_lenses, device):
    """Fine-tune base model with k lenses from target, return metrics."""
    full_dataset, lens_map = build_lens_mapping(target_ds)
    valid_loader = build_target_valid_loader(target_ds)
    all_lenses = sorted(lens_map.keys())

    selected = sample_lenses(all_lenses, k_lenses)
    indices = sorted(sum((lens_map[l] for l in selected), []))
    train_subset = Subset(full_dataset, indices)

    model = copy.deepcopy(base_model)
    loss_fn = getattr(module_loss, "mse_loss_means")
    metric_fns = [getattr(module_metric, m) for m in
                  ["mae_xs", "mae_ys", "mae_xt", "mae_yt"]]
    bs = min(BATCH_SIZE, len(train_subset))
    g = torch.Generator(); g.manual_seed(SEED)
    train_loader = DataLoader(train_subset, batch_size=bs, shuffle=True,
                              num_workers=2, generator=g)
    optimizer = torch.optim.Adam(model.parameters(), lr=FINETUNE_LR)
    best_val_loss = float("inf")
    best_metrics, best_state = None, None
    no_improve = 0

    for epoch in range(FINETUNE_EPOCHS):
        train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = evaluate(model, valid_loader, loss_fn, metric_fns, device)
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_metrics = val_metrics
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= FINETUNE_EARLY_STOP:
            break

    if best_state:
        model.load_state_dict(best_state)
    return best_metrics, epoch + 1, len(indices)


# ── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train base models")
    parser.add_argument("--finetune", action="store_true", help="Run fine-tuning eval")
    parser.add_argument("--variants", nargs="+", default=list(AUG_VARIANTS.keys()),
                        help="Which augmentation variants to run")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_results = []

    for aug_name in args.variants:
        group_aug = AUG_VARIANTS[aug_name]
        model_name = f"GenExp_aug_{aug_name}"

        print(f"\n{'='*60}")
        print(f"Augmentation: {aug_name}")
        print(f"{'='*60}")

        # ── Train base model ──
        if args.train:
            print("  Training base model...")
            train_loader = build_data_loader(BASE_DATASETS, group_aug, shuffle=True)
            valid_loader = build_valid_loader(BASE_DATASETS)
            print(f"  Train samples: {len(train_loader.dataset)}, Valid: {len(valid_loader.dataset)}")

            model = create_model().to(device)
            metrics, epochs = train_full(model, train_loader, valid_loader, device,
                                        TRAIN_LR, TRAIN_EPOCHS, EARLY_STOP,
                                        save_name=model_name)
            print(f"  Trained {epochs} epochs, val MAE_avg={metrics.get('MAE_avg', 0):.4f}")
            del model

        # ── Evaluate ──
        model_path = SAVE_DIR / model_name / "model_best.pth"
        if not model_path.exists():
            print(f"  Model not found at {model_path}, skipping")
            continue

        base_model = create_model()
        ckpt = torch.load(model_path, map_location="cpu")
        base_model.load_state_dict(ckpt.get("state_dict", ckpt), strict=True)
        base_model = base_model.to(device)

        loss_fn = getattr(module_loss, "mse_loss_means")
        metric_fns = [getattr(module_metric, m) for m in
                      ["mae_xs", "mae_ys", "mae_xt", "mae_yt"]]

        # Zero-shot on 0407
        target_valid = build_target_valid_loader(TARGET_DATASET)
        zs = evaluate(base_model, target_valid, loss_fn, metric_fns, device)
        print(f"  Zero-shot on 0407: MAE_avg={zs.get('MAE_avg', 0):.4f}")
        all_results.append({"aug": aug_name, "target": "0407", "k_lenses": 0,
                            "method": "zero_shot", "metrics": zs})

        # Fine-tuning on 0407
        if args.finetune:
            total_lenses = len(build_lens_mapping(TARGET_DATASET)[1])
            k_vals = [1, 2, 3, 5, total_lenses]
            k_vals = sorted(set(k_vals))
            for k in k_vals:
                ft_metrics, ft_ep, n_samples = finetune_on_target(
                    base_model, TARGET_DATASET, k, device)
                print(f"  FT k={k} lens ({n_samples} samp, {ft_ep}ep): "
                      f"MAE_avg={ft_metrics.get('MAE_avg', 0):.4f}")
                all_results.append({"aug": aug_name, "target": "0407",
                                    "k_lenses": k, "k_samples": n_samples,
                                    "method": "finetune", "epochs": ft_ep,
                                    "metrics": ft_metrics})
                torch.cuda.empty_cache()

        del base_model
        torch.cuda.empty_cache()

    # ── Save ──
    out_path = RESULTS_DIR / "augment_experiment.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
