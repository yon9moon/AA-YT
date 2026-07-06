#!/usr/bin/env python3
"""Fine-tuning with augmentation: compare base + FT augmentation combos."""

import os, sys, json, copy, random
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import Subset, DataLoader

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import data_loader.data_loaders as module_data
from data_loader.data_loaders import GroupAugmentation
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from train import SEED

ROOT = Path("/root")
TARGET = "Dataset_Splitted_0407"
BATCH_SIZE = 64
FT_LR = 1e-4; FT_EPOCHS = 80; FT_EARLY_STOP = 15
SAVE_DIR = PROJECT / "saved" / "models"
RESULTS_DIR = PROJECT / "experiments" / "results"

AUGS = {
    "none":   GroupAugmentation(rotation_deg=0, brightness_range=None, blur_sigma_range=(0,0)),
    "rot3":   GroupAugmentation(rotation_deg=3, brightness_range=None, blur_sigma_range=(0,0)),
    "bright": GroupAugmentation(rotation_deg=0, brightness_range=(0.8, 1.2), blur_sigma_range=(0,0)),
}

# Combos to test: (base_aug, ft_aug)
COMBOS = [
    ("none", "none"),       # baseline
    ("rot3", "none"),       # base aug only (already done)
    ("rot3", "rot3"),       # both aug
    ("bright", "none"),     # base bright only (already done)
    ("bright", "bright"),   # both bright
    ("none", "rot3"),       # FT aug only — does FT augmentation alone help?
]


def create_model():
    return module_arch.AAmodel(img_num=5, in_channels=1, num_classes=4)


def build_lens_mapping(group_aug=None):
    loader = module_data.AADataLoader(
        data_dir=[str(ROOT / TARGET / "train")],
        img_num=5, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, training=True, group_aug=group_aug)
    dataset = loader.dataset
    lens_map = defaultdict(list)
    for idx, folder in enumerate(dataset.folder_list):
        lens_name = os.path.basename(os.path.dirname(folder))
        lens_map[lens_name].append(idx)
    return dataset, dict(lens_map)


def build_target_valid_loader():
    g = torch.Generator(); g.manual_seed(SEED)
    return module_data.AADataLoader(
        data_dir=[str(ROOT / TARGET / "valid")],
        img_num=5, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, training=True, generator=g, group_aug=None)


def sample_lenses(all_lenses, k, seed=42):
    rng = random.Random(seed)
    if k >= len(all_lenses): return sorted(all_lenses)
    return sorted(rng.sample(all_lenses, k))


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train(); total_loss, n = 0.0, 0
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
    model.eval(); total_loss, n = 0.0, 0
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


def finetune_with_aug(model, k_lenses, ft_aug, device):
    """Fine-tune with k lenses + augmentation on training data."""
    full_dataset, lens_map = build_lens_mapping(group_aug=ft_aug if ft_aug.active else None)
    valid_loader = build_target_valid_loader()
    all_lenses = sorted(lens_map.keys())

    selected = sample_lenses(all_lenses, k_lenses)
    indices = sorted(sum((lens_map[l] for l in selected), []))
    train_subset = Subset(full_dataset, indices)

    loss_fn = getattr(module_loss, "mse_loss_means")
    metric_fns = [getattr(module_metric, m) for m in
                  ["mae_xs", "mae_ys", "mae_xt", "mae_yt"]]
    bs = min(BATCH_SIZE, len(train_subset))
    g = torch.Generator(); g.manual_seed(SEED)
    train_loader = DataLoader(train_subset, batch_size=bs, shuffle=True,
                              num_workers=2, generator=g)
    optimizer = torch.optim.Adam(model.parameters(), lr=FT_LR)
    best_val_loss, best_metrics, best_state = float("inf"), None, None
    no_improve = 0

    for epoch in range(FT_EPOCHS):
        train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val = evaluate(model, valid_loader, loss_fn, metric_fns, device)
        if val["loss"] < best_val_loss:
            best_val_loss, best_metrics = val["loss"], val
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= FT_EARLY_STOP:
            break
    if best_state:
        model.load_state_dict(best_state)
    return best_metrics, epoch + 1, len(indices)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    total_lenses = len(build_lens_mapping()[1])
    k_vals = sorted(set([1, 2, 3, 5, total_lenses]))
    all_results = []

    for base_aug_name, ft_aug_name in COMBOS:
        base_aug = AUGS[base_aug_name]
        ft_aug = AUGS[ft_aug_name]
        model_name = f"GenExp_aug_{base_aug_name}" if base_aug_name != "none" else "GenExp_cumul_N4"
        model_path = SAVE_DIR / model_name / "model_best.pth"

        if not model_path.exists():
            print(f"  SKIP {base_aug_name}+{ft_aug_name}: model not found at {model_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Base aug: {base_aug_name} | FT aug: {ft_aug_name}")
        print(f"{'='*60}")

        base_model = create_model()
        ckpt = torch.load(model_path, map_location="cpu")
        base_model.load_state_dict(ckpt.get("state_dict", ckpt), strict=True)
        base_model = base_model.to(device)

        # Zero-shot
        loss_fn = getattr(module_loss, "mse_loss_means")
        metric_fns = [getattr(module_metric, m) for m in
                      ["mae_xs", "mae_ys", "mae_xt", "mae_yt"]]
        valid_loader = build_target_valid_loader()
        zs = evaluate(base_model, valid_loader, loss_fn, metric_fns, device)
        print(f"  0-shot: MAE_avg={zs.get('MAE_avg', 0):.4f}")
        all_results.append({"base_aug": base_aug_name, "ft_aug": ft_aug_name,
                            "k_lenses": 0, "method": "zero_shot", "metrics": zs})

        for k in k_vals:
            model = copy.deepcopy(base_model).to(device)
            metrics, ep, n_samp = finetune_with_aug(model, k, ft_aug, device)
            print(f"  k={k:>2} lens ({n_samp:>5} samp, {ep:>3}ep): "
                  f"MAE_avg={metrics.get('MAE_avg', 0):.4f}")
            all_results.append({"base_aug": base_aug_name, "ft_aug": ft_aug_name,
                                "k_lenses": k, "k_samples": n_samp,
                                "method": "finetune", "epochs": ep, "metrics": metrics})
            del model
            torch.cuda.empty_cache()

        del base_model
        torch.cuda.empty_cache()

    out_path = RESULTS_DIR / "augment_ft_experiment.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
