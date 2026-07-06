#!/usr/bin/env python3
"""
Fine-tuning experiment — lens-level sampling.

Base model: cumul_N4 (0409+0509+0523+0623)
Target domains: 0625 (7 lenses), New2 (9 lenses), 0407 (16 lenses)
k = number of LENSES sampled from target's training data.

Compare: fine-tune base model vs train from scratch, for each k.
"""

import os, sys, json, copy, random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Subset, DataLoader

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from train import SEED

# ── Config ─────────────────────────────────────────────────────────
FINETUNE_LR = 1e-4;   FINETUNE_EPOCHS = 80;   FINETUNE_EARLY_STOP = 15
SCRATCH_LR  = 1e-3;   SCRATCH_EPOCHS  = 120;  SCRATCH_EARLY_STOP  = 25
BATCH_SIZE = 64
RESULTS_DIR = PROJECT / "experiments" / "results"
ROOT = Path("/root")

# Target datasets: name → (dataset_dir, max_lenses_in_train)
TARGETS = {
    "0625": ("Dataset_Splitted_0625", 7),
    "New2": ("Dataset_Splitted_New2", 9),
    "0407": ("Dataset_Splitted_0407", 16),
}


def find_base_model():
    candidates = sorted(Path("/root/AA-YT/saved/models/GenExp_cumul_N4").glob("*/model_best.pth"))
    if not candidates:
        candidates = sorted(Path("/root/AA-YT/saved/models/GenExp_cumul_N4").glob("*/*/model_best.pth"))
    if not candidates:
        raise FileNotFoundError("cumul_N4 model not found")
    return str(max(candidates, key=os.path.getmtime))


def create_model():
    return module_arch.AAmodel(img_num=5, in_channels=1, num_classes=4)


def load_base_model(path):
    model = create_model()
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt.get("state_dict", ckpt), strict=True)
    return model


def build_lens_mapping(data_dir):
    """
    Build mapping: lens_name → list of sample indices.
    The XY_Dataset.folder_list gives all sample paths in order.
    Sample path: .../ScanType/LensName/LabelString/*.png
    Lens = the parent of the label folder, e.g. "AYZZO..." or "0403-10"
    """
    loader = module_data.AADataLoader(
        data_dir=[data_dir], img_num=5, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=4, training=True)
    dataset = loader.dataset
    # dataset.folder_list has all leaf folders
    lens_to_indices = defaultdict(list)
    for idx, folder in enumerate(dataset.folder_list):
        # folder = .../ScanType/LensName/LabelString
        lens_name = os.path.basename(os.path.dirname(folder))
        lens_to_indices[lens_name].append(idx)
    return dataset, dict(lens_to_indices)


def build_valid_loader(valid_dir):
    g = torch.Generator(); g.manual_seed(SEED)
    return module_data.AADataLoader(
        data_dir=[valid_dir], img_num=5, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=4, training=True, generator=g)


def sample_lenses(all_lenses, k, seed=42):
    """Randomly sample k lens names. Returns sorted list."""
    rng = random.Random(seed)
    if k >= len(all_lenses):
        return sorted(all_lenses)
    return sorted(rng.sample(all_lenses, k))


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


def train_model(model, train_subset, valid_loader, device, lr, epochs, early_stop):
    g = torch.Generator(); g.manual_seed(SEED)
    bs = min(BATCH_SIZE, len(train_subset))
    train_loader = DataLoader(train_subset, batch_size=bs, shuffle=True,
                              num_workers=2, generator=g)
    loss_fn = getattr(module_loss, "mse_loss_means")
    metric_fns = [getattr(module_metric, m) for m in
                  ["mae_xs", "mae_ys", "mae_xt", "mae_yt"]]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float("inf")
    best_metrics, best_state = None, None
    no_improve = 0

    for epoch in range(epochs):
        train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = evaluate(model, valid_loader, loss_fn, metric_fns, device)
        val_loss = val_metrics["loss"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = val_metrics
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= early_stop:
            break

    if best_state:
        model.load_state_dict(best_state)
    return best_metrics, epoch + 1


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    base_path = find_base_model()
    print(f"Base model: {base_path}")

    all_results = []

    for target_name, (ds_name, max_lenses) in TARGETS.items():
        print(f"\n{'='*60}")
        print(f"Target: {target_name} ({max_lenses} lenses in train)")
        print(f"{'='*60}")

        train_dir = str(ROOT / ds_name / "train")
        valid_dir = str(ROOT / ds_name / "valid")

        dataset, lens_map = build_lens_mapping(train_dir)
        valid_loader = build_valid_loader(valid_dir)
        all_lens_names = sorted(lens_map.keys())
        total_lenses = len(all_lens_names)
        total_samples = len(dataset)
        valid_samples = len(valid_loader.dataset)
        print(f"  Lenses: {total_lenses}, Train samples: {total_samples}, Valid samples: {valid_samples}")
        print(f"  Avg samples/lens: {total_samples/total_lenses:.0f}")

        loss_fn = getattr(module_loss, "mse_loss_means")
        metric_fns = [getattr(module_metric, m) for m in
                      ["mae_xs", "mae_ys", "mae_xt", "mae_yt"]]

        # ── 0-shot ──
        base_model = load_base_model(base_path).to(device)
        zs = evaluate(base_model, valid_loader, loss_fn, metric_fns, device)
        print(f"\n  k=0 lens (zero-shot): MAE_avg={zs.get('MAE_avg', 0):.4f}")
        all_results.append({"target": target_name, "k_lenses": 0, "k_samples": 0,
                            "method": "zero_shot", "metrics": zs})
        del base_model

        # ── k-lens experiments ──
        # k values: reasonable steps given total lens count
        if total_lenses <= 7:
            k_values = [1, 2, 3, 5, total_lenses]
        elif total_lenses <= 10:
            k_values = [1, 2, 3, 5, 7, total_lenses]
        else:
            k_values = [1, 2, 3, 5, 10, total_lenses]
        k_values = sorted(set(k_values))

        for k in k_values:
            if k == 0:
                continue
            selected_lenses = sample_lenses(all_lens_names, k)
            # Gather all sample indices for selected lenses
            indices = []
            for lens in selected_lenses:
                indices.extend(lens_map[lens])
            indices = sorted(indices)
            actual_k_lenses = len(selected_lenses)
            actual_k_samples = len(indices)
            train_subset = Subset(dataset, indices)

            # ── Fine-tune ──
            ft_model = load_base_model(base_path).to(device)
            ft_metrics, ft_ep = train_model(ft_model, train_subset, valid_loader, device,
                                            FINETUNE_LR, FINETUNE_EPOCHS, FINETUNE_EARLY_STOP)
            print(f"  k={actual_k_lenses:>2} lens ({actual_k_samples:>4} samples) | "
                  f"FT ({ft_ep:>3}ep): MAE_avg={ft_metrics.get('MAE_avg', 0):.4f}  "
                  f"x={ft_metrics.get('MAE_x', 0):.4f} y={ft_metrics.get('MAE_y', 0):.4f} "
                  f"u={ft_metrics.get('MAE_u', 0):.4f} v={ft_metrics.get('MAE_v', 0):.4f}")
            all_results.append({"target": target_name, "k_lenses": actual_k_lenses,
                                "k_samples": actual_k_samples, "method": "finetune",
                                "epochs": ft_ep, "metrics": ft_metrics,
                                "lenses": selected_lenses})
            del ft_model

            # ── From scratch ──
            sc_model = create_model().to(device)
            sc_metrics, sc_ep = train_model(sc_model, train_subset, valid_loader, device,
                                            SCRATCH_LR, SCRATCH_EPOCHS, SCRATCH_EARLY_STOP)
            print(f"  k={actual_k_lenses:>2} lens ({actual_k_samples:>4} samples) | "
                  f"SC ({sc_ep:>3}ep): MAE_avg={sc_metrics.get('MAE_avg', 0):.4f}  "
                  f"x={sc_metrics.get('MAE_x', 0):.4f} y={sc_metrics.get('MAE_y', 0):.4f} "
                  f"u={sc_metrics.get('MAE_u', 0):.4f} v={sc_metrics.get('MAE_v', 0):.4f}")
            all_results.append({"target": target_name, "k_lenses": actual_k_lenses,
                                "k_samples": actual_k_samples, "method": "scratch",
                                "epochs": sc_ep, "metrics": sc_metrics,
                                "lenses": selected_lenses})
            del sc_model
            torch.cuda.empty_cache()

    # ── Save ──
    out_path = RESULTS_DIR / "finetune_lens_experiment.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=4, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
