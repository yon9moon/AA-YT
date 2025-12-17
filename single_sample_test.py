#!/usr/bin/env python3
import argparse
import os
import random
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import data_loader.data_loaders as module_data
import model.model as module_arch
from utils import read_json

LABEL_SCALE = 20.0
ORDERED_FILES = ["0-p.png", "5-p.png", "6-p.png", "7-p.png", "8-p.png"]
POSITION_MAP = {
    "0-p.png": (1, 1),
    "5-p.png": (0, 0),
    "6-p.png": (2, 0),
    "7-p.png": (2, 2),
    "8-p.png": (0, 2),
}


def parse_label_from_folder(folder: Path, num_classes: int) -> np.ndarray:
    name = folder.name.replace('_', ' ')
    if ',' in name:
        parts = [p.strip() for p in name.split(',') if p.strip()]
    else:
        parts = [p for p in name.split() if p]
    if len(parts) < num_classes:
        raise ValueError(f"Folder {folder} does not contain {num_classes} label components")
    return np.array([float(p) for p in parts[:num_classes]], dtype=np.float32)


class DatasetIndex:
    """根据当前 DA_70028_Dataset 构建的索引结构，用于单样本最近邻查询。

    - all_folders: 数据集中所有叶子文件夹路径列表（字符串）
    - label_to_folders: (x, y, ...) -> [folder_path, ...]

    标签解析规则与训练时保持一致：直接从文件夹名解析物理坐标，不做 /20 缩放，
    后续通过 LABEL_SCALE 把模型预测还原到同一量纲后再做最近邻搜索。
    """

    def __init__(self, dataset: object, num_classes: int):
        self.all_folders: List[str] = []
        self.label_to_folders: dict[tuple, List[str]] = {}

        folder_list = getattr(dataset, 'folder_list', [])
        for folder in folder_list:
            folder_path = Path(folder)
            try:
                label_arr = parse_label_from_folder(folder_path, num_classes)
            except Exception:
                # 解析失败的样本直接跳过（例如名字不符合“x,y”或“x y”格式）
                continue

            key = tuple(label_arr.tolist())
            folder_str = str(folder_path)
            self.all_folders.append(folder_str)
            self.label_to_folders.setdefault(key, []).append(folder_str)


def load_image_tensor(folder: Path, img_num: int, transform, file_order: Optional[List[str]] = None) -> tuple[torch.Tensor, List[str]]:
    if file_order is None:
        candidates = sorted(f for f in os.listdir(folder) if f.endswith('-p.png'))
        if len(candidates) < img_num:
            raise FileNotFoundError(f"Folder {folder} has only {len(candidates)} '-p.png' files")
        file_list = candidates[:img_num]
    else:
        missing = [name for name in file_order if not (folder / name).exists()]
        if missing:
            raise FileNotFoundError(f"Missing required files in {folder}: {missing}")
        file_list = file_order

    tensors = []
    for fname in file_list:
        img = Image.open(folder / fname).convert('L')
        tensors.append(transform(img).float())
    stacked = torch.stack(tensors, dim=0)
    return stacked, file_list


def find_nearest_label(adjusted: np.ndarray, label_map: dict[tuple, list[str]], lens_id: str) -> tuple[tuple, Path]:
    best_key = None
    best_dist = float('inf')
    best_folder: Optional[Path] = None

    for key, folders in label_map.items():
        matching_folders = [Path(p) for p in folders if Path(p).parent.name == lens_id]
        if not matching_folders:
            continue
        dist = np.linalg.norm(np.array(key, dtype=np.float32) - adjusted)
        if dist < best_dist:
            best_dist = dist
            best_key = key
            best_folder = matching_folders[0]

    if best_key is None or best_folder is None:
        raise RuntimeError(f"No matching labels found for lens {lens_id}")

    return best_key, best_folder


def save_composite_image(src_folder: Path, file_names: list[str], output_path: Path) -> None:
    images = {name: Image.open(src_folder / name).convert('L') for name in file_names}
    any_image = next(iter(images.values()))
    tile_width, tile_height = any_image.width, any_image.height
    canvas_width = tile_width * 3
    canvas_height = tile_height * 3

    canvas = Image.new('L', (canvas_width, canvas_height), color=255)

    for name, (col, row) in POSITION_MAP.items():
        if name not in images:
            continue
        img = images[name]
        x_offset = col * tile_width
        y_offset = row * tile_height
        canvas.paste(img, (x_offset, y_offset))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def build_model(config: dict) -> torch.nn.Module:
    arch_cfg = config['arch']
    model_cls = getattr(module_arch, arch_cfg['type'])
    args = arch_cfg.get('args', {})
    return model_cls(**args)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        stripped = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
        model.load_state_dict(stripped)


def main() -> None:
    parser = argparse.ArgumentParser(description="Single sample inference helper")
    parser.add_argument('-c', '--config', required=True, help='Path to config JSON')
    parser.add_argument('-r', '--resume', required=True, help='Path to checkpoint file')
    parser.add_argument('-d', '--device', default=None, help='CUDA device indices (optional)')
    parser.add_argument('--data-dir', default=None, help='Override data directory for sampling')
    parser.add_argument('--sample-folder', required=True, help='Specific leaf folder to use')
    parser.add_argument('--output-dir', default='analysis/single_sample', help='Directory to store outputs')
    parser.add_argument('--seed', type=int, default=None, help='Optional random seed')
    args = parser.parse_args()

    if args.device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    config = read_json(Path(args.config))
    data_cfg = config['data_loader']['args']
    img_num = int(data_cfg.get('img_num', 5))
    num_classes = int(data_cfg.get('num_classes', 2))
    batch_size = int(data_cfg.get('batch_size', 1))
    split = data_cfg.get('split', 'test')

    if img_num != len(ORDERED_FILES):
        raise ValueError(f"Config expects {img_num} images per sample, but composite layout requires {len(ORDERED_FILES)}")

    # 使用当前的 DA_70028_Dataset 来获得文件列表与 transform，保证与训练/测试完全一致
    dataset = module_data.DA_70028_Dataset(data_cfg['data_dir'], img_num, batch_size, split)
    transform = dataset.transform if hasattr(dataset, 'transform') else transforms.ToTensor()
    index = DatasetIndex(dataset, num_classes)

    sample_folder = Path(args.sample_folder).resolve()
    if not sample_folder.is_dir():
        raise FileNotFoundError(f"Specified sample folder {sample_folder} is missing")

    # 样本必须在当前数据集包含的叶子目录中
    dataset_folders = {Path(path).resolve() for path in index.all_folders}
    if sample_folder.resolve() not in dataset_folders:
        raise ValueError(f"Sample folder {sample_folder} not found within dataset roots {data_cfg['data_dir']}")

    input_label = parse_label_from_folder(sample_folder, num_classes)
    input_tensor, input_files = load_image_tensor(sample_folder, img_num, transform, ORDERED_FILES)
    batch = input_tensor.unsqueeze(0)

    model = build_model(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    load_checkpoint(model, Path(args.resume), device)

    with torch.no_grad():
        output = model(batch.to(device))
        if isinstance(output, (list, tuple)):
            output = output[0]
    prediction = output.squeeze(0).detach().cpu().numpy() * LABEL_SCALE

    adjusted = input_label - prediction
    lens_id = sample_folder.parent.name
    nearest_label, nearest_folder_path = find_nearest_label(adjusted, index.label_to_folders, lens_id)
    nearest_folder = nearest_folder_path.resolve()

    _, nearest_files = load_image_tensor(nearest_folder, img_num, transform, ORDERED_FILES)

    output_root = Path(args.output_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 尽量根据 data_loader 的 data_dir 生成相对路径名；找不到匹配根目录就退回文件夹名
    data_roots = data_cfg['data_dir']
    if isinstance(data_roots, str):
        data_roots = [data_roots]
    relative_name = sample_folder.name
    for root in data_roots:
        root_path = Path(root).resolve()
        try:
            rel = sample_folder.relative_to(root_path)
            relative_name = rel.as_posix().replace('/', '__')
            break
        except ValueError:
            continue
    run_dir = output_root / f"{timestamp}__{relative_name}"
    input_path = run_dir / 'input_composite.png'
    matched_path = run_dir / 'matched_composite.png'
    run_dir.mkdir(parents=True, exist_ok=True)

    save_composite_image(sample_folder, input_files, input_path)
    save_composite_image(nearest_folder, nearest_files, matched_path)

    def fmt(values: np.ndarray) -> str:
        return '[' + ', '.join(f"{v:.3f}" for v in values.tolist()) + ']'

    print(f"Sample folder: {sample_folder}")
    print(f"Input label: {fmt(input_label)}")
    print(f"Model prediction: {fmt(prediction)}")
    print(f"Adjusted label (input - prediction): {fmt(adjusted)}")
    print(f"Nearest label: {fmt(np.array(nearest_label, dtype=np.float32))}")
    print(f"Nearest folder: {nearest_folder}")
    print(f"Saved input image row to: {input_path}")
    print(f"Saved matched image row to: {matched_path}")


if __name__ == '__main__':
    main()
