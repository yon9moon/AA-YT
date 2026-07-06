import sys
# sys.path.append('/mnt1/lwy/AA') # 如果路径变了，这里可能也需要根据实际情况注释或修改
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
from base import BaseDataLoader
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
to_image = ToPILImage()

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import glob
import cv2
from pathlib import Path
import random
from PIL import Image, ImageFilter
from collections import defaultdict
from train import SEED # 如果 train.py 不在同级目录可能会报错，建议作为参数传入或硬编码
import torch.nn.functional as F

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def test_transform(img):
    img = normalization(img)
    return img

def _worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ── Group-consistent augmentation ──────────────────────────────────

class GroupAugmentation:
    """
    Apply the SAME random augmentation to a group of images.
    Ensures all 5 images in a measurement share identical transforms.
    """
    def __init__(self, rotation_deg=3.0, brightness_range=(0.8, 1.2),
                 blur_sigma_range=(0.0, 1.5)):
        self.rotation_deg = rotation_deg
        self.brightness_range = brightness_range
        self.blur_sigma_range = blur_sigma_range
        self.active = (rotation_deg > 0 or brightness_range is not None or
                       (blur_sigma_range is not None and blur_sigma_range[1] > 0))

    def _sample_params(self):
        """Sample augmentation parameters once per group."""
        params = {}
        if self.rotation_deg > 0:
            params['angle'] = random.uniform(-self.rotation_deg, self.rotation_deg)
        if self.brightness_range is not None:
            params['brightness'] = random.uniform(*self.brightness_range)
        if self.blur_sigma_range is not None and self.blur_sigma_range[1] > 0:
            params['blur_sigma'] = random.uniform(*self.blur_sigma_range)
        return params

    def apply(self, pil_images, params=None):
        """Apply same augmentation to a list of PIL images."""
        if not self.active:
            return pil_images
        if params is None:
            params = self._sample_params()
        augmented = []
        for img in pil_images:
            if 'angle' in params and params['angle'] != 0:
                img = img.rotate(params['angle'], resample=Image.BILINEAR, expand=False)
            if 'blur_sigma' in params and params['blur_sigma'] > 0.1:
                img = img.filter(ImageFilter.GaussianBlur(radius=params['blur_sigma']))
            if 'brightness' in params:
                factor = params['brightness']
                if factor != 1.0:
                    img = img.point(lambda p: min(255, max(0, int(p * factor))))
            augmented.append(img)
        return augmented

class AADataLoader(DataLoader):
    def __init__(self, data_dir=None, data_dirs=None, img_num=5, batch_size=64, generator=None,
                 shuffle=True, num_workers=8, training=True, group_aug=None):

        # 支持 data_dir (单个或列表) 或 data_dirs (列表)
        if data_dirs is not None:
            self.data_dirs = data_dirs if isinstance(data_dirs, list) else [data_dirs]
        elif data_dir is not None:
            self.data_dirs = data_dir if isinstance(data_dir, list) else [data_dir]
        else:
            raise ValueError("Either 'data_dir' or 'data_dirs' must be provided")

        self.img_num = img_num
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.dataset = XY_Dataset(self.data_dirs, img_num, batch_size, group_aug=group_aug)
        
        # worker_init_fn 需要在外部定义或者 import，如果原来代码里有 _worker_init_fn 请确保它存在
        # 这里假设你原本的代码环境里有这个函数，如果没有，可以去掉 worker_init_fn 参数
        super().__init__(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                         worker_init_fn=_worker_init_fn, 
                         generator=generator)

        self.info = {
            "list_path_data": self.data_dirs,
            "len(dataset)": len(self.dataset)
        }

    def __str__(self):
        return  "AADataLoader:\n  "+ str(self.info)

    
class XY_Dataset(Dataset):
    def __init__(self, data_dirs, img_num, batch_size, group_aug=None):
        # 支持传入单个路径字符串或路径列表
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]
        elif not isinstance(data_dirs, list):
            raise ValueError("data_dirs must be a string or list of strings")

        self.data_dirs = data_dirs
        self.img_num = img_num
        self.batch_size = batch_size
        self.group_aug = group_aug  # GroupAugmentation or None

        # --- 修改点 1: 支持多个路径 ---
        # 遍历所有 data_dirs，收集包含 .png 的叶子文件夹
        self.folder_list = []
        for data_dir in self.data_dirs:
            for root, dirs, files in os.walk(data_dir):
                has_images = any(f.endswith('.png') for f in files)
                if has_images:
                    self.folder_list.append(root)

        # 排序以保证多线程加载时的确定性
        self.folder_list.sort()

        print(f"Loaded {len(self.folder_list)} data samples from {len(self.data_dirs)} directories: {self.data_dirs}")

        # 定义数据转换
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.folder_list)

    def __getitem__(self, idx):
        images = []
        folder_path = self.folder_list[idx]
        
        # --- 修改点 2: 健壮的路径解析 ---
        # 假设路径结构: .../LensID/Label_String/xxx.png
        
        # 1. 获取文件夹名作为 Label (例如 "-0.002 0.000 0.000 0.000")
        folder_name = os.path.basename(folder_path)
        
        # 2. 获取上一级文件夹名作为 Lens ID (例如 "AYZZO...")
        # 这样写不再依赖绝对路径的层级深度 (parts[:6])
        lens_dir = os.path.basename(os.path.dirname(folder_path))

        # 3. 标签处理
        # 之前的逻辑是取前两个值 (x, y)，这里保持一致
        # 新的数据集文件夹名由4个值组成 (x y u v)
        try:
            label_strings = folder_name.split(' ') 
            # 取前两个做回归 (根据你之前的代码逻辑)
            # 如果需要4维回归，请改为 labels = [float(label) for label in label_strings]
            labels = [float(label) for label in label_strings[:4]] 
            labels = torch.tensor(labels) / 0.1 #
        except Exception as e:
            print(f"Label parse error at {folder_path}: {e}")
            # 发生错误时返回零张量防止程序崩溃
            labels = torch.zeros(4)

        # --- 修改点 3: 图片加载逻辑 ---
        # 确保只读取 png 并排序
        all_files = os.listdir(folder_path)
        image_files = [f for f in all_files if f.endswith('.png') and len(f) == 6]
        image_files = sorted(image_files) # 排序很重要，保证顺序是 02, 03, 08...

        # 检查图片数量
        # 新数据集只有5张图，如果你设置 img_num=5 没问题。
        # 如果 img_num 还是 3，需要确认取哪几张。这里默认取前 img_num 张。
        current_len = len(image_files)
        if current_len < self.img_num:
            # 如果图片不够，通常是生成脚本出了问题，或者读取了错误的文件夹
            raise ValueError(f"Error: Not enough images ({current_len} < {self.img_num}) in {folder_path}")

        # 循环加载图片 — 先加载为 PIL，统一增强后再转 tensor
        pil_images = []
        for i in range(self.img_num):
            img_path = os.path.join(folder_path, image_files[i])
            pil_images.append(Image.open(img_path).convert('L'))

        # 施加 group-consistent 增强（相同随机参数应用于全组）
        if self.group_aug is not None and self.group_aug.active:
            pil_images = self.group_aug.apply(pil_images)

        # 转 tensor
        for image in pil_images:
            image = self.transform(image)
            image = image.float()
            images.append(image)

        # 堆叠图片 (Channel 维度堆叠: C=img_num, H, W)
        images = torch.stack(images)

        return images, labels, lens_dir