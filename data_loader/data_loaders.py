import sys
sys.path.append('/mnt1/lwy/AA')
from torchvision import datasets, transforms
from base import BaseDataLoader
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
to_image = ToPILImage()

import numpy as np
# import pandas as pd
import torch
# utils是工具包
from torch.utils.data import Dataset  # Dataset是个抽象类，只能用于继承
from torch.utils.data import DataLoader # DataLoader需实例化，用于加载数据
import os
import glob
import cv2
from pathlib import Path
import random
from operator import itemgetter
from PIL import Image
from collections import defaultdict
from train import SEED
import torch.nn.functional as F

# ==================== 退化函数（batch 级） ====================
def batch_jpg_down_up(x, scale):
    B, N, C, H, W = x.shape
    x_flat = x.view(B*N, C, H, W)
    new_h, new_w = int(H * scale), int(W * scale)
    down = F.interpolate(x_flat, size=(new_h, new_w), mode='nearest')
    up = F.interpolate(down, size=(H, W), mode='nearest')
    return up.view(B, N, C, H, W)

def batch_gaussian_blur(x, kernel_size, sigma):
    B, N, C, H, W = x.shape
    x_flat = x.view(B*N, C, H, W)
    device = x_flat.device

    # 1D kernel
    k = kernel_size
    t = torch.arange(k, device=device, dtype=torch.float32)
    kernel = torch.exp(-((t - k//2)**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, k, 1).repeat(C, 1, 1, 1)

    pad = k // 2
    x_pad = F.pad(x_flat, (pad, pad, pad, pad), mode='reflect')
    blurred = F.conv2d(x_pad, kernel, groups=C)
    kernel_t = kernel.transpose(2, 3)
    blurred = F.conv2d(blurred, kernel_t, groups=C)
    return blurred.view(B, N, C, H, W)

def batch_gaussian_noise(x, std):
    noise = torch.normal(0, std, size=x.shape, device=x.device, dtype=x.dtype)
    return x + noise

def batch_random_mask(x, ratio):
    mask = torch.rand_like(x) < ratio
    return x * (~mask).float()

# 退化池
DEGRADE_OPS = {
    'jpg':     lambda x: batch_jpg_down_up(x, scale=random.uniform(0.4, 0.7)),
    'blur':    lambda x: batch_gaussian_blur(x, kernel_size=random.choice([3,5,7]), sigma=random.uniform(0.5, 2.0)),
    'noise':   lambda x: batch_gaussian_noise(x, std=random.uniform(0.02, 0.08)),
    'mask':    lambda x: batch_random_mask(x, ratio=random.uniform(0.05, 0.20)),
}

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def test_transform(img):
    img = normalization(img)
    return img

def crop_offaxis(img, size):
    row_sums = np.sum(img, axis=1)
    col_sums = np.sum(img, axis=0)
    max_row_index = np.argmax(row_sums)
    max_col_index = np.argmax(col_sums)

    half_size = size // 2

    # 计算裁剪区域的起始和结束索引，以 max_row_index 和 max_col_index 为中心
    start_row = max(max_row_index - half_size, 0)
    end_row = min(max_row_index + half_size, img.shape[0])
    start_col = max(max_col_index - half_size, 0)
    end_col = min(max_col_index + half_size, img.shape[1])

    cropped_img = img[start_row:end_row, start_col:end_col]

    # 创建一个全零图像用于填充
    padded_img = np.zeros((size, size), dtype=img.dtype)

    # 计算填充图像的起始位置，使裁剪区域居中
    row_offset = (size - cropped_img.shape[0]) // 2
    col_offset = (size - cropped_img.shape[1]) // 2

    padded_img[row_offset:row_offset + cropped_img.shape[0], col_offset:col_offset + cropped_img.shape[1]] = cropped_img

    return padded_img

def crop_centroid(img, size):

    if isinstance(img, Image.Image):
        img = np.array(img)
    else:
        img = np.asarray(img)

    noise_floor = 20 
    _, processed_img = cv2.threshold(img.astype(np.uint8), noise_floor, 255, cv2.THRESH_TOZERO)
    
    # 步骤 2: 精准计算质心
    M = cv2.moments(processed_img)

    # 确保图像不为空，避免除以零的错误
    if M["m00"] != 0:
        cX = M["m10"] / M["m00"]
        cY = M["m01"] / M["m00"]
    else:
        # 如果图像全黑或找不到亮斑，就使用图像的几何中心作为备用方案
        h, w = img.shape
        cX, cY = w / 2, h / 2

    # 将精确的浮点数坐标四舍五入为整数，作为裁剪操作的中心
    center_row = int(round(cY))
    center_col = int(round(cX))
    
    # 步骤 3: 【正确的】裁剪逻辑
    half_size = size // 2
    
    # 计算裁剪窗口的四角坐标，并确保不越界
    start_row = max(center_row - half_size, 0)
    end_row = min(center_row + half_size, img.shape[0])
    start_col = max(center_col - half_size, 0)
    end_col = min(center_col + half_size, img.shape[1])
    
    # 使用正确的Numpy语法进行裁剪
    cropped_img = img[start_row:end_row, start_col:end_col]

    # 步骤 4: 【正确的】粘贴逻辑
    # 创建一个目标大小的全黑背景
    padded_img = np.zeros((size, size), dtype=img.dtype)

    # 计算需要粘贴的区域的起始坐标，以确保裁剪出的内容在新画布上居中
    pad_start_row = (size - cropped_img.shape[0]) // 2
    pad_start_col = (size - cropped_img.shape[1]) // 2
    
    # 使用正确的Numpy语法将裁剪内容粘贴到新画布中心
    padded_img[pad_start_row : pad_start_row + cropped_img.shape[0], 
               pad_start_col : pad_start_col + cropped_img.shape[1]] = cropped_img

    return padded_img


def crop_center(img, size):
    row_sums = np.sum(img, axis=1)
    max_row_index = np.argmax(row_sums)

    center_col_index = img.shape[1] // 2
    left_img = img[:, :center_col_index]
    right_img = img[:, center_col_index:]

    left_col_sums = np.sum(left_img, axis=0)
    right_col_sums = np.sum(right_img, axis=0)
    left_max_col_index = np.argmax(left_col_sums)
    right_max_col_index = np.argmax(right_col_sums)

    max_col_index = int((left_max_col_index + right_max_col_index + center_col_index) // 2)

    half_size = size // 2
    start_row = max_row_index - half_size
    end_row = max_row_index + half_size
    start_col = max_col_index - half_size
    end_col = max_col_index + half_size

    cropped_img = img[start_row:end_row, start_col:end_col]

    return cropped_img

def crop_cross(img, size, step):
    row_sums = np.sum(img, axis=1)
    col_sums = np.sum(img, axis=0)
    max_row_index = np.argmax(row_sums)
    max_col_index = np.argmax(col_sums)

    half_size = size // 2
    start_row = max(max_row_index - half_size, 0)
    end_row = min(max_row_index + half_size, img.shape[0])
    start_col = max(max_col_index - half_size, 0)
    end_col = min(max_col_index + half_size, img.shape[1])

    cross_img = img[start_row:end_row, start_col:end_col]

    # upper
    for i in range(step):
        start_r = start_row  - (i + 1) * size
        end_r = start_r + size
        start_c = start_col
        end_c = start_c + size
        cropped_img = img[start_r:end_r, start_c:end_c]
        cross_img = np.hstack((cross_img, cropped_img))

    # left
    for i in range(step):
        start_r = start_row
        end_r = start_r + size
        start_c = start_col  - (i + 1) * size
        end_c = start_c + size
        cropped_img = img[start_r:end_r, start_c:end_c]
        cross_img = np.hstack((cross_img, cropped_img))

    # lower
    for i in range(step):
        start_r = start_row + (i + 1) * size
        end_r = start_r + size
        start_c = start_col
        end_c = start_c + size
        cropped_img = img[start_r:end_r, start_c:end_c]
        cross_img = np.hstack((cross_img, cropped_img))

    # right
    for i in range(step):
        start_r = start_row
        end_r = start_r + size
        start_c = start_col + (i + 1) * size
        end_c = start_c + size
        cropped_img = img[start_r:end_r, start_c:end_c]
        cross_img = np.hstack((cross_img, cropped_img))

    return cross_img

def adjust_gamma(image):
    gamma = random.uniform(0.8, 1.2)
    # gamma = 1
    invgamma = 1/gamma
    adjusted_image = np.array(np.power((image/255), invgamma)*255, dtype=np.uint8)
    return adjusted_image

def _worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32) 
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class AADataLoader(DataLoader):
    def __init__(self, data_dir, img_num, batch_size, generator: torch.Generator, 
                 shuffle=True, num_workers=8, training=True):
        
        self.data_dir = data_dir
        self.img_num = img_num
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.dataset = XY_Dataset(data_dir, img_num, batch_size)
        super().__init__(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                         worker_init_fn=_worker_init_fn,
                        generator=generator)
        

        self.info = {
            "list_path_data": data_dir,
            "len(dataset)": len(self.dataset)
        }


    def __str__(self):
        return  "AADataLoader:\n  "+ str(self.info)
    

class Z_Dataset(Dataset):
    def __init__(self, data_dir, img_num, batch_size):

        self.data_dir = data_dir
        self.img_num = img_num
        self.batch_size = batch_size
        direct_subfolders = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        # 读取所有文件夹（直到叶子节点）
        # self.folder_list = [os.path.join(dp, d) for dsf in direct_subfolders for dp, dn, filenames in os.walk(dsf) for d in dn]
        # 仅读取下一级子文件夹
        self.folder_list = [os.path.join(dsf, d) for dsf in direct_subfolders for d in os.listdir(dsf) if os.path.isdir(os.path.join(dsf, d))]


        # 定义数据转换
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.folder_list)

    def __getitem__(self, idx):
        images = []
        # 从文件夹名中提取标签
        folder_name = self.folder_list[idx]
        parts = folder_name.split('/')
        last_part = parts[-1]
        lens_dir = '/'.join(parts[:6])
        Z_label = float(last_part)
        Z_label = torch.tensor([Z_label])

        
        image_list = [os.path.join(folder_name, '0', f) for f in os.listdir(os.path.join(folder_name, '0')) if f.endswith('.png')]
        image_list_f = [os.path.join(folder_name, '-4.000000', f) for f in os.listdir(os.path.join(folder_name, '-4.000000')) if f.endswith('.png')]
        image_list_b = [os.path.join(folder_name, '4.000000', f) for f in os.listdir(os.path.join(folder_name, '4.000000')) if f.endswith('.png')]

        image_list = [image_list[0]] + image_list[-4:]
        image_list_f = [image_list_f[0]] + image_list_f[-4:]
        image_list_b = [image_list_b[0]] + image_list_b[-4:]

        image_list = image_list + image_list_f + image_list_b
        image_list = sorted(image_list, key=lambda x: x)
        # image_list = [image_list[0]] + image_list[-4:]

        # 循环加载图片
        for i in range(self.img_num):
            image_path = os.path.join(folder_name, image_list[i])
            image = Image.open(image_path)
            image = np.array(image)
            # image = adjust_gamma(image)
            image = cv2.medianBlur(image, 3)
            image = Image.fromarray(image)
            image = normalization(image)
            image = crop_offaxis(image, 90)
            # image = crop_cross(image, 16, 5)
            image = self.transform(image)
            image = image.float() 
            images.append(image)

        images = torch.stack(images)
        return images, Z_label, lens_dir

    
class XY_Dataset(Dataset):
    def __init__(self, data_dir, img_num, batch_size):

        self.data_dir = data_dir
        self.img_num = img_num
        self.batch_size = batch_size

        # # 读取所有文件夹（直到叶子节点）
        # leaf_folders = [dirpath for dirpath, dirnames, filenames in os.walk(data_dir) if not dirnames]
        is_train_data = '/home/lwy/Stol-dataset-Source/train' in data_dir

        if is_train_data:
            selected_folders = {60, 17, 8, 24, 50, 1, 26, 15, 45, 43}
            top_level_folders = [
            os.path.join(data_dir, folder) for folder in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, folder)) and int(folder) in selected_folders
            ]
            # 遍历筛选后的文件夹，提取叶子节点
            leaf_folders = [
            dirpath for top_folder in top_level_folders
            for dirpath, dirnames, filenames in os.walk(top_folder)
            if not dirnames
            ]

            filtered_folders = []
            for folder in leaf_folders:
                last_part = folder.split('/')[-1]
                label_strings = last_part.split(' ')  # 根据逗号分割
                labels = [float(label) for label in label_strings[:2]]
                if labels[0] % 1 == 0 and labels[1] % 1 ==0:  # 仅保留 label[0] 能被2整除的文件夹
                    filtered_folders.append(folder)
            leaf_folders = filtered_folders
        else:
            top_level_folders = [
            os.path.join(data_dir, folder) for folder in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, folder))
            ]

            # 遍历筛选后的文件夹，提取叶子节点
            leaf_folders = [
                dirpath for top_folder in top_level_folders
                for dirpath, dirnames, filenames in os.walk(top_folder)
                if not dirnames
            ]
        # 将结果赋值给 self.folder_list
        self.folder_list = leaf_folders

        # 定义数据转换
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.folder_list)

    def __getitem__(self, idx):
        images = []
        # 从文件夹名中提取标签
        folder_name = self.folder_list[idx]
        parts = folder_name.split('/')
        lens_dir = '/'.join(parts[:6])
        last_part = parts[-1] # 取出最后一部分，例如 '-4 -6 0'
        label_strings = last_part.split(' ')  # 根据,分割
        labels = [float(label) for label in label_strings[:2]]
        labels = torch.tensor(labels) / 20 # 转换为 PyTorch 张量

        # simulation setting
        image_list = [f for f in os.listdir(folder_name) if f.endswith('.png')]
        image_list = sorted(image_list, key=lambda x: x)
        image_list = [image_list[0]] + image_list[-4:]  # 取第一张和最后四张


        # 循环加载三张图片
        for i in range(self.img_num):
            # 检查image_list的长度与img_num是否匹配
            if len(image_list) != self.img_num:
                raise ValueError(f"Error: The length of image_list ({len(image_list)}) does not match img_num ({self.img_num}).\n"
                                f"Folder name: {folder_name}\n"
                                f"Image list: {image_list}")
            image = Image.open(os.path.join(folder_name, image_list[i])).convert('L')
            image = self.crop_centroid(image, 50)
            image = self.transform(image)
            image = image.float() 
            images.append(image)

        # 三张图片塞进一个tensor中
        images = torch.stack(images)

        return images, labels, lens_dir

    def crop_centroid(self, image, crop_size):
        """Crop image to specified size centered at the centroid using projection method"""

        img_array = np.array(image)
        # Locate centroid using projection
        row_sums = np.sum(img_array, axis=1)
        y_center = np.argmax(row_sums)
        col_sums = np.sum(img_array, axis=0)
        x_center = np.argmax(col_sums)
        half_size = crop_size // 2
        crop = img_array[y_center - half_size:y_center + half_size, x_center - half_size:x_center + half_size]
        
        if crop.shape != (crop_size, crop_size):
            raise ValueError(f"Cropped image size not {crop_size}x{crop_size}")
        
        return Image.fromarray(crop)
    
class DA_70028_DataLoader(DataLoader):
    def __init__(self, data_dir, img_num, batch_size, generator: torch.Generator,
                 shuffle=True, num_workers=8, training=True, split='train'):

        # 如果 data_dir 是字符串，转换为列表
        if isinstance(data_dir, str):
            data_dir = [data_dir]

        self.data_dir = data_dir
        self.img_num = img_num
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.dataset = DA_70028_Dataset(data_dir, img_num, batch_size, split=split)
        super().__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            worker_init_fn=_worker_init_fn,
            generator=generator
        )
        

        self.info = {
            "list_path_data": data_dir,
            "len(dataset)": len(self.dataset)
        }
    
class DA_70028_Dataset(Dataset):
    def __init__(self, data_dir, img_num, batch_size, split='train'):
        # 如果 data_dir 是字符串，转换为列表
        if isinstance(data_dir, str):
            data_dir = [data_dir]

        self.data_dirs = data_dir
        self.img_num = img_num
        self.batch_size = batch_size
        self.split = split  # 新增：'train' 或 'valid'

        all_leaf_folders = []
        for data_dir in self.data_dirs:

            is_train_data = '/home/lwy/724-70028/XY/train-DA' in data_dir

            # 对于 train-DA 数据，过滤 label[0] 不能被2整除的文件夹
            if is_train_data:
                # 筛选第一级子目录中命名
                top_level_folders = [
                    os.path.join(data_dir, folder) for folder in os.listdir(data_dir)
                    if os.path.isdir(os.path.join(data_dir, folder)) and int(folder) > 19 and int(folder) < 22
                ]
                # 遍历筛选后的文件夹，提取叶子节点
                leaf_folders = [
                    dirpath for top_folder in top_level_folders
                    for dirpath, dirnames, filenames in os.walk(top_folder)
                    if not dirnames
                ]
                filtered_folders = []
                for folder in leaf_folders:
                    last_part = folder.split('/')[-1]
                    label_strings = last_part.split(',')  # 根据逗号分割
                    labels = [float(label) for label in label_strings[:2]]
                    if labels[0] % 2 == 0 and labels[1] % 2 ==0:  # 仅保留 label[0] 能被2整除的文件夹
                        filtered_folders.append(folder)
                leaf_folders = filtered_folders

            elif '/home/lwy/stol2trg' in data_dir:
                # 筛选第一级子目录中命名
                top_level_folders = [
                    os.path.join(data_dir, folder) for folder in os.listdir(data_dir)
                    if os.path.isdir(os.path.join(data_dir, folder))
                ]
                # 遍历筛选后的文件夹，提取叶子节点
                leaf_folders = [
                    dirpath for top_folder in top_level_folders
                    for dirpath, dirnames, filenames in os.walk(top_folder)
                    if not dirnames
                ]
            
            elif '/home/lwy/Stol2trg' in data_dir and '/home/lwy/Stol2trg_Cyclegan_labeled' not in data_dir:

                # 读取所有叶子文件夹
                leaf_folders = [dirpath for dirpath, dirnames, filenames in os.walk(data_dir) if not dirnames]
                # 指定需要筛选的数字集合
                selected_numbers = {76, 19, 12, 49, 11, 51, 36, 25, 18, 41}
                # 筛选叶子文件夹
                filtered_leaf_folders = [
                    folder for folder in leaf_folders
                    if folder.split(os.sep)[-1].split('_')[0].isdigit() and
                    int(folder.split(os.sep)[-1].split('_')[0]) in selected_numbers
                ]
                leaf_folders = filtered_leaf_folders
                
            elif '/home/lwy/70028-Stol-' in data_dir:
                # 根据 split 参数选择不同的顶级文件夹
                if self.split == 'train':
                    # selected_folders = {3, 35, 84, 87, 54, 89, 40, 96, 79, 71}  # 训练集文件夹
                    selected_folders = {76, 19, 12, 49, 11, 51, 36, 25, 18, 41}
                elif self.split == 'valid':
                    selected_folders = {14, 21, 73, 81, 58, 69, 77, 82, 70, 7}  # 验证集文件夹
                elif self.split == 'test':
                    # selected_folders = {14, 21, 73, 81, 58, 69, 77, 82, 70, 7} # largest rms
                    selected_folders = {93, 5, 9, 57, 22, 98, 61, 59, 97, 30} # medium rms
                    # selected_folders = {52, 17, 1, 10, 45, 34, 33, 67, 66, 27} # smallest rms
                else:
                    raise ValueError(f"Invalid split: {self.split}. Must be 'train' or 'valid'.")
                # 筛选第一级子目录中命名
                top_level_folders = [
                    os.path.join(data_dir, folder) for folder in os.listdir(data_dir)
                    # if os.path.isdir(os.path.join(data_dir, folder)) and int(folder) > 0 and int(folder) < 101
                    if os.path.isdir(os.path.join(data_dir, folder)) and int(folder) in selected_folders
                ]
                # 遍历筛选后的文件夹，提取叶子节点
                leaf_folders = [
                    dirpath for top_folder in top_level_folders
                    for dirpath, dirnames, filenames in os.walk(top_folder)
                    if not dirnames
                ]

            else:
                # 读取所有文件夹（直到叶子节点）
                leaf_folders = [dirpath for dirpath, dirnames, filenames in os.walk(data_dir) if not dirnames]

            all_leaf_folders.extend(leaf_folders)

        # # 检查是否为特殊路径
        # is_special = any('/home/lwy/70028-Stol-target-dataset' in d or '/home/lwy/Stol2trg' in d for d in self.data_dirs)

        # if is_special:
        #     self.folder_dict = defaultdict(list)
        #     for folder in all_leaf_folders:
        #         last_part = os.path.basename(folder)
        #         if '_' in last_part:
        #             label_str = last_part.split('_')[1]
        #         else:
        #             label_str = last_part
        #         labels = tuple(map(float, label_str.split()[:2]))
        #         self.folder_dict[labels].append(folder)
        #     self.ecc_positions = list(self.folder_dict.keys())
        #     self.ecc_positions.sort()  # 可选：排序以保持一致性
        # else:
        #     self.folder_list = all_leaf_folders

        # 将结果赋值给 self.folder_list
        self.folder_list = all_leaf_folders

        # 定义数据转换
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
    
    def __len__(self):
        # if hasattr(self, 'ecc_positions'):
        #     return len(self.ecc_positions)
        # else:
        #     return len(self.folder_list)
        
        return len(self.folder_list)
    
    def crop_centroid(self, image, crop_size):
        """Crop image to specified size centered at the centroid using projection method"""

        img_array = np.array(image)
        # Locate centroid using projection
        row_sums = np.sum(img_array, axis=1)
        y_center = np.argmax(row_sums)
        col_sums = np.sum(img_array, axis=0)
        x_center = np.argmax(col_sums)
        half_size = crop_size // 2
        crop = img_array[y_center - half_size:y_center + half_size, x_center - half_size:x_center + half_size]
        
        if crop.shape != (crop_size, crop_size):
            raise ValueError(f"Cropped image size not {crop_size}x{crop_size}")
        
        return Image.fromarray(crop)

    def downsample_image(self, image, target_size=70):
        """Downsample image to target_size x target_size using bilinear interpolation"""
        img_array = np.array(image)
        downsampled = cv2.resize(img_array, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        return Image.fromarray(downsampled)
    
    def mask_center(self, image, mask_size):
        """Mask the center region of the image with zeros"""
        img_array = np.array(image)
        center_row, center_col = img_array.shape[0] // 2, img_array.shape[1] // 2
        half_size = mask_size // 2
        img_array[center_row - half_size:center_row + half_size, center_col - half_size:center_col + half_size] = 0
        return Image.fromarray(img_array)

    def __getitem__(self, idx):
        images = []
        # if hasattr(self, 'ecc_positions'):
        #     ecc_key = self.ecc_positions[idx]
        #     folder_name = random.choice(self.folder_dict[ecc_key])
        # else:
        #     folder_name = self.folder_list[idx]
        # 从文件夹名中提取标签
        folder_name = self.folder_list[idx]
        parts = folder_name.split('/')
        lens_dir = '/'.join(parts[:6])
        last_part = parts[-1] # 取出最后一部分，例如 '-4 -6 0'

        is_real_data = '/home/lwy/724-70028/XY' in folder_name or '/home/lwy/Ttest2SowithoutISP' in folder_name or '/home/lwy/Ttest2SowoISP_TOL_train' in folder_name
        
        if is_real_data:
            label_strings = last_part.split(',')  # 根据,分割

            labels = [float(label) for label in label_strings[:2]]
            
            labels = torch.tensor(labels) / 20 # 转换为 PyTorch 张量
            
            # 70028
            image_list = [f for f in os.listdir(folder_name) if f.endswith('-p.png')]
            # image_list = [f for f in os.listdir(folder_name) if f.endswith('.png') and not f.endswith('-p.png')]  # 删除以-p结尾的图片
            
            image_list = sorted(image_list, key=lambda x: x)
            # image_list = [image_list[0]] + image_list[-4:]  # 取第一张和最后四张

            # 循环加载三张图片
            for i in range(self.img_num):
                # 检查image_list的长度与img_num是否匹配
                if len(image_list) != self.img_num:
                    raise ValueError(f"Error: The length of image_list ({len(image_list)}) does not match img_num ({self.img_num}).\n"
                                    f"Folder name: {folder_name}\n"
                                    f"Image list: {image_list}")
                image = Image.open(os.path.join(folder_name, image_list[i])).convert('L')
                # image = np.array(image)
                # image = adjust_gamma(image)
                # image = cv2.medianBlur(image, 3)
                # image = Image.fromarray(image)
                # image = normalization(image)
                # image = self.crop_centroid(image, 280)
                # image = self.downsample_image(image, 70)
                # image = self.mask_center(image, 30)
                image = self.transform(image)
                image = image.float() 
                images.append(image)

            # 三张图片塞进一个tensor中
            images = torch.stack(images)

        else:
            # 检查是否包含下划线
            if '_' in last_part:
                # 如果有下划线，提取下划线后的部分
                label_strings = last_part.split('_')[1].split(' ')
            else:
                # 如果没有下划线，直接按空格分割
                label_strings = last_part.split(' ')
            labels = [float(label) for label in label_strings[:2]]
            labels = torch.tensor(labels) / 20 # 转换为 PyTorch 张量

            # 70028
            image_list = [f for f in os.listdir(folder_name) if f.endswith('.png')]
            image_list = sorted(image_list, key=lambda x: x)
            # 循环加载三张图片
            for i in range(self.img_num):
                # 检查image_list的长度与img_num是否匹配
                if len(image_list) != self.img_num:
                    raise ValueError(f"Error: The length of image_list ({len(image_list)}) does not match img_num ({self.img_num}).\n"
                                    f"Folder name: {folder_name}\n"
                                    f"Image list: {image_list}")
                image = Image.open(os.path.join(folder_name, image_list[i])).convert('L')
                # image = np.array(image)
                # image = adjust_gamma(image)
                # image = cv2.medianBlur(image, 3)
                # image = Image.fromarray(image)
                # image = normalization(image)
                if (
                    '/home/lwy/src2trg' in folder_name or
                    '/home/lwy/stol2trg' in folder_name or
                    '/home/lwy/So_wISP2So_woISP' in folder_name or
                    '/home/lwy/Stol2SowoISP' in folder_name or
                    '/home/lwy/Stol2trg' in folder_name or
                    '/home/lwy/So2trg' in folder_name
                ):
                    pass
                else:
                    image = self.crop_centroid(image, 70)
                # image = self.mask_center(image, 30)
                image = self.transform(image)
                image = image.float() 
                images.append(image)

            # 三张图片塞进一个tensor中
            images = torch.stack(images)

            # ==================== 随机退化（只对模拟域） ====================
        is_simulated = not (
            '/home/lwy/724-70028/XY' in folder_name or
            '/home/lwy/Ttest2SowithoutISP' in folder_name or
            '/home/lwy/Ttest2SowoISP_TOL_train' in folder_name
        )

        ENABLE_DEGRADE = False
        N_DEGRADES = 1
        FIXED_DEGRADE = None  # 可设为 'jpg' / 'blur' / 'noise' / 'mask' 用于消融

        if ENABLE_DEGRADE and is_simulated:
            x = images.unsqueeze(0)  # -> (1, N, 1, H, W) 适配 batch 函数

            if FIXED_DEGRADE:
                ops = [FIXED_DEGRADE]
            else:
                ops = random.sample(list(DEGRADE_OPS.keys()), k=N_DEGRADES)

            for op_name in ops:
                # 关键：这里直接调用你写好的 lambda，内部自动用全局 random（已由 worker_init_fn 锁定）
                x = DEGRADE_OPS[op_name](x)

            images = torch.clamp(x.squeeze(0), 0, 1)

        return images, labels, lens_dir
    

class DA_70028_Dataset_classification(Dataset):
    def __init__(self, data_dir, img_num, batch_size, num_bins=85):
        # 如果 data_dir 是字符串，转换为列表
        if isinstance(data_dir, str):
            data_dir = [data_dir]
        self.data_dirs = data_dir
        self.img_num = img_num
        self.batch_size = batch_size
        self.num_bins = num_bins  # 新增：每个维度的bin数量

        all_leaf_folders = []
        for data_dir in self.data_dirs:

            is_train_data = '/home/lwy/724-70028/XY/train-DA' in data_dir

            # 对于 train-DA 数据，过滤 label[0] 不能被2整除的文件夹
            if is_train_data:
                # 筛选第一级子目录中命名
                top_level_folders = [
                    os.path.join(data_dir, folder) for folder in os.listdir(data_dir)
                    if os.path.isdir(os.path.join(data_dir, folder)) and int(folder) > 20 and int(folder) < 22
                ]
                # 遍历筛选后的文件夹，提取叶子节点
                leaf_folders = [
                    dirpath for top_folder in top_level_folders
                    for dirpath, dirnames, filenames in os.walk(top_folder)
                    if not dirnames
                ]
                filtered_folders = []
                for folder in leaf_folders:
                    last_part = folder.split('/')[-1]
                    label_strings = last_part.split(',')  # 根据逗号分割
                    labels = [float(label) for label in label_strings[:2]]
                    if labels[0] % 10 == 0 and labels[1] % 10 ==0:  # 仅保留 label[0] 能被2整除的文件夹
                        filtered_folders.append(folder)
                leaf_folders = filtered_folders

            elif '/home/lwy/stol2trg' in data_dir:
                # 筛选第一级子目录中命名
                top_level_folders = [
                    os.path.join(data_dir, folder) for folder in os.listdir(data_dir)
                    if os.path.isdir(os.path.join(data_dir, folder))
                ]
                # 遍历筛选后的文件夹，提取叶子节点
                leaf_folders = [
                    dirpath for top_folder in top_level_folders
                    for dirpath, dirnames, filenames in os.walk(top_folder)
                    if not dirnames
                ]
                
            elif '/home/lwy/70028-Stol-' in data_dir:
                # 筛选第一级子目录中命名
                top_level_folders = [
                    os.path.join(data_dir, folder) for folder in os.listdir(data_dir)
                    # if os.path.isdir(os.path.join(data_dir, folder)) and int(folder) > 0 and int(folder) < 101
                    if os.path.isdir(os.path.join(data_dir, folder)) and int(folder) in {52, 31, 85, 77, 71}
                ]
                # 遍历筛选后的文件夹，提取叶子节点
                leaf_folders = [
                    dirpath for top_folder in top_level_folders
                    for dirpath, dirnames, filenames in os.walk(top_folder)
                    if not dirnames
                ]

            else:
                # 读取所有文件夹（直到叶子节点）
                leaf_folders = [dirpath for dirpath, dirnames, filenames in os.walk(data_dir) if not dirnames]

            all_leaf_folders.extend(leaf_folders)

        # 将结果赋值给 self.folder_list
        self.folder_list = all_leaf_folders

        # 定义数据转换
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        # 定义离散化网格
        self.bin_edges = np.linspace(-1, 1, num_bins + 1)  # 假设归一化范围[-1, 1]

    def __len__(self):
        return len(self.folder_list)
    
    def crop_centroid(self, image, crop_size):
        """Crop image to specified size centered at the centroid using projection method"""

        img_array = np.array(image)
        # Locate centroid using projection
        row_sums = np.sum(img_array, axis=1)
        y_center = np.argmax(row_sums)
        col_sums = np.sum(img_array, axis=0)
        x_center = np.argmax(col_sums)
        half_size = crop_size // 2
        crop = img_array[y_center - half_size:y_center + half_size, x_center - half_size:x_center + half_size]
        
        if crop.shape != (crop_size, crop_size):
            raise ValueError(f"Cropped image size not {crop_size}x{crop_size}")
        
        return Image.fromarray(crop)

    def _discretize_labels(self, labels):
        """将连续标签(x, y)离散化为类别索引"""
        x, y = labels[0], labels[1]
        x = x / 42
        y = y / 42
        x_bin = np.digitize(x, self.bin_edges) - 1  # 0到num_bins-1
        y_bin = np.digitize(y, self.bin_edges) - 1
        # 确保索引在有效范围内
        x_bin = np.clip(x_bin, 0, self.num_bins - 1)
        y_bin = np.clip(y_bin, 0, self.num_bins - 1)
        # 转换为单一类别索引：x_bin * num_bins + y_bin
        class_idx = x_bin * self.num_bins + y_bin
        return torch.tensor(class_idx, dtype=torch.long)

    def __getitem__(self, idx):
        images = []
        folder_name = self.folder_list[idx]
        parts = folder_name.split('/')
        lens_dir = '/'.join(parts[:6])
        last_part = parts[-1]

        is_real_data = '/home/lwy/724-70028/XY' in folder_name or '/home/lwy/Ttest2SowithoutISP' in folder_name or '/home/lwy/Ttest2SowoISP_TOL_train' in folder_name
        
        if is_real_data:
            label_strings = last_part.split(',')
            labels = [float(label) for label in label_strings[:2]]
            labels = torch.tensor(labels)
            class_label = self._discretize_labels(labels)  # 离散化为类别索引
            
            image_list = [f for f in os.listdir(folder_name) if f.endswith('-p.png')]
            image_list = sorted(image_list, key=lambda x: x)

            for i in range(self.img_num):
                if len(image_list) != self.img_num:
                    raise ValueError(f"Error: The length of image_list ({len(image_list)}) does not match img_num ({self.img_num}).\n"
                                    f"Folder name: {folder_name}\n"
                                    f"Image list: {image_list}")
                image = Image.open(os.path.join(folder_name, image_list[i])).convert('L')
                image = self.transform(image)
                image = image.float()
                images.append(image)

            images = torch.stack(images)
        else:
            if '_' in last_part:
                label_strings = last_part.split('_')[1].split(' ')
            else:
                label_strings = last_part.split(' ')
            labels = [float(label) for label in label_strings[:2]]
            labels = torch.tensor(labels)
            class_label = self._discretize_labels(labels)  # 离散化为类别索引
            
            image_list = [f for f in os.listdir(folder_name) if f.endswith('.png')]
            image_list = sorted(image_list, key=lambda x: x)

            for i in range(self.img_num):
                if len(image_list) != self.img_num:
                    raise ValueError(f"Error: The length of image_list ({len(image_list)}) does not match img_num ({self.img_num}).\n"
                                    f"Folder name: {folder_name}\n"
                                    f"Image list: {image_list}")
                image = Image.open(os.path.join(folder_name, image_list[i])).convert('L')
                if '/home/lwy/src2trg' in folder_name or '/home/lwy/stol2trg' in folder_name or '/home/lwy/So_wISP2So_woISP' in folder_name or '/home/lwy/Stol2SowoISP' in folder_name:
                    pass
                else:
                    image = self.crop_centroid(image, 70)
                image = self.transform(image)
                image = image.float()
                images.append(image)

            images = torch.stack(images)

        return images, class_label, lens_dir
    
class DA_70028_DataLoader_classification(DataLoader):
    def __init__(self, data_dir, img_num, batch_size, shuffle=True, num_workers=8, training=True, num_bins=10):
        if isinstance(data_dir, str):
            data_dir = [data_dir]
        self.data_dir = data_dir
        self.img_num = img_num
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.dataset = DA_70028_Dataset_classification(data_dir, img_num, batch_size, num_bins=num_bins)
        super().__init__(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        
        self.info = {
            "list_path_data": data_dir,
            "len(dataset)": len(self.dataset)
        }

if __name__ == "__main__":

    train_dataset = DA_70028_DataLoader(data_dir="/home/lwy/724-70028/XY/train", img_num=5, batch_size=64, shuffle=True, num_workers=8, training=True).dataset
    # print(len(train_dataset))
    # for i in range(len(train_dataset)):
    #     tensor, label, _ = train_dataset.__getitem__(i)
    tensor1 = train_dataset[5][0]
    label2 = train_dataset[5][1]
    print(label2)
    # tensor3 = train_dataset[500][0]
    fig, axs = plt.subplots(1, 5, figsize=(15, 3))

    for i in range(5):
        image = to_image(tensor1[i].squeeze(0))
        axs[i].imshow(image, cmap='gray')
        axs[i].axis('off')

    plt.savefig('images1.png')
    # print(label)

    # for batch_idx, (data, target) in enumerate((train_dataset)):
    #     pass