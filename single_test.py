import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import cv2
import sys
sys.path.append('/mnt1/lwy/AA-YT')
from model.model import AAmodel
import time


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


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

    for i in range(step):
        start_r = start_row + (i + 1) * size
        end_r = start_r + size
        start_c = start_col
        end_c = start_c + size
        cropped_img = img[start_r:end_r, start_c:end_c]
        cross_img = np.hstack((cross_img, cropped_img))

    for i in range(step):
        start_r = start_row
        end_r = start_r + size
        start_c = start_col + (i + 1) * size
        end_c = start_c + size
        cropped_img = img[start_r:end_r, start_c:end_c]
        cross_img = np.hstack((cross_img, cropped_img))

    return cross_img


class ImageLoader:
    def __init__(self, image_path, img_num):
        self.image_path = image_path
        self.img_num = img_num
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def load_image(self):
        images = []
        # 从文件夹名中提取标签
        folder_name = self.image_path
        parts = folder_name.split('/')
        last_part = parts[-1] # 取出最后一部分，例如 '-4 -6 0'
        label_strings = last_part.split(',')  # 再次分割，得到 ['-4', '-6', '0']
        labels = [float(label) for label in label_strings[:2]]
        labels = torch.tensor(labels)  # 转换为 PyTorch 张量

        image_list = [f for f in os.listdir(folder_name) if f.endswith('.png')]
        image_list = sorted(image_list, key=lambda x: x)
        image_list = [image_list[0]] + image_list[-4:]



        # 循环加载三张图片
        for i in range(self.img_num):
            image = Image.open(os.path.join(folder_name, image_list[i]))
            image = np.array(image)
            image = cv2.medianBlur(image, 3)
            image = Image.fromarray(image)
            image = normalization(image)
            # if i == 0:
            #     image = crop_center(image, 150)
            # else:
            #     image = crop_offaxis(image, 150)
            image = crop_offaxis(image, 90)
            image = self.transform(image)
            image = image.float() 
            images.append(image)


        # 三张图片塞进一个tensor中
        images = torch.stack(images)

        return images, labels
    
class Z_ImageLoader:
    def __init__(self, image_path, img_num):
        self.image_path = image_path
        self.img_num = img_num
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def load_image(self):
        images = []
        # 从文件夹名中提取标签
        folder_name = self.image_path
        parts = folder_name.split('/')
        last_part = parts[-1]
        Z_label = float(last_part)
        Z_label = torch.tensor([Z_label])

        
        image_list = [os.path.join(folder_name, '0', f) for f in os.listdir(os.path.join(folder_name, '0')) if f.endswith('.png')]
        image_list_f = [os.path.join(folder_name, '-4.000000', f) for f in os.listdir(os.path.join(folder_name, '-4.000000')) if f.endswith('.png')]
        image_list_b = [os.path.join(folder_name, '4.000000', f) for f in os.listdir(os.path.join(folder_name, '4.000000')) if f.endswith('.png')]
        image_list = image_list + image_list_f + image_list_b
        image_list = sorted(image_list, key=lambda x: x)



        # 循环加载三张图片
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

        return images, Z_label
    
# 加载训练好的模型

# XY-10um-5imgs model
# model_path = '/mnt1/lwy/AA-YT/saved/models/AA/0724_200546/model_best.pth'



# Z model
model_path = '/mnt1/lwy/AA-YT/saved/models/AA/0724_192739/model_best.pth'
# model_path = 'saved/models/AA/0613_142808/model_best.pth'
# model_path = 'saved/models/AA/0628_102409/model_best.pth'


checkpoint = torch.load(model_path)
model = AAmodel(img_num=27, in_channels=1, num_classes=1)
model.load_state_dict(checkpoint['state_dict'])
model.eval()  # 设置模型为评估模式


test_dataset = Z_ImageLoader(image_path='/home/lwy/724-70028/Z/valid/16/8', img_num=27)
imgs, label = test_dataset.load_image()
imgs = imgs.unsqueeze(0)


start_time = time.time()
# 进行预测
with torch.no_grad():
    output = model(imgs)
    output = output.squeeze()
end_time = time.time()

print(output)
print(f"Execution time: {end_time - start_time} seconds")

def plot_test(test_dataset, model):
    imgs, label = test_dataset.load_image()
    imgs = imgs.unsqueeze(0)

    # 进行预测
    with torch.no_grad():
        output = model(imgs)
        output = output.squeeze()

    err = torch.sqrt(((output - label) ** 2).mean()).float()
    err = err.item()
    label = label.numpy().tolist()
    label.append(err)
    return label

# data_dir = 'data_0507/XYC/valid/5'
# folder_list = os.listdir(data_dir)
# data_list = []
# for i in range(len(folder_list)):
#     folder_name = os.path.join(data_dir, folder_list[i])
#     test_dataset = ImageLoader(image_path=folder_name, img_num=5)
#     data_list.append(plot_test(test_dataset, model))

# data = np.array(data_list)
# savemat("xy_err_5_crop.mat", {"xy_err_5_crop": data})