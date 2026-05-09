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


class ImageLoader:
    def __init__(self, image_path, img_num):
        self.image_path = image_path
        self.img_num = img_num
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def load_image(self):
        images = []
        folder_path = self.image_path
        
        # 1. 获取文件夹名作为 Label (例如 "-0.002 0.000 0.000 0.000")
        folder_name = os.path.basename(folder_path)
        
        # 2. 获取上一级文件夹名作为 Lens ID (例如 "AYZZO...")
        # 这样写不再依赖绝对路径的层级深度 (parts[:6])
        lens_dir = os.path.basename(os.path.dirname(folder_path))

        # --- 修改点 3: 图片加载逻辑 ---
        # 确保只读取 png 并排序
        all_files = os.listdir(folder_path)
        image_files = [f for f in all_files if f.endswith('.png')]
        image_files = sorted(image_files) # 排序很重要，保证顺序是 02, 03, 08...

        # 检查图片数量
        # 新数据集只有5张图，如果你设置 img_num=5 没问题。
        # 如果 img_num 还是 3，需要确认取哪几张。这里默认取前 img_num 张。
        current_len = len(image_files)
        if current_len < self.img_num:
            # 如果图片不够，通常是生成脚本出了问题，或者读取了错误的文件夹
            raise ValueError(f"Error: Not enough images ({current_len} < {self.img_num}) in {folder_path}")

        # 循环加载图片
        for i in range(self.img_num):
            img_path = os.path.join(folder_path, image_files[i])
            image = Image.open(img_path).convert('L') # 转灰度
            image = self.transform(image)
            image = image.float() 
            images.append(image)

        # 堆叠图片 (Channel 维度堆叠: C=img_num, H, W)
        images = torch.stack(images)

        return images

    
# 加载训练好的模型
model_path = 'saved/models/0407_Ttrain-Ttest_new_xy+uv_mse_data0407/0407_203954/model_best.pth'


checkpoint = torch.load(model_path)
model = AAmodel(img_num=5, in_channels=1, num_classes=4)
model.load_state_dict(checkpoint['state_dict'])
model.eval()  # 设置模型为评估模式


test_dataset = ImageLoader(image_path='/root/Dataset_Splitted_0407/train/AAScan/0403-7/-0.006 0.010 -0.016 -0.003 -0.009 0.000 1', img_num=5)
imgs = test_dataset.load_image()
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