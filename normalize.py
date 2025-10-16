from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 图像文件的路径
image_paths = [
    'data_0507/Z/valid/10/0 0 22/_camera_1.bmp',
    'data_0507/Z/valid/10/0 0 22/_camera_14.bmp',
    'data_0507/Z/valid/10/0 0 22/_camera_15.bmp',
    'data_0507/Z/valid/10/0 0 22/_camera_16.bmp',
    'data_0507/Z/valid/10/0 0 22/_camera_17.bmp',
]

# 读取并标准化图像
normalized_images = []
for image_path in image_paths:
    # 读取图像
    image = np.array(Image.open(image_path))

    # 计算均值和方差
    mean = np.mean(image)
    std = np.std(image)

    # 标准化图像
    normalized_image = (image - mean) / std

    normalized_images.append(normalized_image)

# 将标准化后的图像保存到一张图里
fig, axs = plt.subplots(1, len(normalized_images), figsize=(20, 4))
for i, normalized_image in enumerate(normalized_images):
    axs[i].imshow(normalized_image, cmap='gray')
    axs[i].axis('off')
plt.show()