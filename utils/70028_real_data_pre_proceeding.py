import os
import numpy as np
from PIL import Image
import cv2

def normalization(image):
    """Normalize image to range [0, 1] using min-max normalization"""
    img_array = np.array(image, dtype=np.float32)
    min_val = np.min(img_array)
    max_val = np.max(img_array)
    img_array = (img_array - min_val) / (max_val - min_val)
    return img_array  # 返回浮点数数组

def crop_centroid(img_array, crop_size):
    """Crop image to specified size centered at the centroid using projection method"""
    # Locate centroid using projection
    row_sums = np.sum(img_array, axis=1)
    y_center = np.argmax(row_sums)
    col_sums = np.sum(img_array, axis=0)
    x_center = np.argmax(col_sums)
    half_size = crop_size // 2
    crop = img_array[y_center - half_size:y_center + half_size, x_center - half_size:x_center + half_size]
    
    if crop.shape != (crop_size, crop_size):
        raise ValueError(f"Cropped image size not {crop_size}x{crop_size}")
    
    return crop  # 返回浮点数数组

def downsample_image(img_array, target_size=70):
    """Downsample image to target_size x target_size using bilinear interpolation"""
    downsampled = cv2.resize(img_array, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return downsampled  # 返回浮点数数组

def preprocess_images(base_dirs):
    """Preprocess images in the specified directories"""
    for base_dir in base_dirs:
        # 遍历叶子文件夹
        for root, dirs, files in os.walk(base_dir):
            if not dirs:  # 只处理叶子文件夹
                for i in [0, 5, 6, 7, 8]:  # 只处理指定的图片
                    image_path = os.path.join(root, f"{i}.png")
                    if os.path.exists(image_path):
                        try:
                            # 打开图片
                            image = Image.open(image_path).convert('L')  # 转为灰度图
                            img_array = np.array(image)
                            
                            # 预处理
                            # img_array = normalization(img_array)
                            img_array = crop_centroid(img_array, 280)
                            img_array = downsample_image(img_array, 70)
                            
                            # 保存新图片
                            new_image_path = os.path.join(root, f"{i}-p.png")
                            Image.fromarray(img_array).save(new_image_path)  # 转为uint8保存
                            print(f"Processed and saved: {new_image_path}")
                        except Exception as e:
                            print(f"Error processing {image_path}: {e}")

if __name__ == "__main__":
    # 指定需要处理的两个文件夹
    base_dirs = [
        "/home/lwy/724-70028/XY/test-DA",
        "/home/lwy/724-70028/XY/train-DA"
    ]
    preprocess_images(base_dirs)