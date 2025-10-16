import os
from PIL import Image

def check_png_files(directory):
    # 只遍历最底层文件夹
    for subdir, _, files in os.walk(directory):
        if not os.listdir(subdir):  # 如果子目录为空，跳过
            continue
        png_files = [f for f in files if f.endswith('.png')]
        print(f"目录{subdir}中找到{len(png_files)}个PNG文件。")

        # 检查每张PNG图片的尺寸
        for png_file in png_files:
            file_path = os.path.join(subdir, png_file)
            try:
                with Image.open(file_path) as img:
                    if img.size != (70, 70):
                        print(f"错误: 文件{file_path}的尺寸为{img.size}，而非70x70。")
            except Exception as e:
                print(f"错误: 无法打开文件{file_path}，错误信息: {e}")

if __name__ == "__main__":
    directory = "/home/lwy/Stol2trg"
    check_png_files(directory)