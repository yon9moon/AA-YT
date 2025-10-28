import os
import shutil

def copy_images_and_log(source_dir, target_dir, log_file):
    # 确保目标文件夹存在
    os.makedirs(target_dir, exist_ok=True)

    # 初始化计数器和日志列表
    file_counter = 0
    mapping = []

    # 遍历源目录的所有子目录
    for root, dirs, files in os.walk(source_dir):
        # 检查是否是叶子文件夹（没有子文件夹）
        if not dirs:
            # 筛选出后缀为 "-p.png" 的文件
            images = [f for f in files if f.endswith("-p.png")]
            images.sort()  # 按文件名排序

            # 只处理最多 5 张图像
            for img in images[:5]:
                src_path = os.path.join(root, img)
                dst_name = f"{file_counter:06d}.png"  # 生成目标文件名
                dst_path = os.path.join(target_dir, dst_name)

                # 复制文件到目标文件夹
                shutil.copy(src_path, dst_path)

                # 记录映射关系
                mapping.append(f"{src_path} -> {dst_path}")
                file_counter += 1

    # 将映射关系写入日志文件
    with open(log_file, "w") as log:
        log.write("\n".join(mapping))

    print(f"完成！共复制 {file_counter} 张图像，映射关系已保存到 {log_file}")

# 配置源目录、目标目录和日志文件路径
source_directory = "/home/lwy/724-70028/XY/test-DA"
target_directory = "/home/lwy/70028Ttest_dataset"
log_file_path = "/home/lwy/70028Ttest_dataset/mapping.txt"

# 执行脚本
copy_images_and_log(source_directory, target_directory, log_file_path)