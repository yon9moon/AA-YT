import os

def process_leaf_folders(base_path):
    # 遍历 base_path 下的所有文件和文件夹
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        
        # 确保只处理文件夹
        if not os.path.isdir(folder_path):
            continue
        
        # 检查是否为叶子节点文件夹（没有子文件夹）
        if all(not os.path.isdir(os.path.join(folder_path, sub)) for sub in os.listdir(folder_path)):
            try:
                # 分割文件夹名称为最后一部分
                parts = folder_name.split()
                new_parts = []
                
                for part in parts:
                    try:
                        # 尝试将部分转换为浮点数
                        num = float(part)
                        # 乘以1000并转换为整数
                        new_parts.append(str(int(num * 1000)))
                    except ValueError:
                        # 如果无法转换为数字，则保留原样
                        new_parts.append(part)
                
                # 拼接新的文件夹名称
                new_folder_name = ' '.join(new_parts)
                new_folder_path = os.path.join(base_path, new_folder_name)
                
                # 重命名文件夹
                os.rename(folder_path, new_folder_path)
                print(f"Renamed: {folder_path} -> {new_folder_path}")
            except Exception as e:
                print(f"Error processing folder {folder_name}: {e}")
        else:
            # 如果不是叶子节点文件夹，递归处理子文件夹
            process_leaf_folders(folder_path)


# 设置目标路径
base_path = r'/home/lwy/So2trg_Cyclegan_labeled'
process_leaf_folders(base_path)