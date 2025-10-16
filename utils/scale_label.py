import os

def rename_leaf_folders(base_path):
    # 遍历 base_path 下的所有文件和文件夹
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        
        # 确保只处理文件夹
        if not os.path.isdir(folder_path):
            continue
        
        # 检查是否为叶子节点文件夹（没有子文件夹）
        if all(not os.path.isdir(os.path.join(folder_path, sub)) for sub in os.listdir(folder_path)):
            try:
                # 分割文件夹名称为 `_` 前后两部分
                if '_' in folder_name:
                    prefix, suffix = folder_name.split('_', 1)
                    
                    # 处理 `_` 后的部分
                    new_suffix_parts = []
                    for part in suffix.split():
                        try:
                            # 尝试将部分转换为浮点数
                            num = float(part)
                            # 仅处理小数部分（非整数）
                            if not num.is_integer():
                                new_suffix_parts.append(str(int(num * 1000)))
                            else:
                                # 如果是整数，保留原样
                                new_suffix_parts.append(part)
                        except ValueError:
                            # 如果无法转换为数字，则保留原样
                            new_suffix_parts.append(part)
                    
                    # 拼接新的文件夹名称
                    new_folder_name = f"{prefix}_{' '.join(new_suffix_parts)}"
                else:
                    # 如果没有 `_`，保持文件夹名称不变
                    new_folder_name = folder_name
                
                new_folder_path = os.path.join(base_path, new_folder_name)
                
                # 重命名文件夹
                os.rename(folder_path, new_folder_path)
                print(f"Renamed: {folder_path} -> {new_folder_path}")
            except Exception as e:
                print(f"Error processing folder {folder_name}: {e}")
        else:
            # 如果不是叶子节点文件夹，递归处理子文件夹
            rename_leaf_folders(folder_path)


# 设置目标路径
base_path = "/home/lwy/Stol2trg"
rename_leaf_folders(base_path)