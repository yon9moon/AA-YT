import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import sys
import time
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from tqdm import tqdm

# === 环境配置 ===
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
sys.path.append('/mnt1/lwy/AA-YT')  # 你的工程路径

# 尝试导入你的模型
try:
    from model.model import AAmodel
except ImportError:
    print("⚠️ 警告: 无法导入 AAmodel，请检查 sys.path 或路径。")
    class AAmodel(nn.Module):
        def __init__(self, **kwargs): super().__init__()
        def forward(self, x): return torch.zeros(x.shape[0], 4)

# ==========================================
# 1. ImageLoader
# ==========================================
class ImageLoader:
    def __init__(self, img_num=5):
        self.img_num = img_num
        self.transform = transforms.Compose([transforms.ToTensor()])

    def load_stack(self, folder_path):
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Path not found: {folder_path}")
        
        all_files = os.listdir(folder_path)
        image_files = sorted([f for f in all_files if f.endswith('.png') and len(f) == 6])
        
        current_len = len(image_files)
        if current_len < self.img_num:
            if current_len == 0: raise ValueError(f"No images in {folder_path}")
            image_files = (image_files * (self.img_num // current_len + 1))[:self.img_num]
        
        target_files = image_files[:self.img_num]
        images = []
        for f in target_files:
            img_path = os.path.join(folder_path, f)
            image = Image.open(img_path).convert('L')
            image = self.transform(image).float()
            images.append(image)

        return torch.stack(images).unsqueeze(0)

# ==========================================
# 2. ValidationIndexer
# ==========================================
class ValidationIndexer:
    def __init__(self, valid_root):
        self.valid_root = valid_root
        self.coords = []
        self.folder_paths = []
        
        print(f"🔍 正在建立索引: {valid_root}")
        for root, dirs, files in os.walk(valid_root):
            if any(f.endswith('.png') for f in files):
                folder_name = os.path.basename(root)
                try:
                    parts = folder_name.split()
                    if len(parts) >= 4:
                        val = [float(p) for p in parts[:4]]
                        self.coords.append(val)
                        self.folder_paths.append(root)
                except:
                    continue
        
        self.coords = np.array(self.coords)
        if len(self.coords) == 0:
            raise ValueError("未找到有效数据，请检查路径结构。")
            
        print(f"✅ 索引建立完成，共 {len(self.coords)} 个样本。")
        self.tree = KDTree(self.coords)

    def find_nearest_sample(self, target_pos):
        dist, idx = self.tree.query(target_pos)
        return self.coords[idx], self.folder_paths[idx]

# ==========================================
# 3. 仿真逻辑
# ==========================================
def simulate_3steps(indexer, model, device, loader, steps=1):
    start_positions = []
    final_positions = []
    
    model.eval()
    
    with torch.no_grad():
        for i in tqdm(range(len(indexer.coords)), desc="Simulating"):
            current_pos = indexer.coords[i]
            current_path = indexer.folder_paths[i]
            start_pos_record = current_pos.copy()
            
            for step in range(steps):
                try:
                    imgs = loader.load_stack(current_path).to(device)
                except:
                    break
                
                # 预测
                pred = model(imgs).squeeze().cpu().numpy() * 0.1
                
                # 更新位置
                next_pos_theoretical = current_pos - pred
                
                # 寻找最近邻
                current_pos, current_path = indexer.find_nearest_sample(next_pos_theoretical)
            
            start_positions.append(start_pos_record)
            final_positions.append(current_pos)

    return np.array(start_positions), np.array(final_positions)

# ==========================================
# 4. 改进后的绘图逻辑
# ==========================================
def plot_results_enhanced(starts, finals, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 辅助函数：绘制单组维度的图表
    def plot_dimension_set(dim1_idx, dim2_idx, label1, label2, name):
        s1, s2 = starts[:, dim1_idx], starts[:, dim2_idx]
        f1, f2 = finals[:, dim1_idx], finals[:, dim2_idx]
        deltas1, deltas2 = f1 - s1, f2 - s2
        
        # 计算误差，用于颜色映射
        final_errors = np.sqrt(f1**2 + f2**2)
        
        # --- 图表 1: 全局轨迹图 (Global Trajectory) ---
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 1. 起点 (淡灰色)
        ax.scatter(s1, s2, c='gray', s=5, alpha=0.3, label='Start', zorder=1)
        
        # 2. 箭头 (非常淡，避免遮挡)
        ax.quiver(s1, s2, deltas1, deltas2, angles='xy', scale_units='xy', scale=1, 
                  color='lightblue', alpha=0.2, width=0.002, zorder=2)
        
        # 3. 终点 (根据误差着色，层级最高)
        # 误差越小越绿，误差越大越红
        scatter = ax.scatter(f1, f2, c=final_errors, cmap='RdYlGn_r', s=15, alpha=0.8, label='Final', zorder=10)
        
        ax.set_title(f"Global Trajectory ({label1}-{label2})")
        ax.set_xlabel(label1)
        ax.set_ylabel(label2)
        ax.axis('equal')
        plt.colorbar(scatter, label='Distance to Origin')
        ax.grid(True, linestyle='--', alpha=0.2)
        
        path_global = os.path.join(output_dir, f"1_trajectory_{name}_global.png")
        plt.savefig(path_global, dpi=300)
        plt.close()

        # --- 图表 2: 终点局部放大散点图 (Zoomed-in Final Scatter) ---
        # 这里的目的是只看最终结果，不看起点，自动缩放到中心
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 画出原点
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
        
        # 绘制终点
        scatter = ax.scatter(f1, f2, c=final_errors, cmap='viridis', s=20, alpha=0.6)
        
        # 自动限制范围：取 98% 的数据范围，排除极个别离群点导致视野过大
        max_val = np.percentile(np.abs(np.concatenate([f1, f2])), 98)
        limit = max(max_val * 1.2, 0.005) # 至少保留 0.005 的视野
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        
        # 添加同心圆 (例如 0.002, 0.005, 0.01)
        circles = [0.002, 0.005, 0.01]
        for r in circles:
            if r < limit:
                circle = plt.Circle((0, 0), r, color='red', fill=False, linestyle='--', alpha=0.3)
                ax.add_artist(circle)
                ax.text(r, 0, f'{r}', color='red', fontsize=8)

        mae1 = np.mean(np.abs(f1))
        mae2 = np.mean(np.abs(f2))
        
        ax.set_title(f"Final Positions Zoomed-in ({label1}-{label2})\nMAE: {mae1:.5f}, {mae2:.5f}")
        ax.set_xlabel(label1)
        ax.set_ylabel(label2)
        ax.grid(True, alpha=0.3)
        
        path_zoom = os.path.join(output_dir, f"2_final_scatter_{name}_zoom.png")
        plt.savefig(path_zoom, dpi=300)
        plt.close()

        # --- 图表 3: 误差分布直方图 (Histogram) ---
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Dim 1 直方图
        axes[0].hist(f1, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0].set_title(f"Final Error Distribution: {label1}")
        axes[0].set_xlabel("Error Value")
        axes[0].axvline(0, color='red', linestyle='--')
        
        # Dim 2 直方图
        axes[1].hist(f2, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
        axes[1].set_title(f"Final Error Distribution: {label2}")
        axes[1].set_xlabel("Error Value")
        axes[1].axvline(0, color='red', linestyle='--')
        
        path_hist = os.path.join(output_dir, f"3_histogram_{name}.png")
        plt.savefig(path_hist, dpi=300)
        plt.close()
        
        print(f"✅ Generated 3 plots for {name}: Global, Zoomed-in, Histogram")

    # 执行绘图
    plot_dimension_set(0, 1, "X", "Y", "xy")
    plot_dimension_set(2, 3, "U", "V", "uv")

# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    # --- 配置参数 ---
    # 替换成你的真实路径
    VALID_DIR = '/root/Dataset_Splitted_0407/valid'
    MODEL_PATH = 'saved/models/0407_Ttrain-Ttest_new_xy+uv_mse_data0407/0407_203954/model_best.pth'
    OUTPUT_DIR = 'analysis_results/3step_simulation_enhanced_mse_0407'
    IMG_NUM = 5
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 建立索引
    indexer = ValidationIndexer(VALID_DIR)
    
    # 2. 初始化加载器
    loader = ImageLoader(img_num=IMG_NUM)
    
    # 3. 加载模型
    print(f"Loading model from: {MODEL_PATH}")
    model = AAmodel(img_num=IMG_NUM, in_channels=1, num_classes=4)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
    model.to(device)
    
    # 4. 开始仿真
    print("🚀 开始 3 步闭环调整仿真...")
    starts, finals = simulate_3steps(indexer, model, device, loader, steps=3)
    
    # 5. 绘图
    print("\n📊 正在生成增强版图表...")
    plot_results_enhanced(starts, finals, OUTPUT_DIR)
    
    # 打印全局 MAE
    final_mae = np.mean(np.abs(finals), axis=0)
    print("\n========= Final Results =========")
    print(f"MAE X: {final_mae[0]:.6f}")
    print(f"MAE Y: {final_mae[1]:.6f}")
    print(f"MAE U: {final_mae[2]:.6f}")
    print(f"MAE V: {final_mae[3]:.6f}")