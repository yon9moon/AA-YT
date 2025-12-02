import torch
import numpy as np


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

def mae_xs(output, target):
    """计算 x 坐标的 MAE"""
    return torch.mean(torch.abs(output[:, 0] - target[:, 0])) * 20


def mae_ys(output, target):
    """计算 y 坐标的 MAE"""
    return torch.mean(torch.abs(output[:, 1] - target[:, 1])) * 20

def mae_xt(output, target):
    with torch.no_grad():
        MAE_x = torch.mean(torch.abs(output[:,2]-target[:,2])).item()
    return MAE_x

def mae_yt(output, target):
    with torch.no_grad():
        MAE_y = torch.mean(torch.abs(output[:,3]-target[:,3])).item()
    return MAE_y

# ---------- 新增：用来在 validation 时一次性计算所有统计指标（MAE ± Std） ----------
def mae_with_std(output, target, scale=20.0):
    """
    在 validation 阶段一次性计算：
    MAE_x ± Std_x, MAE_y ± Std_y, RMSE_x, RMSE_y
    返回一个 dict，方便 Trainer 直接 log
    """
    with torch.no_grad():
        pred = output.cpu().numpy() * scale
        gt   = target.cpu().numpy() * scale

        ae_x = np.abs(pred[:, 0] - gt[:, 0])
        ae_y = np.abs(pred[:, 1] - gt[:, 1])

        result = {
            'MAE_x':     ae_x.mean(),
            'MAE_x_std': ae_x.std(),
            'MAE_y':     ae_y.mean(),
            'MAE_y_std': ae_y.std(),
        }
        # 可选：整体 MAE（两个通道平均）
        result['MAE_avg'] = (result['MAE_x'] + result['MAE_y']) / 2
        result['MAE_avg_std'] = np.std(np.abs(pred - gt))  # 所有误差的 std

        return result