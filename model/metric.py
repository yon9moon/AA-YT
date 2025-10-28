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

#（Mean Absolute Error，MAE）
def mae_xs(output, target):
    with torch.no_grad():
        MAE_x = torch.mean(torch.abs(output[:,0]-target[:,0])).item()
    return MAE_x

def mae_ys(output, target):
    with torch.no_grad():
        MAE_y = torch.mean(torch.abs(output[:,1]-target[:,1])).item()
    return MAE_y

def mae_xt(output, target):
    with torch.no_grad():
        MAE_x = torch.mean(torch.abs(output[:,2]-target[:,2])).item()
    return MAE_x

def mae_yt(output, target):
    with torch.no_grad():
        MAE_y = torch.mean(torch.abs(output[:,3]-target[:,3])).item()
    return MAE_y

def mae_x(output, target, num_bins=85):
    with torch.no_grad():
        bin_edges = torch.linspace(-1, 1, num_bins + 1, device=output.device)
        pred = torch.argmax(output, dim=1)
        x_bins = pred // num_bins
        x_values = (bin_edges[x_bins] + bin_edges[x_bins + 1]) / 2 * 42
        true_x_bins = target // num_bins
        true_x_values = (bin_edges[true_x_bins] + bin_edges[true_x_bins + 1]) / 2 * 42
        mae = torch.mean(torch.abs(x_values - true_x_values))
    return mae.item()

def mae_y(output, target, num_bins=85):
    with torch.no_grad():
        bin_edges = torch.linspace(-1, 1, num_bins + 1, device=output.device)
        pred = torch.argmax(output, dim=1)
        y_bins = pred % num_bins
        y_values = (bin_edges[y_bins] + bin_edges[y_bins + 1]) / 2 * 42
        true_y_bins = target % num_bins
        true_y_values = (bin_edges[true_y_bins] + bin_edges[true_y_bins + 1]) / 2 * 42
        mae = torch.mean(torch.abs(y_values - true_y_values))
    return mae.item()