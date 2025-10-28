import torch.nn.functional as F
import torch


def nll_loss(output, target):
    return F.nll_loss(output, target)

def mse_loss_means(output, target):
    criterion = torch.nn.MSELoss(reduction='mean')
    loss =  criterion(output, target)
    return loss

def cross_entropy_loss(output, target):
    criterion = torch.nn.CrossEntropyLoss()
    return criterion(output, target)