import torch


# def accuracy(output, target):
#     with torch.no_grad():
#         pred = torch.argmax(output, dim=1)
#         assert pred.shape[0] == len(target)
#         correct = 0
#         correct += torch.sum(pred == target).item()
#     return correct / len(target)


# def top_k_acc(output, target, k=3):
#     with torch.no_grad():
#         pred = torch.topk(output, k, dim=1)[1]
#         assert pred.shape[0] == len(target)
#         correct = 0
#         for i in range(k):
#             correct += torch.sum(pred[:, i] == target).item()
#     return correct / len(target)

#（Mean Absolute Error，MAE）
def mae_xs(output, target):
    with torch.no_grad():
        MAE_x = torch.mean(torch.abs(output[:,0]-target[:,0])).item()
    return MAE_x * 20

def mae_ys(output, target):
    with torch.no_grad():
        MAE_y = torch.mean(torch.abs(output[:,1]-target[:,1])).item()
    return MAE_y * 20

def mae_xt(output, target):
    with torch.no_grad():
        MAE_x = torch.mean(torch.abs(output[:,2]-target[:,2])).item()
    return MAE_x

def mae_yt(output, target):
    with torch.no_grad():
        MAE_y = torch.mean(torch.abs(output[:,3]-target[:,3])).item()
    return MAE_y