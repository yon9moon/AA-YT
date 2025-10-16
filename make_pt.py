import argparse
import torch
import numpy as np
from model.model import *
import os

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def main(args):
    device = 'cpu'
    img_input = int(args.imgs)
    # 获取 args.path 的目录部分
    pretrained_path = args.path
    print(pretrained_path)
    path = os.path.dirname(pretrained_path)
    print(path)
    
    net = AAmodel(img_num=img_input, in_channels=1, num_classes=2).to(device)
    weights = torch.load(pretrained_path, map_location='cpu')    
    net.load_state_dict(weights['state_dict'])
    net.eval()

    input = torch.rand(1, img_input, 1, 480, 640)

    traced_script_module_resnet = torch.jit.trace(net, input)
    traced_script_module_resnet.save(path + "/model_best.pt") 
    print("model_best.pt saved in " + path + "/model_best.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-p', '--path', default=None, type=str)
    parser.add_argument('-i', '--imgs', default=None, type=str)
    parser.add_argument('-d', '--device', default=None, type=str)
    args = parser.parse_args()
    main(args)