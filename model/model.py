import torch.nn as nn
import torch.nn.functional as F
# from base import BaseModel

import torch
from torchvision import datasets, transforms
from torchvision import models

class AAmodel(nn.Module):
    def __init__(self, img_num, in_channels, num_classes):
        super(AAmodel, self).__init__()
        self.model = models.resnet18(pretrained=True)
        # self.model.load_state_dict(torch.load('/home/lhb/backbone/resnet50-19c8e357.pth'))
        conv1 =  self.model.conv1
        conv1 = torch.nn.Conv2d(img_num * in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.conv1 = conv1
        fc = self.model.fc
        fc_in_features = fc.in_features


        # 删除 Dropout 层，并将全连接层直接连接
        self.model.fc = nn.Linear(fc_in_features, num_classes)
        
    def forward(self, x):
        x = x.view(x.size(0), -1, x.size(3), x.size(4))
        x = self.model(x) 

        return x
    

class AAmodel_classification(nn.Module):
    def __init__(self, img_num, in_channels, num_bins=85):
        super(AAmodel_classification, self).__init__()
        self.model = models.resnet18(pretrained=True)
        conv1 = self.model.conv1
        conv1 = torch.nn.Conv2d(img_num * in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.conv1 = conv1
        fc = self.model.fc
        fc_in_features = fc.in_features
        self.model.fc = nn.Linear(fc_in_features, num_bins * num_bins)  # 输出num_bins * num_bins类
        
    def forward(self, x):
        x = x.view(x.size(0), -1, x.size(3), x.size(4))
        x = self.model(x)
        return x