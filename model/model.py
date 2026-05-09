import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models

class AAmodel(nn.Module):
    def __init__(self, img_num, in_channels, num_classes=4):
        super(AAmodel, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        
        # 修改第一层
        self.backbone.conv1 = nn.Conv2d(
            img_num * in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        fc_in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # XY 分支：增加 BN 和 Dropout，提升稳定性
        self.head_xy = nn.Sequential(
            nn.Linear(fc_in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        
        # UV 分支：进一步加强正则化，防止死记硬背
        self.head_uv = nn.Sequential(
            nn.Linear(fc_in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),               # 提高 Dropout 比例
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),           # 增加 BN
            nn.ReLU(),
            nn.Dropout(0.3),               # 多层 Dropout
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        
    def forward(self, x):
        if x.dim() == 5:
            x = x.view(x.size(0), -1, x.size(3), x.size(4))
        
        # 共享 Backbone 提取底层特征
        features = self.backbone(x)
        
        # 分支各自进行高维映射
        xy = self.head_xy(features)
        uv = self.head_uv(features)
        
        return torch.cat([xy, uv], dim=1)