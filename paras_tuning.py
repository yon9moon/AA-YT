import json
import subprocess
import numpy as np

# 定义学习率和权重衰减的值
lr_values = np.linspace(0.001, 0.01, num=5).tolist()
weight_decay_values = np.linspace(0, 0.001, num=5).tolist()


## 打开现有的config.json文件
with open('config.json', 'r') as file:
    config = json.load(file)

# 遍历学习率和权重衰减的值
for lr in lr_values:
    for weight_decay in weight_decay_values:
        # 更新lr和weight_decay
        config['optimizer']['args']['lr'] = lr
        config['optimizer']['args']['weight_decay'] = weight_decay

        # 写回到config.json
        with open('config.json', 'w') as file:
            json.dump(config, file, indent=4)

        # 运行训练命令
        subprocess.run("CUDA_VISIBLE_DEVICES=1 python train.py -c config.json", shell=True)