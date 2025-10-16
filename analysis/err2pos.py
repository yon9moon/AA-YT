import matplotlib.pyplot as plt
import numpy as np
import json
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict


def XYplot(data):
    # 提取数据
    x = []
    y = []
    z = []

    for key, value in data.items():
        cat_1, cat_2 = map(int, key.split())
        x.append(cat_1)
        y.append(cat_2)
        z.append(np.sqrt(np.mean(np.square(value["absolute_errors"]))) )  # 计算均方根误差


    # 创建3D图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制散点图
    ax.scatter(x, y, z)

    # # 绘制柱状图
    # ax.bar3d(x, y, np.zeros(len(z)), 1, 1, z)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 保存图像
    plt.savefig('analysis/XY_valid.png')

def XorYplot(data):
    # 提取数据
    x = []
    y = []
    z1 = []
    z2 = []

    # 提取 "truth" 中的数据
    for pair in data["truth"]:
        x.append(pair[0])
        y.append(pair[1])

    for pair in data["absolute_errors"]:
        z1.append(pair[0])
        z2.append(pair[1])

    # 创建一个字典来存储每个唯一值及其对应的平均值
    avg_dictX = defaultdict(list)
    avg_dictY = defaultdict(list)

    for x_value, z_value in zip(x, z1):
        avg_dictX[x_value].append(z_value)

    avg_dictX = {x: sum(z_values) / len(z_values) for x, z_values in avg_dictX.items()}

    for y_value, z_value in zip(y, z2):
        avg_dictY[y_value].append(z_value)

    avg_dictY = {y: sum(z_values) / len(z_values) for y, z_values in avg_dictY.items()}

    fig, axs = plt.subplots(2, figsize=(10, 10))

    # 绘制第一个柱状图
    axs[0].bar(avg_dictX.keys(), avg_dictX.values())
    axs[0].set_xlabel('X position')
    axs[0].set_ylabel('Average absolute error')

    # 绘制第二个柱状图
    axs[1].bar(avg_dictY.keys(), avg_dictY.values())
    axs[1].set_xlabel('Y position')
    axs[1].set_ylabel('Average absolute error')

    # 保存图像
    plt.savefig('analysis/XorY_train.png')

def Zplot(data):
    # 提取数据
    x = []
    z = []

    for num in data["truth"]:
        x.append(num[0])
    
    for num in data["absolute_errors"]:
        z.append(num[0])

    # 创建一个字典来存储每个唯一值及其对应的平均值
    avg_dictX = defaultdict(list)

    for x_value, z_value in zip(x, z):
        avg_dictX[x_value].append(z_value)

    avg_dictX = {x: sum(z_values) / len(z_values) for x, z_values in avg_dictX.items()}

    # 创建柱状图
    plt.bar(avg_dictX.keys(), avg_dictX.values())
    plt.xlabel('Z pos')
    plt.ylabel('Average Errors')

    # 保存图像
    plt.savefig('analysis/Z_valid.png')
   

if __name__ == "__main__":
    # 打开并读取.json文件
    with open('analysis/output.json', 'r') as f:
        data = json.load(f)

    # XYplot(data)
    Zplot(data)