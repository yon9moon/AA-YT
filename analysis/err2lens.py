import json
import matplotlib.pyplot as plt
from collections import defaultdict

# 加载output.json文件
with open('analysis/output.json', 'r') as f:
    data = json.load(f)

# 获取lens_dir和absolute_errors的数据
lens_dirs = data['lens_dirs']
absolute_errors = [error[0] for error in data['absolute_errors']]

# 使用defaultdict来存储每个lens_dir的absolute_errors的总和和数量
errors_by_lens_dir = defaultdict(lambda: [0, 0])
for lens_dir, error in zip(lens_dirs, absolute_errors):
    errors_by_lens_dir[lens_dir][0] += error
    errors_by_lens_dir[lens_dir][1] += 1

# 计算每个lens_dir的absolute_errors的平均值
average_errors_by_lens_dir = {lens_dir: total / count for lens_dir, (total, count) in errors_by_lens_dir.items()}

# 获取lens_dir的名字（只取最后一个斜杠后的部分）和对应的平均错误
lens_dirs = [lens_dir.split('/')[-1] for lens_dir in average_errors_by_lens_dir.keys()]
average_errors = list(average_errors_by_lens_dir.values())

# 将lens_dirs和average_errors打包成元组列表，然后按照lens_dir排序
sorted_pairs = sorted(zip(lens_dirs, average_errors), key=lambda pair: pair[0])

# 解压排序后的元组列表，得到排序后的lens_dirs和average_errors
lens_dirs, average_errors = zip(*sorted_pairs)

# 创建一个新的图形
plt.figure(figsize=(10, 6))

# 画柱状图
plt.bar(lens_dirs, average_errors)

# 设置图形的标题和坐标轴的标签
plt.title('Average Absolute Errors for Each Lens')
plt.xlabel('Lens')
plt.ylabel('Average Absolute Error')

# 保存图像
plt.savefig('analysis/lens_output_valid.png')