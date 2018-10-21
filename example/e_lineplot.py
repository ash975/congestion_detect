"""
=======================================
A simple plot with a custom dashed line
=======================================

A Line object's ``set_dashes`` method allows you to specify dashes with
a series of on/off lengths (in points).
"""
import numpy as np
import matplotlib.pyplot as plt
import json

# from scipy import interpolate

data_file = open('./opencv_sample/result_file_101_0828_2', 'r')
data = data_file.read()
data_json = json.loads(data)
x = []
l1 = []
l2 = []
l3 = []
l4 = []

for item in data_json.items():
    x.append(int(item[0]))

x.sort()

for i in x:
    l1.append(data_json[str(i)][0])
    l2.append(data_json[str(i)][1])
    l3.append(data_json[str(i)][2])
    l4.append(data_json[str(i)][3])

# 数据处理
step = int(len(x) / 20 * 1.8)
x_position = [x[step*i] for i in range(1, 11)]
l1_mean_1800 = [int(np.mean(l1[step * i: step * (i + 1)])) for i in range(0, 10)]
l2_mean_1800 = [int(np.mean(l2[step * i: step * (i + 1)])) for i in range(0, 10)]
l3_mean_1800 = [int(np.mean(l3[step * i: step * (i + 1)])) for i in range(0, 10)]
l4_mean_1800 = [int(np.mean(l4[step * i: step * (i + 1)])) for i in range(0, 10)]
print('step ', step)
print('x_position', x_position)
print('l1_mean_1800: ', l1_mean_1800)
print('l2_mean_1800: ', l2_mean_1800)
print('l3_mean_1800: ', l3_mean_1800)
print('l4_mean_1800: ', l4_mean_1800)


# 绘制图像
fig, ax = plt.subplots()
line1, = ax.plot(x_position, l1_mean_1800, '--', linewidth=2, label='l1_mean_1800')
line2, = ax.plot(x_position, l2_mean_1800, '-+', linewidth=2, label='l2_mean_1800')
line3, = ax.plot(x_position, l3_mean_1800, '-^', linewidth=2, label='l3_mean_1800')
line4, = ax.plot(x_position, l4_mean_1800, '-*', linewidth=2, label='l4_mean_1800')

m = int(len(x))
plt.scatter(x[0:m], l1[0:m], c='r', marker='o', label='tf')
plt.scatter(x[0:m], l2[0:m], c='k', marker='*', label='t1')
plt.scatter(x[0:m], l3[0:m], c='b', marker='+', label='t2')
plt.scatter(x[0:m], l4[0:m], marker='>', label='t3')
plt.legend(['tf_mean_1800', 't1_mean_1800', 't2_mean_1800', 't3_mean_1800', 'tf', 't1', 't2', 't3'])
plt.show()
print(x[int(len(l2)/20 * 1.8)])




