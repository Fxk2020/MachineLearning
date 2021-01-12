# coding=utf-8

import numpy as np
import math

# 实验数据输入
w = np.array([
    [0.67, 0.173, 0.85, -0.15, 0.84, 0.359, 1.36, 1.86, 0.256],
    [0.05, -3.04, -3.14, -0.06, 0.53, 0.23, 1.41, 1.86, 0.75],
    [1.55, -0.06, 1.96, 0.63, 0.315, 0.235, 1.22, -0.15, 0.59],
    [0.64, 0.96, 0.5, 0.1, 0.79, 0.281, 2.46, -0.19, 1.67],
    [-1.35, 5.56, 0.11, -0.1, 0.73, 0.304, 0.68, 0.61, 3.37],
    [0.221, 1.14, -4.44, 0.42, 0.95, 0.37, 2.51, -0.22, 0.38],
    [0.02, 2.16, 2.46, 0.239, 0.81, 0.09, 0.6, 0.181, 0.41],
    [0.52, -0.04, -0.6, -0.02, 0.87, 0.39, 0.64, 0.04, 2.47],
    [-1.65, 1.02, -1.83, 0.185, 0.75, 0.271, 0.85, 1.46, -0.19],
    [1.12, -0.75, -2.33, 0.13, 0.314, 0.207, 0.66, 0.15, -0.22]
])

# 训练集
w1 = w[:, 0:3]
w2 = w[:, 3:6]
w3 = w[:, 6:9]

x1 = np.array([[0.3, 1.5, 0.4]])
x2 = np.array([[0.21, 0.42, 0.18]])
x3 = np.array([[0.2, 0.56, -0.1]])

# 题目已知
h = 1
list_x = [x1, x2, x3]
list_w = [w1, w2, w3]


# Parzen窗估计
def parzen_windows(x, w, h):
    p_x = list(range(len(x)))
    for i in range(len(x)):
        x_temp = x[i]
        max_p_x = 0
        # print(x_temp)
        for j in range(len(w)):
            k_n = 0
            row = w[j].shape[0]
            for k in range(row):
                tp = w[j][k] - x_temp
                k_n = k_n + math.exp(-np.dot(tp, tp.T) / (2 * math.pow(h, 2)))
                # 体积都是相同的无需计算
            temp_p_x = k_n / row
            # print(temp_p_x)
            if temp_p_x > max_p_x:
                max_p_x = temp_p_x
                p_x[i] = j + 1
    return p_x


print('==========当h=1时============')
classification = parzen_windows(list_x, list_w, 1)
for i in range(len(classification)):
    print('样本点{0}属于{1}类'.format(list_x[i][0], classification[i]))
