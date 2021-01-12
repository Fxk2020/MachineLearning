# coding=utf-8
import random
import sys
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D  # 用来给出三维坐标系

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

# 一维数据
w_1 = np.array([w[:, 6]])
# 二维数据
w_2 = np.array([w[:, 3:5]])[0]

# 测试集
x_1 = np.array([0.04, 0.62, 3.2])
x_2 = np.array([-0.7, 0.61, -0.28])
x_3 = np.array([0.3, 1.61, -0.25])
test_x = [x_1, x_2, x_3]

w1 = np.array(w[:, 0:3])
w2 = np.array(w[:, 3:6])
w3 = np.array(w[:, 6:9])
train_w = [w1, w2, w3]


def get_p(w, x, k, N, dimension):
    if dimension == 1:
        ls = []
        for i in x:
            temp = []
            for j in w:
                temp.append(j)
            for l in range(k):
                min_ri = sys.maxsize
                min_value = temp[0]
                for w_i in temp:
                    current_ri = abs(w_i - i)
                    if current_ri < min_ri:
                        min_ri = current_ri
                        min_value = w_i
                if l < k - 1:
                    temp.remove(min_value)  # 更新样本集
                else:
                    ls.append(k / N / min_ri)  # 一维时概率
    if dimension == 2:
        X, Y = np.mgrid[-3:3:50j, -2:4:50j]
        ls = np.zeros((50, 50))
        for i in range(50):
            for j in range(50):
                distances = []
                x = np.array([[X[i][j], Y[i][j]]]).T
                for w_i in w:
                    distances.append(math.sqrt(math.pow(w_i[0] - x[0][0], 2) + math.pow(w_i[1] - x[1][0], 2)))  # 欧式距离
                distances = sorted(distances)
                ls[i][j] = k / ((distances[k - 1] ** 2 * np.pi) * N)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(X, Y, ls, rstride=1, cstride=1, cmap=plt.cm.coolwarm)
        ax.set_xlabel('x', color='b')
        ax.set_ylabel('y', color='r')
        ax.set_zlabel('Pn(x)', color='g')
        plt.title("k=" + str(k))
        plt.show()
    if dimension == 3:
        distances = []
        ls = []
        for wi in w:
            distances.append(np.linalg.norm(x - wi))
        distances = sorted(distances)
        value = k / (((4 / 3) * math.pi * distances[k - 1] ** 3) * N)
        ls.append(value)

    return ls


def show_one(k_in):
    rand1 = []
    for i in range(1000):
        rand1.append(-5 + random.random() * 10)  # 随机数处于-5～5之间
    rand1.sort()  # 对数据进行排序，画的图更有利于观察

    k = k_in
    dimension = 1
    p_x = get_p(w_1[0], rand1, k, len(w_1[0]), dimension)

    plt.title("N=" + str(len(w_1[0])) + ",k=" + str(k) + "")
    plt.xlabel("x")
    plt.ylabel("P(x)")
    plt.plot(rand1, p_x)
    plt.show()


def show_two(k_in):
    k = k_in
    dimension = 2
    Z = get_p(w_2, np.array([]), k, len(w_2), dimension)


def judgment(k_in):
    k = k_in
    dimension = 3
    for xi in test_x:
        for wi in train_w:
            value = get_p(wi, xi, k, len(w1), dimension)
            print("当k=" + str(k) + "时对于W1,在{}点的概率密度值为:{:.6f}".format(xi, value[0]))


show_one(1)
show_one(3)
show_one(5)
# show_two(1)
# show_two(3)
# show_two(5)
# judgment(1)
# print()
# judgment(3)
# print()
# judgment(5)
