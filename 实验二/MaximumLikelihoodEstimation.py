# coding=utf-8
from typing import List

import numpy as np

# 输入数据
w1 = np.array([
    [0.011, 1.03, -0.21],
    [1.27, 1.28, 0.08],
    [0.13, 3.12, 0.16],
    [-0.21, 1.23, -0.11],
    [-2.18, 1.39, -0.19],
    [0.34, 1.96, -0.16],
    [-1.38, 0.94, 0.45],
    [-1.02, 0.82, 0.17],
    [-1.44, 2.31, 0.14],
    [0.26, 1.94, 0.08]
])

w2 = np.array([
    [1.36, 2.17, 0.14],
    [1.41, 1.45, -0.38],
    [1.22, 0.99, 0.69],
    [2.46, 2.19, 1.31],
    [0.68, 0.79, 0.87],
    [2.51, 3.22, 1.35],
    [0.60, 2.44, 0.92],
    [0.64, 0.13, 0.97],
    [0.85, 0.58, 0.99],
    [0.66, 0.51, 0.88]
])


# 1.求最大似然估计的均值和方差
def get_u_oneDim(w):
    row = w.shape[0]
    sum = 0
    for i in range(row):
        sum += w[i]
    number_average = sum / row
    return number_average


def get_sigma_oneDim(w):
    u = get_u_oneDim(w)
    sum = 0
    row = w.shape[0]
    for i in range(row):
        temp = (w[i] - u) * (w[i] - u)
        sum += temp
    return sum / row


# 2处理多维数据
# 获得均值U
def get_u(w):
    row = w.shape[0]  # 获取第一维度的数目（行）
    col = w.shape[1]  # 获取第二维度的数目（列）
    ls_average = []
    for i in range(col):
        sum = 0
        for j in range(row):
            sum += w[j][i]
        ls_average.append(sum / row)
    ls_u = []
    ls_u.append(ls_average)
    return np.array(ls_u).T


# 获得方差/协方差矩阵
def get_sigma(w):
    row = w.shape[0]  # 获取第一维度的数目（行）
    col = w.shape[1]  # 获取第二维度的数目（列）

    u = get_u(w)  # 获得均值
    sum_matrix = np.zeros([col, col])  # 初始化矩阵
    for i in range(row):
        sum_matrix += np.dot(np.array([w[i, :]]).T - u, (np.array([w[i, :]]).T - u).T)
    return (1 / row) * sum_matrix


def get_sigma_known(w):
    col = w.shape[1]  # 获取第二维度的数目（列）
    array_sigma = np.zeros(col)
    for i in range(col):
        array_sigma[i] = get_sigma_oneDim(w[:, i])
    return np.diag(array_sigma)


def main1():
    # T1：均值和方差
    for i in range(w1.shape[1]):
        print("类一的x" + str(i + 1) + "的均值和方差分别为：")
        print("𝝁̂=" + str(get_u_oneDim(w1[:, i])))
        print("𝜎̂2=" + str(get_sigma_oneDim(w1[:, i])))
    print
    for i in range(w2.shape[1]):
        print("类二的x" + str(i + 1) + "的均值和方差分别为：")
        print("𝝁̂="+str(get_u_oneDim(w2[:, i])))
        print("𝜎̂2="+str(get_sigma_oneDim(w2[:, i])))
    print()


def main2():
    # T2处理二维数据
    w1_x1 = w1[:, 0:2]
    w1_x2 = w1[:, 1:3]
    w1_x3 = np.array(np.row_stack((w1[:, 0], w1[:, 2]))).T  # 将两个列向量合称为一个矩阵
    w2_x1 = w2[:, 0:2]
    w2_x2 = w2[:, 1:3]
    w2_x3 = np.array(np.row_stack((w2[:, 0], w2[:, 2]))).T
    w1_s = [w1_x1, w1_x2, w1_x3]
    w2_s = [w2_x1, w2_x2, w2_x3]
    for i in range(3):
        print("第一类数据，第" + str(i + 1) + "种可能的情况下，二维似然估计的均值𝝁̂为")
        print(get_u(w1_s[i]))
        print("二维似然估计的方差𝚺̂为")
        print(get_sigma(w1_s[i]))
    print()
    for i in range(3):
        print("第二类数据，第" + str(i + 1) + "种可能的情况下，二维似然估计的均值𝝁̂为")
        print(get_u(w2_s[i]))
        print("二维似然估计的方差𝚺̂为")
        print(get_sigma(w2_s[i]))


def main3():
    # T3处理三维数据（𝛍, 𝚺均未知）
    print("𝛍, 𝚺均未知的情况下，类一的三维似然估计的均值𝝁̂和方差𝚺̂分别为")
    print(get_u(w1))
    print(get_sigma(w1))
    print()
    print("𝛍, 𝚺均未知的情况下，类二的三维似然估计的均值𝝁̂和方差𝚺̂分别为")
    print(get_u(w2))
    print(get_sigma(w2))
    print()


def main4():
    # T4处理三维数据（𝛍未知𝚺已知)
    print("𝛍未知𝚺已知的情况下，类一的三维似然估计的均值𝝁̂和方差𝚺̂分别为:")
    print(get_u(w1))
    print(get_sigma_known(w1))
    print("𝛍未知𝚺已知的情况下，类二的三维似然估计的均值𝝁̂和方差𝚺̂分别为:")
    print(get_u(w2))
    print(get_sigma_known(w2))


main1()
main2()
main3()
main4()
