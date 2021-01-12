# coding=utf-8
import sys

import numpy
import math

# 实验数据
w1 = numpy.array([[-5.01, -8.12, -3.68], [-5.43, -3.48, -3.54], [1.08, -5.52, 1.66],
                  [0.86, -3.78, -4.11], [-2.67, 0.63, 7.39], [4.94, 3.29, 2.08], [-2.51, 2.09, -2.59],
                  [-2.25, -2.12, -6.94], [5.56, 2.86, -2.26], [1.03, -3.33, 4.33]])
w2 = numpy.array([[-0.91, -0.18, -0.05], [1.3, -2.06, -3.53], [-7.75, -4.54, -0.95],
                  [-5.47, 0.5, 3.92], [6.14, 5.72, -4.85], [3.6, 1.26, 4.36], [5.37, -4.63, -3.65],
                  [7.18, 1.46, -6.66], [-7.39, 1.17, 6.3], [-7.5, -6.32, -0.31]])
w3 = numpy.array([[5.35, 2.26, 8.13], [5.12, 3.22, -2.66], [-1.34, -5.31, -9.87],
                  [4.48, 3.42, 5.19], [7.11, 2.39, 9.21], [7.17, 4.33, -0.98], [5.75, 3.97, 6.65],
                  [0.77, 0.27, 2.41], [0.9, -0.43, -8.71], [3.52, -0.36, 6.43]])


#  获得均值向量u
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
    return numpy.array(ls_u).T  # 转置得到列向量


# 求协方差矩阵
def get_sigma(w):
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

    for i in range(col):
        for j in range(row):
            # print(str(ls_u[0][i]))
            w[j][i] = w[j][i] - ls_u[0][i]

    sigma = 1 / (row - 1)
    sigma = sigma * numpy.dot(w.T, w)
    return sigma


#  求马氏距离
def get_mahalanobis_distance(x, w):
    # sigma = numpy.cov(w.T)  # 求协方差矩阵 库函数
    temp = w.copy()
    sigma = get_sigma(temp)  # 求协方差矩阵 自己实现
    u = get_u(w)
    sigma_inverse = numpy.linalg.inv(sigma)  # 矩阵求逆
    tp = x - u
    return numpy.sqrt(numpy.dot(numpy.dot(tp.T, sigma_inverse), tp))[0][0]  # 矩阵相乘


# 判别函数
def discriminant_function(x, w, p):
    temp = w.copy()
    sigma = get_sigma(temp)
    u = get_u(w)
    sigma_inverse = numpy.linalg.inv(sigma)
    tp = x - u
    # 返回判别函数的值 d是维数
    return -0.5 * numpy.dot(numpy.dot(tp.T, sigma_inverse), tp) - u.shape[0] / 2.0 * numpy.log(
        2 * math.pi) - 0.5 * numpy.log(abs(numpy.linalg.det(sigma))) + numpy.log(p)


# 根据判别函数分类
def classification(x, w, p):
    print("当先验分别为：")
    for p_index in p:
        print(p_index)

    classifications = []
    for i in x:
        class_number = 0
        max_g = -sys.maxsize-1  # 最小的数
        ls = []
        for j in range(len(w)):
            g = discriminant_function(i, w[j], p[j])
            print(str(i.T[0])+"在w"+str(j+1)+"的判别函数值为"+str(g[0][0]))
            if g > max_g:
                class_number = j + 1  # 判定所属类别
                max_g = g
        ls.append(str(i.T[0]))
        ls.append(class_number)
        classifications.append(ls)
    return classifications


x1 = numpy.array([[1], [2], [1]])
x2 = numpy.array([[5], [3], [2]])
x3 = numpy.array([[0], [0], [0]])
x4 = numpy.array([[1], [0], [0]])

list_x = [x1, x2, x3, x4]
list_w = [w1, w2, w3]


for i in list_x:
    index = 0
    for j in list_w:
        index = index + 1
        print("测试点" + str(i.T[0]) + "到w" + str(index) + "的马氏距离为："+str(get_mahalanobis_distance(i, j)))
print()

p1 = [1 / 3, 1 / 3, 1 / 3]
classes_1 = classification(list_x, list_w, p1)
for i in classes_1:
    print("测试点" + str(i[0]) + "属于" + str(i[1]) + "类")
print()

p2 = [0.8, 0.1, 0.1]
classes_2 = classification(list_x, list_w, p2)
for j in classes_2:
    print("测试点" + str(j[0]) + "属于" + str(j[1]) + "类")
