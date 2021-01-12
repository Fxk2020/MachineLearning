# coding=utf-8
import numpy as np
import pickle

fileUrl = "./cifar-10-batches-py/data_batch_2"
test_fileUrl = "./cifar-10-batches-py/test_batch"


# 在字典结构中，每一张图片是以被展开的形式存储（即一张32x32的3通道图片被展开成了3072长度的list），每一个数据的格式为uint8
# 前1024个数据表示红色通道，接下来的1024个数据表示绿色通道，最后的1024个通道表示蓝色通道。
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# 载入全部训练集
def unpickleAll():
    fileUrl_one = "./cifar-10-batches-py/data_batch_"
    data = unpickle(fileUrl_one + str(1))[b'data']
    labels = []
    for i in range(5):
        # print(fileUrl_one+str(i+1))
        if i > 0:
            data = np.vstack((data, unpickle(fileUrl_one + str(i + 1))[b'data']))
        labels += unpickle(fileUrl_one + str(i + 1))[b'labels']
    return data, labels


# 载入单个训练集
def unpickleOne():
    fileUrl_one = "./cifar-10-batches-py/data_batch_"
    data = unpickle(fileUrl_one + str(1))[b'data']
    labels = unpickle(fileUrl_one + str(1))[b'labels']
    return data, labels


# 为实验载入数据
def load(kind):
    if kind == 'train':
        # img_matrix, labels = unpickleOne()
        img_matrix,labels = unpickleAll()
    if kind == 'test':
        d = unpickle(test_fileUrl)
        img_matrix = d[b'data']  # 是一个10000x3072的array，每一行的元素组成了一个32x32的3通道图片
        labels = d[b'labels']  # 记录图片的标签（0～9）
    # 读取到的labels为0-9的数字（表示该数据所属的类别），需转化为十位的numpy数组
    ls = []
    for i in range(len(labels)):
        temp = np.zeros(10)
        temp[labels[i]] = 1
        # print("l"+str(labels[i]))
        # print("t"+str(temp.T))
        ls.append(temp.T)
    labels = np.array(ls)
    # 不改变数据大小会导致计算激活函数时溢出
    # print(img_matrix / 2550)
    return (img_matrix / 2550), labels


# images_data, labels, test_data, test_labels = CreatData()
# for i in range(3072):
#     print(images_data[0][i]/2550)
# print(labels)
#
# images_data2, labels2 = load("train")
# for i in range(3072):
#     if images_data[0][i]/2550 == images_data2[0][i]:
#         print("True")
#     else:
#         print(str(images_data[0][i]/2550)+"so so so "+str(images_data2[0][i]))

# d = unpickle(fileUrl)
# img_matrix = d[b'data']  # 是一个10000x3072的array，每一行的元素组成了一个32x32的3通道图片
# lable = d[b'labels']  # 记录图片的标签（0～9）
# batch_labels = d[b'batch_label']  # 记录batch的名称
# filenames = d[b'filenames']  # 记录file的名称
#
# imageUrl = "./img/"
# for i in range(10):
#     image = img_matrix[i]
#     print(str(image.shape[0]))
#
#     print(lable[i])
#
#     # print(batch_labels[i])
#
#     # print(filenames[i])
# img_matrix,labels = unpickleAll()
# print(img_matrix.shape[0])
