# coding=utf-8
import numpy
import matplotlib.pyplot as plt
from TestDic import load


#  定义numNeuronLayers, numNeurons_perLayer，learningrate来初始化网络层数，每层神经元个数和学习率
class NeuralNetwork:
    def __init__(self, numNeuronLayers, numNeurons_perLayer, learningrate):
        self.numNeurons_perLayer = numNeurons_perLayer
        self.numNeuronLayers = numNeuronLayers
        self.learningrate = learningrate
        self.weight = []
        self.bias = []
        # w和b的初始化
        for i in range(numNeuronLayers):
            self.weight.append(numpy.random.normal(0.0, pow(self.numNeurons_perLayer[i + 1], -0.5),
                                                   (self.numNeurons_perLayer[i + 1], self.numNeurons_perLayer[i])))
        for i in range(numNeuronLayers):
            self.bias.append(numpy.random.randn(numNeurons_perLayer[i + 1], 1))
        print(self.weight[1].shape[1])
        # 另外的可行的激活函数
        # x: (numpy.exp(x) - numpy.exp(-x)) / (numpy.exp(x) + numpy.exp(-x))
        self.activation_function = lambda x: 1.0 / (1.0 + numpy.exp(-x))

    # 训练函数
    def update(self, input_nodes, targets):
        inputs = numpy.array(input_nodes, ndmin=2).T
        targets = numpy.array(targets, ndmin=2).T

        # 前向传播——关键在于计算出每层的损失
        # 定义输出值列表（outputs[0]为输入值）
        self.outputs = []
        self.outputs.append(inputs)
        # 将神经网络的每一层计算输出值作为输入放入激活函数中，并保存到outputs列表中
        for i in range(self.numNeuronLayers):
            # 每个神经元的输入信号z，由参数w、b，和前一层的输出信号a决定
            temp_inputs = numpy.dot(self.weight[i], inputs) + self.bias[i]
            temp_outputs = self.activation_function(temp_inputs)
            inputs = temp_outputs
            self.outputs.append(temp_outputs)
        # 计算每层的损失
        self.output_errors = []
        for i in range(self.numNeuronLayers):
            # 输出层的误差=目标值-输出值
            if i == 0:
                self.output_errors.append(targets - self.outputs[-1])
            # 隐藏层的误差=（当前隐藏层与下一层之间的权值矩阵）的转置与下一层的误差矩阵的乘积
            else:
                self.output_errors.append(numpy.dot((self.weight[self.numNeuronLayers - i]).T,
                                                    self.output_errors[i - 1]))
        # 反向传播
        for i in range(self.numNeuronLayers):
            # 权值更新规则为之前 新权值=权值+学习率*误差*激活函数的导数*上一层输出
            # 偏移量b的更新规则 新偏执因子=偏执因子+学习率*误差*激活函数的导数
            # f(x)*（1-f(x)）即为激活函数f(x)的导函数
            self.weight[self.numNeuronLayers - i - 1] += \
                self.learningrate * numpy.dot(
                    self.output_errors[i] * self.outputs[-1 - i] * (1 - self.outputs[-1 - i]),
                    self.outputs[-2 - i].T)
            self.bias[self.numNeuronLayers - i - 1] += self.learningrate * (
                    self.output_errors[i] * self.outputs[-1 - i] * (1 - self.outputs[-1 - i]))

    # 将测试用例作为输入，让模型走一遍前向传播过程得到输出，然后返回输出结果与测试用例标签的一致比例
    def test(self, test_inputnodes, test_labels):
        inputs = numpy.array(test_inputnodes, ndmin=2).T
        # 走一遍前向传播得到输出
        for i in range(self.numNeuronLayers):
            temp_inputs = numpy.dot(self.weight[i], inputs) + self.bias[i]
            temp_outputs = self.activation_function(temp_inputs)
            inputs = temp_outputs
        # 返回模型输出结果是否与测试用例标签一致
        # 测试用例在模型中输出的是第几类：
        # print(list(inputs).index(max(list(inputs))))
        # 测试用例属于第几类：
        # print(list(test_labels).index(1))
        return list(inputs).index(max(list(inputs))) == list(test_labels).index(1)


# 获取分类正确的个数用来计算loss
def getTrueNumber(images_data, labels, n):
    count_temp = 0
    for i in range(len(images_data)):
        if n.test(images_data[i], labels[i]):
            count_temp += 1
    return count_temp


# 载入训练集和测试集，神经网络定义为三层，输入层节点为3072，隐藏层节点为50，输出节点为10个
if __name__ == '__main__':
    learning_rate = 0.01
    # 定义训练集——分为训练和验证
    images_data, labels = load('train')
    verification_data = images_data[49000:50000, :]
    images_data = images_data[0:49000, :]
    # images_data = images_data[0:9000, :] 看图的效果时使用的小的集合
    verification_labels = labels[49000:50000, :]
    labels = labels[0:49000, :]
    # labels = labels[0:9000, :]
    # 定义测试集
    test_images_data, test_labels = load('test')
    ls = [3072, 50, 10]
    # 神经网络的层数是从隐藏层开始计数的，输入层不计入总层数。
    n = NeuralNetwork(2, ls, learning_rate)
    train_loss = []
    test_loss = []
    cycles = 50
    for i in range(cycles):
        print(i)
        for i in range(len(images_data)):
            n.update(images_data[i], labels[i])
        print("train_loss=" + str((len(images_data) - getTrueNumber(images_data, labels, n)) / len(images_data)))
        train_loss.append((len(images_data) - getTrueNumber(images_data, labels, n)) / len(images_data))
        print("test_loss=" + str(
            (len(verification_data) - getTrueNumber(verification_data, verification_labels, n)) / len(
                verification_data)))
        test_loss.append((len(verification_data) - getTrueNumber(verification_data, verification_labels, n)) / len(
            verification_data))
    x = []
    for i in range(cycles):
        x.append(i + 1)
    plt.plot(x, train_loss, color="r", linestyle="--", marker="*", linewidth=2.0, label='loss')
    plt.plot(x, test_loss, color="b", linewidth=2.0, label='val_loss')
    plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))
    plt.title("LOSS")
    plt.show()

    count = 0
    for i in range(1000):
        if n.test(test_images_data[i], test_labels[i]):
            count = count + 1
    print("学习率为" + str(learning_rate))
    print(count / 1000)
