"""
network.py
~~~~~~~~~~

这是一个示例性质的采用随机梯度下降法的亲阿奎神经网络算法。侧重于演示神经网络的工作原理。
采用反向传播法啦计算每个参数的梯度。
此示例代码精简但功能有限

"""


# Standard library
import random
import pprint

# Third-party libraries
import numpy as np


# math_fuction

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

# main_class


class Network:

    def __init__(self, sizes):
        """列表sizes包含每层神经元的数量。
        比如sizes=[2, 3, 1]是指会建立三层神经元的列表，
        第一层是2个神经元，第二层是3个神经元，第三层是1个神经元。
        这个模型偏置是设置在神经元处的，不是和输入在一起的。
        但要注意sizes[0]指的是神经元第一层，这一层实际上是输入，不设置偏置。
        权重和偏置以Numpy矩阵表的形式存储。
        weights[1]是第二层到第三层的权重矩阵。biases[1]是第二层的偏置向量。
        注意下标和层数的关系。
        此模型随机初始化所有权重和偏置的第一组参数，以继续迭代。"""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        # sizes[:-1]取不到最后一个，sizes[1:]取不到索引0。保证x从第二个是左边一层的神经元个数，保证y是右边一层神经元个数。

    def feedforward(self, a):
        """给定一个输入a返回一个输出"""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """该神经网络模型采用小批量随机梯度下降法。并且支持修改批量的大小、学习速率、训练迭代期
        taining_data是一个内容为元组的列表，用来存放训练集中的每一对x、y。epoch是迭代期数量，而mini_batch是小批量数据大小。eta代表学习速率。
        如果给了参数test_data会在每个训练器后评估网络并打印出进展，但会拖慢运行速度。"""
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)  # 乱序
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            # 将training_data按照小批量数据个数划分成若干块，以二维列表的形式放在mini_batches
            for mini_batch in mini_batches:  # mini_batch是每一个小批量数据快
                self.update_mini_batch(mini_batch, eta)  # 对每个小批量数据块mini_batch更新一次数据
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))

            else:
                print("Epoch {0} complete".format(j))
            # test_data拖慢进度的原因：每利用一个小批量数据块计算完下一个参数，都会计算上一个参数的准确率并打印出来

    def update_mini_batch(self, mini_batch, eta):
        """这是有一个小批量数据块的所有询量元素，更新一组新参数的函数。
        它的核心是重读计算每个参数的梯度，然后以小批量块为单元求平均。
        最后求出下一组的参数，并且使用反向传播算法，来快速计算代价函数的梯度。
        传入的mini_batch应该是从training_data中分出来的小批量数据块，因此它也是元组组成的列表。"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]  # 复制一个空的参数组
        for x, y in mini_batch:  # 对小批量块中每个训练集元素迭代，并且对每个训练集的元素运用反向传播
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """返回值是一个元组``(nabla_b, nabla_w)``，这个元组里面有两个元素nabla_b, nabla_w，
        他们的形状是类似于 ``self.biases`` 和 ``self.weights``的神经网络层与层之间的参数矩阵。
        而nabla_b, nabla_w中每个参数存着的都是某个参数所对应的代价函数的梯度。"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # 前向传播：计算每一层的输入向量和输出向量
        activation = x
        activations = [x]  # 这是一个列表，列表中每个元素都是一维数组，该一维数组存储着每一层的输出a向量
        zs = []  # 这是一个列表，列表中每个元素都是一维数组，该一维数组存储着每一层的输入z向量
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # 反向传播：先计算最后一层的梯度矩阵，再通过l层和l+1层的递推关系算所有层的梯度矩阵
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
        # 这是一个生成器推导

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)


