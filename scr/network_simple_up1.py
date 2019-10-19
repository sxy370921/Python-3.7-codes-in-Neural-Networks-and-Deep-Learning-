"""
S型神经元的完全前馈网络，可设置层数和神经元数
Sigmoid激活函数
二次代价、交叉熵可选代价函数
L2规范化
标准正态分布、非标准正态分布可选参数初始化
采用小批量块梯度下降法
自实现反向传播
迭代期次数控制训练停止
使用MNIST数据集
方法介绍：主要是依靠Network类实现训练。
Network中构造函数是用来建立神经网络的
SGD方法是用来完成训练的。在SGD的适用中无论是训练集还是测试集都可以是任意长度，通过切片来设置。
save方法是用来将当前的权重和偏置参数以及神经网络模型存到指定文件中去的。
identification_one是用来对表示灰度图像信息的单个输入一位数组，进行辨识的
identification_data是用来对表示灰度图像信息的一位数组的列表进行全体辨识的
全局函数load是在某个文件中读取一个神经网络结构和参数的，并根据读取回来的数据建立一个新的Network对象的
其他全局函数都是为计算过程服务的。

训练与测试流程：
1.建立Network的对象net
2.对net调用SGD方法训练参数，训练完的参数存储在了net的实力属性里面。
也可以通过打开标志位，监控整个训练过程的效果以及变化。
3.训练完成后可以保存到json文件中
4.训练好以后可以利用identification_one和identification_data方法实现对陌生输入的辨识
5.可以使用load读取某个json文件中上次存储的神经网络结构和参数，建立一个已经训练好的net对象.
可以对这个对象继续训练，也可以直接拿来做手写辨识（利用net对象的identification_one和identification_data方法）。
6.这个up1神经网络也是对浅层的神经网络有效，对深层神经网络效果很差，原因暂时还不知道有待慢慢探索研究

超参数建议：在学习速率为0.1，epoches为60，小批量大小为10，L2规范化系数为5.0，
只有一个隐藏层且隐藏层神经元为100个的情况下可达到97.8%左右的正确率
"""

# Standard library
import json
import random
import sys

# Third-party libraries
import numpy as np


# translation
def vectorized_result(j):
    """
    输出格式的转化，将输出的数字形式转化成含有10个0/1数字一维数组的形式
    为数字手写辨识量身定做。
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


# activation functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


# cost functions
class QuadraticCost:  # 二次代价函数

    @staticmethod
    def fn(a, y):
        """此处计算的并不是真正的代价函数，而是某个训练样本下由所有输出共同求得的单个训练样本的代价函数。
        真正需要算代价函数值的时候还需要对所有小批量块中的所有训练样本累加求平均。
        a是实际的输出向量，y是训练样本中的正确输出。因此一下是向量化运算。
        """
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        """delta并非仅仅是代价函数关于输出的导数，而是用在计算输出误差中的（代价函数关于输出的导数）与（激活函数关于最后一层z的导数)的乘积
        并且要注意这里是向量化运算。a是实际的输出向量，y是训练样本中的正确输出，z是输出层的带权输入z向量。
        """
        return (a-y) * sigmoid_prime(z)


class CrossEntropyCost:

    @staticmethod
    def fn(a, y):
        """此处计算的并不是真正的代价函数，而是某个训练样本下由所有输出共同求得的单个训练样本的代价函数。
        真正需要算代价函数值的时候还需要对所有小批量块中的所有训练样本累加求平均
        a是实际的输出向量，y是训练样本中的正确输出，因此一下是向量化运算。
        -y*np.log(a)-(1-y)*np.log(1-a)就是向量化的运算，得出的结果是一个一维数组
        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
    # -y*np.log(a)-(1-y)*np.log(1-a)得到的数组采用nan_to_num函数快速转化其中的NaN和inf元素，为0.0和一个很大的有限数
    # 使用nan_to_num转化完的数组，进行sum运算完成每个元素的相加求和，最后得到一个所有元素和的数值

    @staticmethod
    def delta(z, a, y):
        """delta并非仅仅是代价函数关于输出的导数，而是用在计算输出误差中的（代价函数关于输出的导数）与（激活函数关于最后一层z的导数)的乘积
        并且要注意这里是向量化运算。a是实际的输出向量，y是训练样本中的正确输出，z是输出层的带权输入z向量，但是由于交叉熵与代价函数关于z的导数相消。
        sigmoid_prime(z)这一项被消掉了，因此z并不出现在公式中。
        """
        return (a-y)


# main_class
class Network:

    def __init__(self, sizes, cost=CrossEntropyCost):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost = cost
    # 注意：self.cost是一个类名而非一个对象，一下调用CrossEntropyCost类中的方法都是通过类名调用的。
    # 这也是为什么要把CrossEntropyCost中的方法都是设为方法的原因了

    def default_weight_initializer(self):
        """
        默认的初始化方式：适用于sigmoid的初始化，w参数采用均值为0，标准差为1/np.sqrt(x)
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        """
        普通的标准正态分布参数初始化
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        """完全前馈网络，向此函数传递输入值，输出最后一层的实际输出"""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda = 0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):
        """
        使用小批量快的方法训练。训练集可以使任意数量，即``training_data``可以使任意长度的列表，但至少要大于批量大小。
        training_data这个列表的每个元素应该是元组``(x, y)``，(x, y)构成某个具体的训练样本.
        x是训练样本的输入向量（行数与网络输入个数保持一致），y是训练样本的输出向量（行数与网络输出个数保持一致）。
        其中x，y都是ndarray类型的一维数组来表示向量
        epochs是迭代器次数，mini_batch_size小批量块大小，eta学习速率，lmbda是L2规范化的系数。
        其他的参数是用来选择是否增加某些参数的计算的以监视运行过程。整个方法返回一个元组，元组含有
        evaluation_cost, evaluation_accuracy, training_cost, training_accuracy这四个列表。
        比如training_cost会记录着所有epoch中训练结果对于训练集的代价值。
        lmdba会随着训练集个数而改变，根据书中测试50000个训练集的时候lmdba取5比较好，保持这个比例，10000个训练集lmdba取1。
        但是可以通过调lmdba来改变防过拟合的程度，ladbm越大越会选接近于0的参数。
        在SGD的适用中无论是训练集还是测试集都可以是任意长度，通过切片来设置。

        """
        if evaluation_data:
            n_data = len(evaluation_data)
        n = len(training_data)  # 主程序可以设置任何长度的训练集
        # 在SGD的适用中无论是训练集还是测试集都可以是任意长度，通过切片来设置。
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]  # 生成每个迭代器的所有小批量块
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, len(training_data))
                # 相较于原来增加了lmbda和len(training_data))这两项参数
                # lmbda和len(training_data))都是用于L2规范化的计算中的，二者相除作为L2规范化的参数的系数
            print("Epoch %s training complete" % j)  # 按迭代期进行参数监控
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print("Accuracy on training data: {} / {}".format(accuracy, n))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on evaluation data: {} / {}".format(self.accuracy(evaluation_data), n_data))
            print('******')
        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """
        更新参数，求出此小批量的所有训练样本的参数梯度，将这些训练样本进行累加，得到总的参数梯度。
        将迭代玩的总梯度带入L2规范化的参数更新公式，更新参数。

        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]  # 完成对各参数在每个此小批量块的每个训练样本下的梯度累加
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]
        # 更新公式和原来是一样的，但是加入了推导之后的L2规范化公式

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]  # 建立空的梯度数据
        # 前向传递。同时存储所有层的z和a向量值
        activation = x
        activations = [x]  # 这是一个列表，列表中每个元素都是一维数组，该一维数组存储着每一层的输出a向量
        zs = []  # 这是一个列表，列表中每个元素都是一维数组，该一维数组存储着每一层的输入z向量
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # 反向传播  delta为误差列表，每个元素为每一层的误差向量。nabla_w为权重参数的梯度矩阵，每个元素为每一层的权重的梯度矩阵
        delta = self.cost.delta(zs[-1], activations[-1], y)  # 计算输出误差，这是矩阵运算
        # 注意这里的self.cost.delta函数算出的值已经是代价函数关于输出的导数与激活函数导数的乘积了，也就是self.cost.delta得出的已经是输出误差了。
        # 下面只需要让delta和上一层输出值相乘就能得到最后一层的梯度向量了
        nabla_b[-1] = delta  # 计算最外层偏置关于代价函数的偏导值（它就等于该层的误差值）
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())  # 计算最外层权重关于代价函数的偏导值（它就等于该层的误差值乘上一层输出值想来那个的转置）
        # 这里注意迭代次数，range从2开始到总层数的，从而实现从倒数第二层一直到第一层。因为索引为负的时候，是从-开始的而非像索引为正的时候从开始
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp  # 用后一层权重矩阵来计算误差
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())  # 用前一层的输出来计算梯度，因此计算某一层的梯度实际上要考虑它前一层和后一层的
        return nabla_b, nabla_w

    def accuracy(self, data, convert=False):
        """
        会逐个计算data中所有的输入所对应的实际输出并与标准输出比较，统计二者相符的个数，作为准确度。
        返回值就是计算出的data中能够正确输出的数据个数。
        convert这个标志位是来区分data中给定的正确输出，是一个数值形式的还是一个由0,1组成的列表形式的.
        如果data给定的标准输出是10个数组成的列表形式，那么可以让convert为1，否则让它为0。
        这样可以根据要分析的data的类型灵活的选择，当对training_data能输出正确结果的个数的时候就用convert=True的形式，
        而data为test_data和validation_data，就需要让convert=False
        因为根据神经网络对数据格式的要求，训练集的输出就是一个由0,1组成的列表形式的，
        而test_data和validation_data中的输出都是保留的MNIST数据集的原始的一个数值形式的输出结果，
        为数字手写辨识量身定做。
        """
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y) for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    def total_cost(self, data, lmbda, convert=False):
        """
        这个函数是用来计算data数据集子啊当前参数下的总代价值。
        这个代价是考虑L2规范的总代价，其中原始代价是对每个数据样本的代价求平均值，在加上L2中的带系数参数部分。
        为数字手写辨识量身定做。
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert:
                y = vectorized_result(y)  # 将输出转化成一个由0,1组成的列表形式
            cost += self.cost.fn(a, y)/len(data)  # 计算data中每一个数据样本的cost值，并对所有数据样本的cpst值累加并求平均。这是代价函数的真正定义
        cost += 0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights)
        # 考虑L2规范化的情况下的总代价函数的计算，就是用上面的原始代价函数加上权重矩阵Frobenius范数的平方乘以一个系数。
        return cost

    def save(self, filename):
        """将当前神经网络模型和参数存入名叫filename的文件"""
        data = {"sizes": self.sizes, "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases], "cost": str(self.cost.__name__)}
        # 利用tolist()将w参数这个矩阵转换成列表，这样便于存储在字典中。最终"weights"将会是三维列表嵌套。"weights"的元素是每层的权重列表。每个权重列表又是由两层的列表嵌套组成。
        # 注意：data因为要转换为json格式，因此需要把array转换为多层列表再存储。
        # self.cost.__name__:self.cost是一个类，一个类的__name__属性就是这个类名。因此str(self.cost.__name__）就是‘CrossEntropyCost’
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

    def identification_one(self, x):
        """
        identification_one是用来对784个元素组成的一个灰度图像输入向量(一维数组)计算该图片辨识后的数字
        :param x: 784个元素组成的一个表示灰度图像的一维数组，类型是ndarray类型
        :return: 该图片辨识后的数字
        """
        return np.argmax(self.feedforward(x))

    def identification_data(self, data):
        """
        identification_data是将存着多个一维数组的列表中每个存着灰度图片信息的一维数组进行图片辨识最后得到每个数字以列表的形式返回
        :param data: 存着多个一维数组的列表。其中每个一维数组都是由784个元素组成，表示灰度图像的，类型是ndarray类型
        :return: 所有输入对应的辨识结果组成的列表
        """
        return [np.argmax(self.feedforward(x)) for x in data]

# load a network


def load(filename):
    """
    从某个文件中读取神经网络的结构（包括层数等信息），还可以读取参数信息。利用这些信息初始化一个Network类的对象net。
    net对象已经是包含有上次训练完的所有参数以及上次训练的神经网络结构等信息了。
    可以利用load返回的这个net对象，可以利用net对象本身的SGD方法继续训练，
    也可以直接使用net对象来对某个输入得到次神经网络计算的结果了，即进行真正的手写辨识。
    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    # 每个模块都有一个__name__属性，来表明这个模块的唯一名字。
    # getattr在这里表明要返回当前模块的以data["cost"]中内容为名的类。
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net



