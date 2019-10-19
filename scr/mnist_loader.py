"""
mnist_训练集数据加载
~~~~~~~~~~~~

记载MNIST数据集的模块. ``load_data``和 ``load_data_wrapper``字符串中有关于这个数据集书记结构的详细信息。
 实际上，`load_data_wrapper``是被神经网络经常调用的函数，
"""


# Standard library
import pickle
import gzip
# cPickle可以对任意一种类型的python对象进行序列化操作


# Third-party libraries
import numpy as np
def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    MNIST中的原始数据格式：一个有两个元素的元组。其中第一个元素是一个ndarray类型的二维数组，
    该二维数组最外层有50000个元素，每个元素有是一个有784个元素的一维数组，也就是这是一个50000行784列的矩阵代表输入集。
    第二个元素同理，也是一个ndarray类型的一维数组，第一维50000个元素，每个元素都是一个0-9之间的数字。
    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f,encoding='bytes')
    f.close()
    return (training_data, validation_data, test_data)


def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.

    神经网络使用的数据格式：是一个列表，列表有50000个元素。
    每个元素是一个元组。每个元组有2个元素(x, y)。x是一个有784个元素的一维数组，
    其中存储着784个像素信息。y是一个有10个元素的一维数组，相当于用十个数表示0-9这10个数字，哪一位是1就是表示的几。
    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code.这个函数是用来把MNIST中的数据转化成接下来神经网路要用的形式的"""
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    # 把原始数据中50000个表示像素信息的一维数组，逐个提取出来然后到training_inputs列表中。
    # 并将提取出来的含784个元素的一维数组转化成，784行1列的二维数组。然后50000个这种二维数组组成了列表training_inputs
    # 其实本质顺序没什么变化就是把一维数组每个元素数组化了，不知道为什么这么做
    training_results = [vectorized_result(y) for y in tr_d[1]]
    # 首先此处把0-9转化成十个数来表示0-9，然后同样把这十个数数转化成了一个10行1列的二维数组,然后50000个这种二维数组组成了列表training_results
    training_data = list(zip(training_inputs, training_results))
    # 把前面两个有50000个元素的列表组合起来，成为一个含有50000个元组的列表（每一个元组有两个元素，分别对应着一组x，y）。
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return (training_data, validation_data, test_data)


def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def a():
    pass