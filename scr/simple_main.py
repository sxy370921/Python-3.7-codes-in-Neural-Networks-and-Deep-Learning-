# network_simple.py 通过测试来看仅适用于浅层(一个隐藏层)

import scr.mnist_loader as data
import scr.network_simple as nt


# shallow

training_data, validation_data, test_data = data.load_data_wrapper()
net = nt.Network([784, 100, 10])
net.SGD(training_data, 60, 10, 3.0, test_data=test_data)

# deep
#
# training_data, validation_data, test_data = data.load_data_wrapper()
# net = nt.Network([784, 60, 60, 60, 60, 60, 60, 10])
# net.SGD(training_data, 40, 10, 3.0, test_data=test_data)