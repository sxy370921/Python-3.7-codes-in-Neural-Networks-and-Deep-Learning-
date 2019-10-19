
# network_simple_up1.py

import scr.network_simple_up1 as nt2
import scr.mnist_loader as data
import copy


# shallow
# lmdba会随着训练集个数而改变，根据书中测试50000个训练集的时候lmdba取5比较好，保持这个比例，10000个训练集lmdba取1。
# 但是可以通过调lmdba来改变防过拟合的程度，ladbm越大越会选接近于0的参数。

# 训练过程：
training_data, validation_data, test_data = data.load_data_wrapper()
net = nt2.Network([784, 100, 10])
net.SGD(training_data, 60, 10, 0.1, lmbda=5.0, evaluation_data=test_data,
        monitor_evaluation_cost=True, monitor_evaluation_accuracy=True,
        monitor_training_accuracy=True, monitor_training_cost=True)

net.save('111.json')  # 利用save来存储训练完成的参数到111.json文件，可以用于继续训练，也可以直接使用来在某个手写图片上来识别结果。


# 利用load存的网络数据辨识实际手写辨识的过程模拟：
exist_net = nt2.load('111.json')  # 从111.json文件中读取上次训练的参数和网络结构，返回一个Network类的对象
# training_data, validation_data, test_data = data.load_data_wrapper()
data = copy.deepcopy(test_data)

#
# # 单个数据辨识
x = data[0][0]
print('辨识结果：', exist_net.identification_one(x), '正确结果：', (data[0][1]))

#
# # 输入数据块辨识
x_data = []
y_data = []
for i in data[:50]:
    x_data.append(i[0])
    y_data.append(i[1])
print('辨识结果：')
print(exist_net.identification_data(x_data))
print('正确结果：')
print(y_data)
# deep效果很差几乎烂掉了，所以删了

