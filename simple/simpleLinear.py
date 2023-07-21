# 简洁的线性回归实现-使用Pytorch的api
# 2023.7.20
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn


def load_array(data_arrays, batch_size, is_train=True):  # @save
    """构造⼀个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)  # 生成数据集

batch_size = 10
data_iter = load_array((features, labels), batch_size)  # 获得数据迭代器


net = nn.Sequential(nn.Linear(2, 1))  # 两个输入对应一个输出，
# Sequential类为串联在一起的多个层定义了一个容器。当给定输入数据，
# Sequential实例将数据传入到第一层，然后将第一层的输出作为第二层的输入，依此类推
net[0].weight.data.normal_(0, 0.01)  # # 权重参数从均值为0，标准差为0.01的正态分布中随机采样
net[0].bias.data.fill_(0)  # 偏置参数初始化为0
loss = nn.MSELoss()  # 均方误差损失函数
trainer = torch.optim.SGD(net.parameters(), lr=0.03)  # 定义优化算法——梯度下降优化算法
num_epochs = 3  # 迭代三个周期
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)  # loss是损失函数

        trainer.zero_grad()  # trainer优化器，先把梯度清零
        l.backward()  # 等价于l.sum().backward()——求和之后算梯度
        trainer.step()  # 调用优化算法进行模型更新

    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
