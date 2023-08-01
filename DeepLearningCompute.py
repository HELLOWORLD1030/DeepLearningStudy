# 深度学习计算
import torch
from torch import nn
from torch.nn import functional as F


def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5


net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
X = torch.rand(2, 20)
# print(X)
# print(net(X))
#
# print(type(net[2].bias))
# print(net[2].bias)
# print(net[2].bias.data)
#
# net.apply(init_normal)
# print(net[0].weight.data[0], net[0].bias.data[0])
net.apply(my_init)
print(net[0].weight[:2])
# print(*[(name, param.shape) for name, param in net[0].named_parameters()])
# print(*[(name, param.shape) for name, param in net.named_parameters()])

# class MLP(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.hidden = nn.Linear(20, 256)
#         self.out = nn.Linear(256, 10)
#
#     def forward(self, X):
#         # 注意，这⾥我们使⽤ReLU的函数版本，其在nn.functional模块中定义。
#         return self.out(F.relu(self.hidden(X)))
#
# class MySequential(nn.Module):
#     def __init__(self,*args):
#         super().__init__()
#         for idx, module in enumerate(args):
#             self._modules[str(idx)] = module
#
#     def forward(self, X):
#
#         # OrderedDict保证了按照成员添加的顺序遍历它们
#         for block in self._modules.values():
#             X = block(X)
#         return X
#
# class FixedHiddenMLP(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # 不计算梯度的随机权重参数。因此其在训练期间保持不变
#         self.rand_weight = torch.rand((20, 20), requires_grad=False)
#         self.linear = nn.Linear(20, 20)
#
#     def forward(self, X):
#         X = self.linear(X)
#
#         # 使⽤创建的常量参数以及relu和mm函数
#         X = F.relu(torch.mm(X, self.rand_weight) + 1)# torch.mm矩阵相乘
#         # 复⽤全连接层。这相当于两个全连接层共享参数
#         X = self.linear(X)
#         # 控制流
#         while X.abs().sum() > 1:
#             X /= 2
#         return X.sum()
#
#
# net = MLP()
# print(net(X))
# net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
# print(net(X))
# net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
# X = torch.rand(size=(2, 4))
# print(net(X))
# print(net[2].state_dict(),net[2].weight.grad)
