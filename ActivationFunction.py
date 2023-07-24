# 激活函数
# 2023.7.21
import torch
from d2l import torch as d2l


# ReLU(x) = max(x;0)

def ReLU():
    x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
    y = torch.relu(x)
    d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))
    y.backward(torch.ones_like(x), retain_graph=True)
    d2l.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
# def sigmoid():

if __name__=="__main__":
    ReLU()