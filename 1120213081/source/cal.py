import math

import torch

import random

import matplotlib.pyplot as plt

# 引入GPU
GPU = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def y_transform_to_tensor():
    VYPATH = '../../../data/train.txt'
    ys = open(VYPATH, 'r')
    y1_list = []
    y2_list = []

    while True:
        y = ys.readline()
        if y != "":
            if "[0,0]" in y:
                y1_list.append(0)
                y2_list.append(0)
            if "[0,1]" in y:
                y1_list.append(0)
                y2_list.append(1)
            if "[1,0]" in y:
                y1_list.append(1)
                y2_list.append(0)
            if "[1,1]" in y:
                y1_list.append(1)
                y2_list.append(1)
        else:
            break

    return torch.Tensor(y1_list), torch.Tensor(y2_list)


def x_transform_to_tensor():
    VXPATH = "./evFrequentWords.txt"
    xs = open(VXPATH, 'r')
    x_list = []
    i = 0
    while True:
        x_list.append([])
        x = xs.readline()
        if x == '\n':
            continue
        if x != "":
            x = x.split()
            numList = list(map(int, x))
            x_list[i] = numList
            i = i + 1
        else:
            break
    return x_list


def valid_y_transform_to_tensor():
    VVYPATH = '../../../data/validation.txt'
    ys = open(VVYPATH, 'r')
    y_list = []

    while True:
        y = ys.readline()
        if y != "":
            if "[0,0]" in y:
                y_list.append([0, 0])
            if "[0,1]" in y:
                y_list.append([0, 1])
            if "[1,0]" in y:
                y_list.append([1, 0])
            if "[1,1]" in y:
                y_list.append([1, 1])
        else:
            break

    return torch.Tensor(y_list)


# sigmoid function
def func_g(x):
    return 1 / (1 + torch.exp(-x))


# input: x and y, x is a Matrix of input data, y is a vector of result
# output: parameter vector theta
def batch_gradient_hand(x, y, sample_num):
    theta = torch.zeros(5001).to(GPU)
    grad = 0.0
    for i in range(sample_num):
        a = torch.dot(theta.T,  x[(i + 1):(i + 2), :])
        print(a)
        b = y - func_g(a)
        print(b)
        c = b * x[(i + 1):(i + 2), :]
        print(c)
        grad = grad + (y - func_g(theta.T * x[(i + 1):(i + 2), :])) * x[(i + 1):(i + 2), :]

    alpha = random.random() / 10
    theta = theta + alpha * grad

    return theta


def batch_gradient_auto(x, y):
    # 注意这里要是不追踪放置到gpu上的张量的话，自动求导的时候就无法追踪到，因为子叶结点已经更换了
    theta_cpu = torch.zeros(5001, dtype=torch.float, requires_grad=True)
    theta_gpu = theta_cpu.to(GPU)
    loss = []
    tensor_x = torch.as_tensor(x, dtype=torch.float).to(GPU)
    tensor_y = torch.as_tensor(y, dtype=torch.float).to(GPU)
    for j in range(100):
        j = j + 1

        log_l_theta = (tensor_y * torch.log10(func_g(torch.mv(tensor_x, theta_gpu))) - (tensor_y - 1) * torch.log10((1 - func_g(torch.mv(tensor_x, theta_gpu))))).sum()
        # 对样本集所有的一次迭代计算
        # for i in range(sample_num):
        #     vxi = torch.tensor(x[i], dtype=torch.float).to(GPU)
        #     yi = float(y[i])
        #     a = func_g(torch.dot(theta, vxi))
        #     log_l_theta += (yi * torch.log10(a) + (1 - yi) * torch.log10(1 - a))

        log_l_theta.backward()
        loss.append(float(log_l_theta))
        # 每次下降的梯度
        # grad_1 = torch.autograd.grad(log_l_theta, theta, create_graph=True)[0]
        # print(grad_1)

        # 随机生成的步长
        alpha = random.random() / 50000
        # 这里又变成cpu了，注意
        grad = theta_cpu.grad
        # 参与运算的子叶结点不接受数据更改，虽然已经正确算出来了，但是不能直接用,或许我们能现将子叶的跟踪停止下来，因为，再下一次计算与这次没有任何关系了，先把requires关闭了试试
        # print(theta_gpu.is_leaf)
        # print(theta_cpu.is_leaf)
        theta_cpu.requires_grad_(False)
        theta_cpu += alpha * grad
        theta_cpu.requires_grad_(True)
        theta_gpu = theta_cpu.to(GPU)
        theta_cpu.grad.zero_()
        print(j)
        print(theta_gpu)
        print(log_l_theta)
        # 这个损失函数的值需要清零，否则会影响下一次计算
        # log_l_theta = 0.0

    return loss, theta_gpu


# print(torch.cuda.is_available())

# Matrix_x shape[样本数量，5001]
x_list = x_transform_to_tensor()
# print(type(x_list))
# Matrix_y shape[样本数量，2]
# 第一列是接见会见
# 第二列是考察检查
Matrix_y1, Matrix_y2 = y_transform_to_tensor()

sampleNum = Matrix_y1.shape[0]
print(sampleNum)

# loss1_list, model1 = batch_gradient_auto(x_list[:-1], Matrix_y1)
loss2_list, model2 = batch_gradient_auto(x_list[:-1], Matrix_y2)

print("model1: ")
# print(model1)
print("after 1000 iterator, the loss is :")
# print(loss1_list)
print("model2: ")
print(model2)
print("after 1000 iterator, the loss is :")
print(loss2_list)

axis_x = list(range(0, 100, 1))
plt.plot(axis_x, loss2_list)
#第3步：显示图形
plt.show()