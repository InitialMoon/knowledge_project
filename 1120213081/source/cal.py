# -*- coding: utf-8 -*-

import torch

import random

import matplotlib.pyplot as plt

import time

TRAIN_Y_PATH = '../../../data/train.txt'
TRAIN_X_PATH = './train_x.txt'
VALID_Y_PATH = '../../../data/validation.txt'
VALID_X_PATH = './valid_x.txt'
TEST_Y_PATH = '../../../data/test.txt'
TEST_X_PATH = './test_x.txt'
STEP = 1 / 1000

# 引入GPU
GPU = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# input: y向量的数据来源
# output: y1和y2的两个张量,在CPU上
def y_transform_to_tensor(path):
    ys = open(path, 'r')
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


# input: x向量的数据来源
# output: x的一个矩阵，是所有样本的集合，行是样本数量，列是5001维
def x_transform_to_tensor(path):
    xs = open(path, 'r')
    x_list_transform = []
    i = 0
    while True:
        x_list_transform.append([])
        x = xs.readline()
        if x == '\n':
            continue
        if x != "":
            x = x.split()
            num_list = list(map(int, x))
            x_list_transform[i] = num_list
            i = i + 1
        else:
            break
    return x_list_transform


# sigmoid function
def func_g(x):
    return 1 / (1 + torch.exp(-x))


# input : x is a 2D list, y is a 1D tensor on CPU, step_rate is the step length, is_tensor_cal decide use for or matrix
# output : loss_function list, which record the value of the value by iterator
#          theta_list, which record the theta by iterator, every model is in this
def batch_gradient_hand(x, y, step_rate, iterator_times):
    loss = []
    theta_list = []
    tensor_x = torch.as_tensor(x, dtype=torch.float).to(GPU)
    tensor_y = torch.as_tensor(y, dtype=torch.float).to(GPU)
    theta = torch.zeros(5001).to(GPU)
    for j in range(iterator_times):
        grad = (torch.mv(tensor_x.T, (tensor_y - func_g(torch.mv(tensor_x, theta)))))
        alpha = random.random() * step_rate
        theta = theta + alpha * grad
        theta_list.append(theta)
        log_l_theta = (tensor_y * torch.log10(func_g(torch.mv(tensor_x, theta))) - (tensor_y - 1) * torch.log10(
            (1 - func_g(torch.mv(tensor_x, theta))))).sum()

        loss.append(float(log_l_theta))
        theta_list.append(theta)

    return loss, theta_list


# input : x is a 2D list, y is a 1D tensor on CPU, step_rate is the step length, is_tensor_cal decide use for or matrix
# output : loss_function list, which record the value of the value by iterator
#          theta_list, which record the theta by iterator, every model is in this
def random_gradient_hand(x, y, step_rate, iterator_times, every_num):
    loss = []
    theta_list = []
    tensor_x = torch.as_tensor(x, dtype=torch.float).to(GPU)
    tensor_y = torch.as_tensor(y, dtype=torch.float).to(GPU)
    theta = torch.zeros(5001).to(GPU)
    for j in range(iterator_times):
        index = torch.LongTensor(random.sample(range(tensor_y.shape[0]), every_num)).to(GPU)
        r_tensor_x = torch.index_select(tensor_x, 0, index)
        r_tensor_y = torch.index_select(tensor_y, 0, index)
        grad = (torch.mv(r_tensor_x.T, (r_tensor_y - func_g(torch.mv(r_tensor_x, theta)))))
        alpha = random.random() * step_rate
        theta = theta + alpha * grad
        theta_list.append(theta)
        log_l_theta = (r_tensor_y * torch.log10(func_g(torch.mv(r_tensor_x, theta))) - (r_tensor_y - 1) * torch.log10(
            (1 - func_g(torch.mv(r_tensor_x, theta))))).sum()

        loss.append(float(log_l_theta))
        theta_list.append(theta)

    return loss, theta_list


# input : x is a 2D list, y is a 1D tensor on CPU, step_rate is the step length, is_tensor_cal decide use for or matrix
# output : loss_function list, which record the value of the value by iterator
#          theta_list, which record the theta by iterator, every model is in this
def batch_gradient_auto(x, y, step_rate, iterator_times, is_tensor_cal):
    loss = []
    theta_list = []
    # 注意这里要是不追踪放置到gpu上的张量的话，自动求导的时候就无法追踪到，因为子叶结点已经更换了
    theta_cpu = torch.zeros(5001, dtype=torch.float, requires_grad=True)
    theta_gpu = theta_cpu.to(GPU)
    tensor_x = torch.as_tensor(x, dtype=torch.float).to(GPU)
    tensor_y = torch.as_tensor(y, dtype=torch.float).to(GPU)
    if is_tensor_cal:
        # 利用矩阵进行所有样本的计算
        for j in range(iterator_times):
            log_l_theta = (tensor_y * torch.log10(func_g(torch.mv(tensor_x, theta_gpu))) - (tensor_y - 1) * torch.log10(
                (1 - func_g(torch.mv(tensor_x, theta_gpu))))).sum()
            log_l_theta.backward()
            loss.append(float(log_l_theta))
            # 随机生成的步长
            alpha = step_rate * random.random()
            # 这里又变成cpu了，注意
            grad = theta_cpu.grad
            # 参与运算的子叶结点不接受数据更改，虽然已经正确算出来了，但是不能直接用,或许我们能现将子叶的跟踪停止下来，因为，再下一次计算与这次没
            # 有任何关系了，先把requires关闭了试试
            # print(theta_gpu.is_leaf)
            # print(theta_cpu.is_leaf)
            theta_cpu.requires_grad_(False)
            theta_cpu += alpha * grad
            theta_cpu.requires_grad_(True)
            theta_gpu = theta_cpu.to(GPU)
            theta_cpu.grad.zero_()
            theta_list.append(theta_gpu)
            # print(j)
            # print(theta_gpu)
            # print(log_l_theta)
    else:
        sample_num = tensor_x.shape[0]
        # 利用for循环进行所有样本的计算
        for j in range(iterator_times):
            log_l_theta = 0.0
            for i in range(sample_num):
                vxi = torch.tensor(x[i], dtype=torch.float).to(GPU)
                yi = float(y[i])
                a = func_g(torch.dot(theta_gpu, vxi))
                log_l_theta += (yi * torch.log10(a) + (1 - yi) * torch.log10(1 - a))

            # 每次下降的梯度
            # grad_1 = torch.autograd.grad(log_l_theta, theta_gpu, create_graph=True)[0]
            log_l_theta.backward()
            grad = theta_cpu.grad
            loss.append(float(log_l_theta))
            alpha = step_rate * random.random() / 1000

            theta_cpu.requires_grad_(False)
            theta_cpu += alpha * grad
            theta_cpu.requires_grad_(True)
            theta_gpu = theta_cpu.to(GPU)
            theta_cpu.grad.zero_()
            theta_list.append(theta_gpu)

    return loss, theta_list


def random_gradient_auto(x, y, step_rate, iterator_times, every_num, is_tensor_cal):
    loss = []
    theta_list = []
    # 注意这里要是不追踪放置到gpu上的张量的话，自动求导的时候就无法追踪到，因为子叶结点已经更换了
    theta_cpu = torch.zeros(5001, dtype=torch.float, requires_grad=True)
    theta_gpu = theta_cpu.to(GPU)
    tensor_x = torch.as_tensor(x, dtype=torch.float).to(GPU)
    tensor_y = torch.as_tensor(y, dtype=torch.float).to(GPU)
    if is_tensor_cal:
        # 利用矩阵进行所有样本的计算
        for j in range(iterator_times):
            index = torch.LongTensor(random.sample(range(tensor_y.shape[0]), every_num)).to(GPU)
            r_tensor_x = torch.index_select(tensor_x, 0, index)
            r_tensor_y = torch.index_select(tensor_y, 0, index)
            log_l_theta = (r_tensor_y * torch.log10(func_g(torch.mv(r_tensor_x, theta_gpu))) - (r_tensor_y - 1) * torch.log10(
                (1 - func_g(torch.mv(r_tensor_x, theta_gpu))))).sum()
            log_l_theta.backward()
            loss.append(float(log_l_theta))
            # 随机生成的步长
            alpha = step_rate * random.random()
            # 这里又变成cpu了，注意
            grad = theta_cpu.grad
            # 参与运算的子叶结点不接受数据更改，虽然已经正确算出来了，但是不能直接用,或许我们能现将子叶的跟踪停止下来，因为，再下一次计算与这次没
            # 有任何关系了，先把requires关闭了试试
            # print(theta_gpu.is_leaf)
            # print(theta_cpu.is_leaf)
            theta_cpu.requires_grad_(False)
            theta_cpu += alpha * grad
            theta_cpu.requires_grad_(True)
            theta_gpu = theta_cpu.to(GPU)
            theta_cpu.grad.zero_()
            theta_list.append(theta_gpu)
            # print(j)
            # print(theta_gpu)
            # print(log_l_theta)
    else:
        sample_num = tensor_x.shape[0]
        # 利用for循环进行所有样本的计算
        for j in range(iterator_times):
            log_l_theta = 0.0
            for i in range(sample_num):
                vxi = torch.tensor(x[i], dtype=torch.float).to(GPU)
                yi = float(y[i])
                a = func_g(torch.dot(theta_gpu, vxi))
                log_l_theta += (yi * torch.log10(a) + (1 - yi) * torch.log10(1 - a))

            # 每次下降的梯度
            # grad_1 = torch.autograd.grad(log_l_theta, theta_gpu, create_graph=True)[0]
            log_l_theta.backward()
            grad = theta_cpu.grad
            loss.append(float(log_l_theta))
            alpha = step_rate * random.random() / 1000

            theta_cpu.requires_grad_(False)
            theta_cpu += alpha * grad
            theta_cpu.requires_grad_(True)
            theta_gpu = theta_cpu.to(GPU)
            theta_cpu.grad.zero_()
            theta_list.append(theta_gpu)

    return loss, theta_list


# input: validation/test x, validation/test y, iterator model_list
# output: validation_loss : with the iterator of model, change of loss value, back as a list
#         min_loss
def calculate_loss(x, y, model_list):
    validation_loss = []
    tensor_x = torch.as_tensor(x, dtype=torch.float).to(GPU)
    tensor_y = torch.as_tensor(y, dtype=torch.float).to(GPU)
    for model in model_list:
        # print(type(model))
        model.to(GPU)
        # print(model)
        log_l_theta = (tensor_y * torch.log10(func_g(torch.mv(tensor_x, model))) - (tensor_y - 1) * torch.log10(
            (1 - func_g(torch.mv(tensor_x, model))))).sum()
        validation_loss.append(float(log_l_theta))

    min_loss = max(validation_loss)
    i = 0
    min_i = 0
    min_model = model_list[0]
    for loss in validation_loss:
        if loss == min_loss:
            min_i = i
            min_model = model_list[i]
        i += 1

    return validation_loss, min_loss, min_model, min_i


# input : model_list, which is the trained model, used to calculate the F1_measure list
#         test_x is a Matrix, test_y is a list as we cal above
def f1_measure(model_list, test_x, test_y):
    f1m = []
    recall_list = []
    precision_list = []
    tensor_x = torch.as_tensor(test_x, dtype=torch.float).to(GPU)
    tensor_y = torch.as_tensor(test_y, dtype=torch.float).to(GPU)
    # 定义用于条件判断要赋给的原始数据
    # one_tensor = torch.ones(tensor_y.shape[0]).to(GPU)
    # zero_tensor = torch.zeros(tensor_y.shape[0]).to(GPU)

    for model in model_list:
        sigmoid = func_g(torch.mv(tensor_x, model))
        # 如果概率超过0.5就记作发生了
        # 对结果进行统计
        # 查准率 = 正确预测数 / 模型预测数
        # 查全率 = 正确预测数 / 实体总数

        # 模型预测数
        model_forecast_num = 0
        model_forecast_list = sigmoid.tolist()
        # print(model_forecast_list)
        # 正确预测数
        right_num = 0
        # 真正发生总数
        real_num = 0
        real_list = tensor_y.tolist()
        # print(real_list)

        for i in range(len(model_forecast_list)):
            if model_forecast_list[i] > 0.5:
                model_forecast_num += 1
            if real_list[i] > 0.5:
                real_num += 1
            if real_list[i] > 0.5 and model_forecast_list[i] > 0.5:
                right_num += 1

        print("模型预测数：" + str(model_forecast_num), end=' ')
        print("正确预测数：" + str(right_num), end=' ')
        print("真正发生数：" + str(real_num))
        if real_num * model_forecast_num != 0:
            recall = float(right_num) / real_num
            precision = float(right_num) / model_forecast_num
            if recall + precision != 0:
                f1 = 2 * (recall * precision) / (recall + precision)
                f1m.append(f1)
                recall_list.append(recall)
                precision_list.append(precision)
                print("查全率：" + str(recall))
                print("查准率：" + str(precision))
                print("F1-measure = " + str(f1))

    return recall_list, precision_list, f1m


# input : a value list
# output : a graph
def draw_single_line(name, y, xlab, ylab, step, min_pos):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    axis_x = list(range(0, len(y), 1))
    plt.plot(axis_x, y, label=name)
    plt.vlines([min_pos], 0, 1, linestyles='dashed', colors='red')
    plt.legend(loc="lower right")
    plt.xlabel(xlab)  # x轴坐标名称及字体样式
    plt.ylabel(ylab)  # y轴坐标名称及字体样式
    plt.text(0, 1, "步长是" + str(step), fontsize=10)  # 在图中添加文本
    plt.show()


# input : 2 value lists
# output : a graph
def draw_lines(name1, y1, name2, y2, name3, y3, xlab, ylab, step, min_pos):
    axis_x = list(range(0, len(y1), 1))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.plot(axis_x, y1, label=name1)
    plt.plot(axis_x, y2, label=name2)
    plt.plot(axis_x, y3, label=name3)
    plt.vlines([min_pos], 0, 1, linestyles='dashed', colors='red')
    plt.legend(loc="lower right")
    plt.text(0, 1, "步长是" + str(step), fontsize=10)  # 在图中添加文本
    plt.xlabel(xlab)  # x轴坐标名称及字体样式
    plt.ylabel(ylab)  # y轴坐标名称及字体样式
    plt.show()


# 导入所有的数据
# -1是因为最后一行是空的直接用会出问题
train_x_list = x_transform_to_tensor(TRAIN_X_PATH)[:-1]
valid_x_list = x_transform_to_tensor(VALID_X_PATH)[:-1]
test_x_list = x_transform_to_tensor(TEST_X_PATH)[:-1]

# 第一列是接见会见
# 第二列是考察检查
train_y1_list, train_y2_list = y_transform_to_tensor(TRAIN_Y_PATH)
valid_y1_list, valid_y2_list = y_transform_to_tensor(VALID_Y_PATH)
test_y1_list, test_y2_list = y_transform_to_tensor(TEST_Y_PATH)

T1 = time.time()

# loss1_list1, model11 = batch_gradient_hand(train_x_list, train_y1_list, STEP, 2000)
# loss1_list1, model11 = random_gradient_hand(train_x_list, train_y1_list, STEP, 60000, 100)
loss1_list1, model11 = batch_gradient_auto(train_x_list, train_y1_list, STEP, 2000, True)
T2 = time.time()

# recall, pre, f1m = f1_measure(model11, train_x_list, train_y1_list)
loss_list, min_loss, min_model, pos = calculate_loss(test_x_list, test_y1_list, model11)
recall, pre, f1m = f1_measure(model11, test_x_list, test_y1_list)

print('使用张量优化运算运行时间:%s毫秒' % ((T2 - T1) * 1000))

draw_lines("查全率", recall, "准确率", pre, "F1", f1m, "迭代次数", "比率(%)", STEP, pos)
draw_single_line("F1", f1m, "迭代次数", "比率(%)", STEP, pos)
draw_single_line("准确率", pre, "迭代次数", "比率(%)", STEP, pos)
draw_single_line("查全率", recall, "迭代次数", "比率(%)", STEP, pos)
# T1 = time.time()
#
# loss1_list, model1 = batch_gradient_auto(train_x_list, train_y1_list, 1, 1000, False)
#
# T2 = time.time()
# print('不使用张量优化运算运行时间:%s毫秒' % ((T2 - T1)*1000))
