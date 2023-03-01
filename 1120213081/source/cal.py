import torch

def yTransformToTensor():
    VYPATH = '../../../data/train.txt'
    ys = open(VYPATH, 'r')
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


def xTransformToTensor():
    VXPATH = "./evFrequentWords.txt"
    xs = open(VXPATH, 'r')
    x_list = []
    while True:
        x = xs.readline()
        if x == '\n':
            continue
        if x != "":
            x = x.split()
            numList = list(map(int, x))
            x_list.append(numList)
        else:
            break
    return torch.Tensor(x_list)


# 引入GPU
# print(torch.cuda.is_available())
GPU = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Matrix_x = xTransformToTensor().to(GPU)
# Matrix_y = yTransformToTensor().to(GPU)
# print(Matrix_x.shape)
# print(Matrix_y.shape)

# theta 参数是一个列向量
Matrix_theta = torch.zeros((5000, 1)).to(GPU)
print(Matrix_theta)