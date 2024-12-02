# Project:PyTorch deep learning practice
# FileName:LinearModel
# Time:2023-07-14
import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


# 1.穷举法求解线性模型
class ExhaustiveMethod:
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def forward(self, x, w):
        return x * w

    def loss(self, x, w, y):
        y_pred = self.forward(x, w)
        return (y_pred - y) * (y_pred - y)

    def algorithm(self):
        w_list = []
        loss_list = []
        for w in np.arange(0, 4.1, 0.1):
            l_sum = 0
            for x_val, y_val in zip(self.x_data, self.y_data):
                loss = self.loss(x_val, w, y_val)
                l_sum += loss
            w_list.append(w)
            loss_list.append(l_sum / 3)
        return w_list, loss_list

    def show(self):
        w_list, exhaustive = ExhaustiveMethod(x_data, y_data)
        exhaustive.algorithm()
        exhaustive.show()
        loss_list = self.algorithm()
        min_id = np.argmin(loss_list)
        w_min, loss_min = w_list[min_id], loss_list[min_id]
        plt.scatter(w_min, loss_min, color='red', s=50)
        plt.annotate(f'min:{w_min:.2f},{loss_min:.2f}', xy=(w_min, loss_min), xytext=(w_min - 0.4, loss_min + 1))
        plt.plot(w_list, loss_list)
        plt.show()


# 2.(随机)梯度学习下降算法
class Gradient_Descent_algorithm():
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def forward(self, w, x):
        return w * x

    def cost(self, w):
        cost = 0
        n = len(self.x_data)
        for x_val, y_val in zip(self.x_data, self.y_data):
            cost += (w * x_val - y_val) * (w * x_val - y_val)
        return cost / n

    # 随机梯度学习计算loss
    def loss(self, w, x_val, y_val):
        return (w * x_val - y_val) * (w * x_val - y_val)

    def gradient(self, w):
        grad = 0
        n = len(x_data)
        for x_val, y_val in zip(self.x_data, self.y_data):
            grad += 2 * x_val * (w * x_val - y_val)
        return grad / n

    # 随机梯度计算
    def stochastic_gradient(self, w, x_val, y_val):
        return 2 * x_val * (w * x_val - y_val)

    def algorithm(self):
        w = 1.0
        cost_list = []
        w_list = []
        for epoch in range(100):
            cost_val = self.cost(w)
            cost_list.append(cost_val)
            w_list.append(w)
            w -= 0.01 * self.gradient(w)
        return cost_list, w_list

    # 随机梯度算法
    def stochastic_algorithm(self):
        w = 1.0
        w_list = []
        loss_list = []
        for epoch in range(100):
            for x_val, y_val in zip(self.x_data, self.y_data):  # 对每一个样本进行梯度的更新
                loss_val = self.loss(w, x_val, y_val)
                w_list.append(w)
                w -= 0.01 * self.stochastic_gradient(w, x_val, y_val)
            loss_list.append(loss_val)
        return loss_list, w_list

    def show(self):
        # cost_list, w_list = self.algorithm()
        loss_list, w_list = self.stochastic_algorithm()
        # plt.plot(range(100), cost_list)
        plt.plot(range(100), loss_list)
        plt.show()


# gradient = Gradient_Descent_algorithm(x_data, y_data)
# gradient.show()

# pytorch构建线性网络
import torch

x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])


class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


linearmodel = LinearModel()

criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(linearmodel.parameters(), lr=0.01)

for epoch in range(1000):
    y_pred = linearmodel.forward(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.data)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('w:', linearmodel.linear.weight.item())
print('b:', linearmodel.linear.bias.item())

x_test = torch.tensor([[4.0]])
y_test = linearmodel(x_test)
print(y_test.data)
