# Project:PyTorch deep learning practice
# FileName:MultiClassModel
# Time:2023-07-23

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torch import optim

batch_size = 64
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                                ])
# 1.加载数据
# 训练数据
train_dataset = MNIST(root='./data/mnist',
                      train=True,
                      download=True,
                      transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# 测试数据
test_dataset = MNIST(root='./data/mnist',
                     train=False,
                     download=True,
                     transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


# 2.构建模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.L1 = torch.nn.Linear(784, 512)
        self.L2 = torch.nn.Linear(512, 256)
        self.L3 = torch.nn.Linear(256, 128)
        self.L4 = torch.nn.Linear(128, 64)
        self.L5 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.L1(x))
        x = F.relu(self.L2(x))
        x = F.relu(self.L3(x))
        x = F.relu(self.L4(x))
        return self.L5(x)


model = Model()

# 3.损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


# 4.定义训练和测试函数
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_dataloader, 0):
        inputs, label = data
        optimizer.zero_grad()
        y_pred = model(inputs)

        loss = criterion(y_pred, label)
        loss.backward()
        optimizer.step()

        running_loss += loss
        if batch_idx % 300 == 299:
            print('[%d,%5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy on test set: %d %%' % (100 * correct / total))

if __name__ == "__main__":
    for epoch in range(10):
        train(epoch)
        test()