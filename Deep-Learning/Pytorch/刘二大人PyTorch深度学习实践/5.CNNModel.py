# Project:PyTorch deep learning practice
# FileName:5.CnnModel
# Time:2023-07-24
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import MNIST

batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
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


# 2.0构建模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = Conv2d(in_channels=1, out_channels=10, kernel_size=(5, 5))
        self.conv2 = Conv2d(in_channels=10, out_channels=20, kernel_size=(5, 5))
        self.pooling = nn.MaxPool2d(2)
        self.linear = nn.Linear(320, 10)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size, -1)
        x = self.linear(x)
        return x


model = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 3.损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    running_loss = 0.0
    for batchidx, data in enumerate(train_dataloader, 0):
        train_data, target = data
        train_data, target = train_data.to(device), target.to(device)
        y_pred = model(train_data)
        optimizer.zero_grad()

        loss = criterion(y_pred, target)
        loss.backward()
        optimizer.step()

        running_loss += loss
        if batchidx % 300 == 299:
            print('[%d,%5d] loss:%.3f' % (epoch + 1, batchidx + 1, running_loss / 300))
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy on test set: %d %%' % (100 * correct / total))


if __name__ == "__main__":
    for epoch in range(10):
        train(epoch)
        test()
