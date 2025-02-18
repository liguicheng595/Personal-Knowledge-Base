# Project:PyTorch deep learning practice
# FileName:3.dataset_loader
# Time:2023-07-15
import os.path
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

abspath = os.path.abspath(__file__)
filepath = os.path.join(abspath, '../data/diabetes.csv.gz')


class DatasetsDataset(Dataset):
    def __init__(self, filepath1):
        super(DatasetsDataset, self).__init__()
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


if __name__ == '__main__':
    dataset = DatasetsDataset(filepath)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

    model = Model()

    criterion = torch.nn.BCELoss(size_average=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(100):
        for i, data in enumerate(train_loader, 0):
            inputs, lables = data
            y_pred = model(inputs)
            loss = criterion(y_pred, lables)
            print(epoch, i, loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
