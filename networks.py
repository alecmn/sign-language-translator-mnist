from torch import nn
import torch.nn.functional as F


class BaseNet(nn.Module):
    """
    Inspired by https://ieeexplore.ieee.org/document/8441304
    """
    def __init__(self):
        super(BaseNet, self).__init__()
        self.c1 = nn.Conv2d(1, 6, 3)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.c2 = nn.Conv2d(6, 6, 3)
        self.c3 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 48)
        self.fc3 = nn.Linear(48, 24)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = self.max_pool(x)
        x = F.relu(self.c3(x))
        x = self.pool(x)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SmallNet(nn.Module):
    """
    Inspired by https://link.springer.com/article/10.1007/s00521-019-04691-y
    """
    def __init__(self):
        super(SmallNet, self).__init__()
        self.c1 = nn.Conv2d(1, 16, 3)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(16 * 13 * 13, 128)
        self.fc2 = nn.Linear(128, 24)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = self.max_pool(x)
        x = x.view(-1, 16 * 13 * 13)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
