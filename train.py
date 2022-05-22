import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset

from preprocessing import get_train_test_loader
from networks import BaseNet, SmallNet, LargeNet


def loadNet(netname):
    if netname == 'BaseNet':
        net = BaseNet().float()
    if netname == 'SmallNet':
        net = SmallNet().float()
    if netname == 'LargeNet':
        net = LargeNet().float()

    return net


def main():
    for name in ['SmallNet']:
        print(f"Running {name}")
        net = loadNet(name)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

        trainloader, _ = get_train_test_loader()
        for epoch in range(20):
            train(net, criterion, optimizer, trainloader, epoch)
        torch.save(net.state_dict(), f"models/model_params_{name}2.pth")


def train(net, criterion, optimizer, trainloader, epoch):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs = Variable(data['image'].float())
        labels = Variable(data['label'].long())
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels[:, 0])
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 0:
            print(f"[{epoch},{i}] loss: {running_loss / (i + 1)}")


if __name__ == '__main__':
    main()
