from cProfile import label
from turtle import color

import matplotlib
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset

import os
import numpy as np
import matplotlib.pyplot as plt

from preprocessing import get_train_test_loader
# from augmentation import concat_get_train_test_loader
from load_asl import get_train_test_loader_asl
from networks import BaseNet, SmallNet


def loadNet(netname):
    if netname == 'BaseNet':
        net = BaseNet().float()
    if netname == 'SmallNet':
        net = SmallNet().float()

    return net


def main():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    matplotlib.use('TkAgg')
    print(matplotlib.rcParams['backend'])
    for name in ['BaseNet']:
        print(f"Running {name}")
        net = loadNet(name)
        print(net)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

        # trainloader, testloader = get_train_test_loader()
        # trainloader, testloader = concat_get_train_test_loader()

        trainloader, testloader, _, _ = get_train_test_loader_asl()

        train_losses = []
        train_accs = []

        test_losses = []
        test_accs = []

        epochs = 20

        for epoch in range(epochs):
            train_loss, train_acc = train(net, criterion, optimizer, trainloader, epoch)

            test_loss, test_acc = test(net, criterion, testloader)

            train_losses.append(train_loss)
            train_accs.append(train_acc)

            test_losses.append(test_loss)
            test_accs.append(test_acc)

        print("Plotting ")
        # test_plot([train_losses, test_losses], [train_accs, test_accs], epochs)
        plot_learning_curves([train_losses, test_losses], [train_accs, test_accs], epochs)

        torch.save(net.state_dict(), f"models/model_params_{name}_ASL.pth")

        # plot_learning_curves([train_losses, test_losses], [train_accs, test_accs], 5)


def test_plot(losses, accs, epochs):
    x = np.linspace(0, epochs, epochs)
    plt.plot(x, losses[0], label='Training Loss')
    plt.plot(x, losses[1], label='Validation Loss')
    plt.legend()
    plt.show()
    plt.plot(x, accs[0], label='Training accuracy')
    plt.plot(x, accs[1], label='Validation Accuracy')
    plt.legend()
    plt.show()


def plot_learning_curves(losses, accs, epochs):
    fig, axes = plt.subplots(2)
    fig.suptitle('Learning curves')

    x = np.linspace(0, epochs, epochs)

    axes[0].plot(x, losses[0], color='blue', label='Training Loss')
    axes[0].plot(x, losses[1], color='orange', label='Validation Loss')
    axes[0].legend()
    axes[0].set_xlabel('epochs')
    axes[0].set_ylabel('loss')

    axes[1].plot(x, accs[0], color='blue', label='Training accuracy')
    axes[1].plot(x, accs[1], color='orange', label='Validation Accuracy')
    axes[1].legend()
    axes[1].set_xlabel('epochs')
    axes[1].set_ylabel('accuracy')

    plt.show()


def train(net, criterion, optimizer, trainloader, epoch):
    running_loss = 0.0
    total = 0
    correct = 0
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

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels[:, 0]).sum().item()

    return running_loss / len(trainloader), 100 * correct / total


def test(net, criterion, test_loader):
    print("Testing")
    running_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for data in test_loader:
            inputs = Variable(data['image'].float())
            labels = Variable(data['label'].long())

            # forward pass
            outputs = net(inputs)
            loss = criterion(outputs, labels[:, 0])

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels[:, 0]).sum().item()

    print("Testing done")
    return running_loss / len(test_loader), 100 * correct / total


if __name__ == '__main__':
    main()
