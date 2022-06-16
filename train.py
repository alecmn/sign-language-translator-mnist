import matplotlib
import torch
from torch import nn, optim
from torch.autograd import Variable

import os
import numpy as np
import matplotlib.pyplot as plt

from preprocessing import get_train_test_loader
from augmentation import concat_get_train_test_loader
from load_asl import get_train_test_loader_asl
from networks import BaseNet, SmallNet


def loadNet(netname):
    """
    Init network
    :param netname: Network type to be instantiated
    :return: network
    """
    if netname == 'SmallNet':
        net = SmallNet().float()
    else:
        net = BaseNet().float()

    return net


def main():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    matplotlib.use('TkAgg')
    for name in ['BaseNet']:

        print(f"Running {name}")
        net = loadNet(name)
        print(net)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

        trainloader, testloader = concat_get_train_test_loader()

        train_losses, test_losses, train_accs, test_accs = [], [], [], []

        epochs = 20

        for epoch in range(epochs):
            train_loss, train_acc = train(net, criterion, optimizer, trainloader, epoch)

            test_loss, test_acc = test(net, criterion, testloader)

            train_losses.append(train_loss)
            train_accs.append(train_acc)

            test_losses.append(test_loss)
            test_accs.append(test_acc)

        print("Plotting ")
        plot_learning_curves([train_losses, test_losses], [train_accs, test_accs], epochs)

        torch.save(net.state_dict(), f"models/model_params_{name}.pth")


def train(net, criterion, optimizer, trainloader, epoch):
    running_loss = 0.0
    total = 0
    correct = 0
    for i, data in enumerate(trainloader, 0):
        images = Variable(data['image'].float())
        labels = Variable(data['label'].long())
        optimizer.zero_grad()

        # Forward pass
        outputs = net(images)
        loss = criterion(outputs, labels[:, 0])
        # Backward pass
        loss.backward()
        # SGD step
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 0:
            print(f"[Epoch, i]: [{epoch},{i}] loss: {running_loss / (i + 1)}")

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels[:, 0]).sum().item()

    return running_loss / len(trainloader), 100 * correct / total


def test(net, criterion, test_loader):
    """
    Validate network with test data
    :param net: network
    :param criterion: loss function
    :param test_loader: loader for testset
    :return:
    """
    print("Testing")
    running_loss = 0.0
    total = 0
    correct = 0

    # No gradiant calculations needed for validation
    with torch.no_grad():
        for data in test_loader:
            images = Variable(data['image'].float())
            labels = Variable(data['label'].long())

            # forward pass
            outputs = net(images)
            loss = criterion(outputs, labels[:, 0])

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels[:, 0]).sum().item()

    print("Testing done")
    return running_loss / len(test_loader), 100 * correct / total


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


if __name__ == '__main__':
    main()
