from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np

import onnx
import onnxruntime as ort

from preprocessing import get_train_test_loader
from networks import BaseNet, SmallNet


def evaluate(outputs: Variable, labels: Variable) -> float:
    # evaluate the neural network outputs against non encoded data
    Y = labels.numpy()
    Y_hat = np.argmax(outputs, axis=1)
    return float(np.sum(Y_hat == Y))


def batch_evaluation(net: SmallNet, dataloader: torch.utils.data.DataLoader) -> float:
    # batch evaluation in case of larger datasets 
    score = n = 0.0
    for batch in dataloader:
        n += len(batch['image'])
        outputs = net(batch['image'])
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.detach().numpy()
        score += evaluate(outputs, batch['label'][:, 0])

    return score / n


def validate():
    trainloader, testloader = get_train_test_loader()
    net = SmallNet().float()

    pretrained_model = torch.load("models/model_params_SmallNet2.pth")
    net.load_state_dict(pretrained_model)

    print('=' * 10, 'Pytorch', '=' * 10)
    train_acc = batch_evaluation(net, trainloader) * 100
    print('Training accuracy " %.1f' % train_acc)
    test_acc = batch_evaluation(net, testloader) * 100
    print('Validation accuracy " %.1f' % test_acc)

    trainloader, testloader = get_train_test_loader(1)

    # export to onnx 
    fname = 'signlanguage.onnx'
    dummy = torch.randn(1, 1, 28, 28)
    torch.onnx.export(net, dummy, fname, input_names=['input'])

    # exported model 
    model = onnx.load(fname)
    onnx.checker.check_model(model)

    # create runnable 
    ort_session = ort.InferenceSession(fname)
    net = lambda inp: ort_session.run(None, {'input': inp.data.numpy()})[0]

    print('=' * 10, 'ONNX', '=' * 10)
    train_acc = batch_evaluation(net, trainloader) * 100
    print('Training accuracy: %.1f' % train_acc)
    test_acc = batch_evaluation(net, testloader) * 100
    print('Validation accuracy: %.1f' % test_acc)


if __name__ == '__main__':
    validate()
