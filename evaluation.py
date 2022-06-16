from torch.utils.data import Dataset
from torch.autograd import Variable
import torch
import numpy as np

import onnx
import onnxruntime as ort

from preprocessing import get_train_test_loader
from load_asl import get_train_test_loader_asl
from networks import BaseNet, SmallNet


def eval(out: Variable, labels: Variable) -> float:
    # evaluate the neural network outputs against non encoded data
    y_true = labels.numpy()
    y_pred = np.argmax(out, axis=1)
    return float(np.sum(y_pred == y_true))


def batch_eval(net: BaseNet, dataloader: torch.utils.data.DataLoader) -> float:
    # batch evaluation in case of larger datasets
    num_samples = 0.0
    eval_score = 0.0
    for b in dataloader:
        num_samples += len(b['image'])
        out = net(b['image'])
        if isinstance(out, torch.Tensor):
            out = out.detach().numpy()
        eval_score += eval(out, b['label'][:, 0])

    return eval_score / num_samples


def val():
    """
    Validate the model with the test set
    :return:
    """

    # Specify network to be used and model to be loaded
    trainloader, testloader, _, _ = get_train_test_loader()
    net = BaseNet().float()

    model = torch.load("models/model_params_SmallNet2.pth")
    net.load_state_dict(model)

    train_acc = batch_eval(net, trainloader) * 100
    test_acc = batch_eval(net, testloader) * 100
    print('==========', 'Pytorch', '==========')
    print(f'Training accuracy {train_acc}')
    print(f'Validation accuracy {test_acc}')

    trainloader, testloader, _, _ = get_train_test_loader(1)

    # export to onnx 
    onnx_file = 'signlanguage.onnx'
    template = torch.randn(1, 1, 28, 28)
    torch.onnx.export(net, template, onnx_file, input_names=['input'])

    # exported model 
    onnx_model = onnx.load(onnx_file)
    onnx.checker.check_model(onnx_model)

    # create runnable 
    onnx_sess = ort.InferenceSession(onnx_file)
    net = lambda x: onnx_sess.run(None, {'input': x.data.numpy()})[0]

    train_acc = batch_eval(net, trainloader) * 100
    test_acc = batch_eval(net, testloader) * 100
    print('==========', 'Onnx', '==========')
    print(f'Training accuracy {train_acc}')
    print(f'Validation accuracy {test_acc}')


def val_asl_set():
    _, testloader, _, _ = get_train_test_loader()
    net = BaseNet().float()

    pretrained_model = torch.load("models/model_params_BaseNet.pth")
    net.load_state_dict(pretrained_model)

    test_acc = batch_eval(net, testloader) * 100
    print(f'Validation accuracy: {test_acc}')


if __name__ == '__main__':
    val()
    # for x in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']:
    #     validate_asl_set(x)
