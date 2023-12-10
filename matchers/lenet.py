from networks.cnn_utils import *


def lenet():
    net = nn.Sequential(
        nn.Conv2d(1, 6, 5),
        nn.BatchNorm2d(6),
        nn.Relu(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(6, 16, 5),
        nn.BatchNorm2d(16),
        nn.Relu(),
        nn.MaxPool2d(2, 2),
        FlattenLayer(),
        nn.Linear(16 * 21 * 21, 120),
        nn.BatchNorm1d(120),
        nn.Relu(),
        nn.Linear(120, 84),
        nn.BatchNorm1d(84),
        nn.Relu(),
        nn.Linear(84, 128)
    )
    return net
