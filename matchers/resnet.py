from matchers.networks.cnn_utils import *
import os


class Residual(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_1x1conv: bool = False, stride: int = 1) -> None:
        super(Residual, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = None
        if use_1x1conv:
            self.conv3 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        return F.relu(x + y)


def resnet_block(in_channels: int, out_channels: int, num_residuals: int,
                 first_block: bool = False) -> torch.nn.Sequential:
    if first_block:
        assert in_channels == out_channels
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return torch.nn.Sequential(*blk)


def resnet(num_class: int, in_channels: int):
    net = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)),
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
    net.add_module("global_avg_pool", GlobalAvgPool2d())
    net.add_module("fc", torch.nn.Sequential(FlattenLayer(), torch.nn.Linear(512, num_class)))
    return net


def res_encoder(in_channels: int, feature_dim: int = 128):
    net = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=3),
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
    net.add_module("global_avg_pool", GlobalAvgPool2d())
    net.add_module("fc", torch.nn.Sequential(FlattenLayer(), torch.nn.Linear(512, feature_dim)))
    return net


if __name__ == "__main__":
    resize = 96
    input_transformer = T.Compose([
        T.ToTensor(),
        T.Resize((resize, resize)),
        T.Grayscale(num_output_channels=1)
    ])
    encoder = res_encoder(1)
    encoder.load_state_dict(torch.load(os.path.join("..", "output//best_model.pth")))
    db_path = os.path.join("..", "datasets//CS2003-dataset//CS2003-DB")
    target_path = os.path.join("..", "datasets//CS2003-dataset//train//李志航//train_413_0.jpg")
