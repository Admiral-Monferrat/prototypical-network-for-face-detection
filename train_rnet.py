from model.networks.RefineNetwork import *
from trainer import *

if __name__ == '__main__':
    net = RefineNetwork()
    trainer = Trainer(net, data_path='data/24', save_path='pretrained/rnet.pth', batch_size=512)
    trainer(stop_value=0.001)
