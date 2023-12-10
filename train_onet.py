from model.networks.OutputNetwork import *
from trainer import *

if __name__ == '__main__':
    net = OutputNetwork()
    trainer = Trainer(net, data_path='data/48', save_path='pretrained/onet.pth', batch_size=512)
    trainer(stop_value=0.001, net='onet')
