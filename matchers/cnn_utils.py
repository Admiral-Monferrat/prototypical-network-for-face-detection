from torch.utils.data import DataLoader
from collections import OrderedDict
from collections import defaultdict
import torchvision.transforms as T
import torch.nn.functional as F
import torch.nn as nn
import random
import torch


class FlattenLayer(torch.nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class GlobalAvgPool2d(torch.nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])


def euclidean_distances(a, b):
    sq_a = a**2
    sum_sq_a = torch.sum(sq_a, dim=1).unsqueeze(1)
    sq_b = b**2
    sum_sq_b = torch.sum(sq_b, dim=1).unsqueeze(0)
    bt = b.t()
    return torch.sqrt(sum_sq_a+sum_sq_b-2*a.mm(bt))


class RandomizedTripletDataLoader:
    def __init__(self, iterator: DataLoader, batch_size: int = 10):
        samples, labels = [], []
        data_db = defaultdict(list)
        self.batch_size = batch_size
        length = iterator.batch_size
        self.num_batches = length // self.batch_size
        for batch, (x, y) in enumerate(iterator):
            for label in y:
                labels.append(label.item())
            for sample in x:
                samples.append(sample)
        for i in range(len(samples)):
            data_db[labels[i]].append(samples[i])
        num_class = len(data_db.keys())
        self.__triplet_db = self.__create_triplet_db(data_db, length, num_class)

    def __create_triplet_db(self, data_db: defaultdict, length: int, num_class: int) -> list:
        triplet_data_db = []
        index = 0
        for i in range(self.num_batches):
            anchor_db, positive_db, negative_db = [], [], []
            anchor = index
            index = (index + 1) % num_class
            negative = anchor
            while negative == anchor:
                negative = random.randint(0, num_class - 1)
            for j in range(min(self.batch_size, length)):
                anchor_idx = random.randint(0, len(data_db[anchor]) - 1)
                positive_idx = anchor_idx
                while positive_idx == anchor_idx:
                    positive_idx = random.randint(0, len(data_db[anchor]) - 1)
                negative_idx = random.randint(0, len(data_db[negative]) - 1)
                anchor_db.append(data_db[anchor][anchor_idx])
                positive_db.append(data_db[anchor][positive_idx])
                negative_db.append(data_db[negative][negative_idx])
            anchor_tensor = torch.stack(anchor_db, 0)
            positive_tensor = torch.stack(positive_db, 0)
            negative_tensor = torch.stack(negative_db, 0)
            triplet_data_db.append((anchor_tensor, positive_tensor, negative_tensor))
            length -= self.batch_size
        return triplet_data_db

    def __getitem__(self, item):
        return item, self.__triplet_db[item]

    def __len__(self):
        return self.num_batches
