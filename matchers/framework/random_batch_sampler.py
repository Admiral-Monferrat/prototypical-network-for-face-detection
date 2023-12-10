import numpy as np
import torch


class RandomizedBatchSampler(object):
    def __init__(self, labels, classes_per_it: int, num_samples: int, batch_size: int):
        super(RandomizedBatchSampler, self).__init__()
        self.labels = labels
        self.classes_per_it = classes_per_it
        self.samples_per_class = num_samples
        self.batch_size = batch_size
        self.classes, self.counts = np.unique(self.labels, return_counts=True)
        self.classes = torch.LongTensor(self.classes)
        self.idxs = range(len(self.labels))
        self.indexes = np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan
        self.indexes = torch.Tensor(self.indexes)
        self.numel_per_class = torch.zeros_like(self.classes)
        for i, label in enumerate(self.labels):
            label_loc = np.argwhere(self.classes == label).item()
            self.indexes[label_loc, np.where(np.isnan(self.indexes[label_loc]))[0][0]] = i
            self.numel_per_class[label_loc] += 1

    def __iter__(self):
        spc = self.samples_per_class
        cpi = self.classes_per_it
        for j in range(self.batch_size):
            batch_size = spc * cpi
            batch = torch.LongTensor(batch_size)
            class_idxs = torch.randperm(len(self.classes))[:cpi]
            for i, c in enumerate(self.classes[class_idxs]):
                s = slice(i * spc, (i + 1) * spc)
                label_idx = torch.arange(len(self.classes)).long()[self.classes == c].item()
                sample_idxs = torch.randperm(self.numel_per_class[label_idx])[:spc]
                batch[s] = self.indexes[label_idx][sample_idxs]
            batch = batch[torch.randperm(len(batch))]
            yield batch

    def __len__(self):
        return self.batch_size


if __name__ == "__main__":
    pass
