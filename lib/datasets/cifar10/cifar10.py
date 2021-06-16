import torch.utils.data as data
import torchvision
import numpy as np


class Dataset(data.Dataset):

    def __init__(self, cfg, transform=None, is_train=False):
        super(Dataset, self).__init__()

        self.cfg = cfg
        self.data_root = cfg.data_root
        self.is_train = is_train
        self.split = cfg.split
        self._transform = transform

        train = True if cfg.split == 'train' else False
        self.dataset = torchvision.datasets.CIFAR10(self.data_root, train, download=True)
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def __getitem__(self, index):
        img, label = self.dataset[index]
        inp = self._transform(img)
        return {'inp': inp, 'label': label, 'img': np.asarray(img).astype(np.uint8), 'meta': {'classes': self.classes}}

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    from lib.config import cfg
    train_set = Dataset(cfg.train.dataset, 'train', None)
    print(len(train_set))
