# Author: weiwei
import torch.nn as nn


class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()

        self.net = net
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, batch):
        output = self.net(batch['inp'])
        label = batch['label']

        scalar_stats = {}
        loss = self.criterion(output['prob'], label)

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return output, loss, scalar_stats, image_stats
