import torch
from lib.utils.optimizer.radam import RAdam


_optimizer_factory = {
    'adam': torch.optim.Adam,
    'radam': RAdam,
    'sgd': torch.optim.SGD,
}


def make_optimizer(optimizer_cfg, net):
    params = []
    lr = optimizer_cfg.lr
    weight_decay = optimizer_cfg.weight_decay
    for key, value in net.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if 'adam' in optimizer_cfg.name:
        optimizer = _optimizer_factory[optimizer_cfg.name](params, lr, weight_decay=weight_decay)
    else:
        optimizer = _optimizer_factory[optimizer_cfg.name](params, lr, momentum=0.9)

    return optimizer
