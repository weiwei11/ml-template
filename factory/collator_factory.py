# Author: weiwei
from factory._factory import _factory

_collator_component = {
    'default': ['torch.utils.data', '.dataloader', 'default_collate'],
}


def make_collator(dataloader_cfg, is_train):
    collator_cfg = dataloader_cfg.collator
    return _factory(collator_cfg, _collator_component)


if __name__ == '__main__':
    from lib.config import cfg
    print(make_collator(cfg.train.dataloader, True))
