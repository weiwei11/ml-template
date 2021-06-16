# Author: weiwei

from factory._factory import _factory

_dataset_component = {
    'cifar10': ['lib.datasets.cifar10', '.cifar10', 'Dataset'],
}


def make_dataset(dataset_cfg, transform, is_train=True):
    # if is_train:
    #     dataset_cfg = cfg.train.dataset
    #     split = 'train'
    #     dataset = _factory(dataset_cfg.name, _dataset_component)
    # else:
    #     dataset_cfg = cfg.test.dataset
    #     split = 'test'
    #     dataset = _factory(dataset_cfg.name, _dataset_component)
    # split = 'train' if is_train else 'test'
    dataset = _factory(dataset_cfg.name, _dataset_component)
    return dataset(dataset_cfg, transform, is_train)


if __name__ == '__main__':
    def main():
        from lib.config import cfg
        dataset = _factory('cifar10', _dataset_component)
        print(dataset(cfg.train.dataset, True))
    main()
