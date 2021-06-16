# Author: weiwei

from factory._factory import _factory

_transform_component = {
    'simple': ['lib.datasets', '.transforms', 'simple_transform'],
}


def make_transform(transform_cfg, is_train):
    # if is_train:
    #     transform_cfg = cfg.train.transform
    #     transform = _factory(transform_cfg.name, _transform_component)
    # else:
    #     transform_cfg = cfg.test.transform
    #     transform = _factory(transform_cfg.name, _transform_component)
    transform = _factory(transform_cfg.name, _transform_component)
    return transform(transform_cfg, is_train)


if __name__ == '__main__':
    def main():
        from lib.config import cfg
        transform = _factory('simple', _transform_component)
        print(transform(cfg.train.transform, True))
    main()
