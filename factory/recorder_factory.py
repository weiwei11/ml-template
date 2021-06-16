# Author: weiwei
from factory._factory import _factory

_recorder_component = {
    'default': ['lib.train', '.recorder', 'Recorder'],
}


def make_recorder(recorder_cfg):
    recorder_class = _factory(recorder_cfg.name, _recorder_component)
    return recorder_class(recorder_cfg)


if __name__ == '__main__':
    from lib.config import cfg
    print(make_recorder(cfg.train.recorder))
