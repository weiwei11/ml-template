# Author: weiwei
from omegaconf import OmegaConf

from factory._factory import _factory
from factory.network_factory import make_network

_wrapper_component = {
    'cifar10': ['lib.wrappers', '.cifar10', 'NetworkWrapper'],
}


def make_wrapper(wrapper_cfg, net):
    # wrapper_cfg = cfg.train.network_wrapper
    net_class = _factory(wrapper_cfg.name, _wrapper_component)
    net = net_class(net, **{k: v for k, v in OmegaConf.to_object(wrapper_cfg).items() if k != 'name'})
    return net


if __name__ == '__main__':
    def main():
        from lib.config import cfg
        net = make_network(cfg)
        net = make_wrapper(cfg, net)
        print(net)
    main()
