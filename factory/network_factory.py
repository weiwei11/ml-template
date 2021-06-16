# Author: weiwei
from omegaconf import OmegaConf

from factory._factory import _factory

_network_component = {
    'cifar10': ['lib.networks', '.cifar10', 'Net'],
}


def make_network(network_cfg):
    # network_cfg = cfg.network
    net_class = _factory(network_cfg.name, _network_component)
    net = net_class(**{k: v for k, v in OmegaConf.to_object(network_cfg).items() if k != 'name'})
    return net


if __name__ == '__main__':
    def main():
        from lib.config import cfg
        net = make_network(cfg)
        print(net)
    main()
