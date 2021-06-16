# Author: weiwei

from omegaconf import OmegaConf

from factory._factory import _factory

_visualizer_component = {
    'default': ['lib.visualizers', '.base_visualizer', 'BaseVisualizer'],
    'cifar10': ['lib.visualizers', '.cifar10', 'Visualizer'],
}


def make_visualizer(visualizer_cfg):
    visualizer_class = _factory(visualizer_cfg.name, _visualizer_component)
    visualizer = visualizer_class(**{k: v for k, v in OmegaConf.to_object(visualizer_cfg).items() if k != 'name'})
    return visualizer


if __name__ == '__main__':
    def main():
        from lib.config import cfg
        visualizer = make_visualizer(cfg.visualizer)
        print(visualizer)
    main()
