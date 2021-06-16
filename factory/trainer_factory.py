# Author: weiwei

from omegaconf import OmegaConf

from factory._factory import _factory
# from factory.network_factory import make_network

_trainer_component = {
    'default': ['lib.train.trainers', '.trainer', 'Trainer'],
}


def make_trainer(trainer_cfg):
    # trainer_cfg = cfg.train.trainer
    trainer_class = _factory(trainer_cfg.name, _trainer_component)
    trainer = trainer_class(**dict(filter(lambda kv: kv[0] != 'name', trainer_cfg.items())))
    return trainer


if __name__ == '__main__':
    def main():
        from lib.config import cfg
        # net = make_network(cfg)
        trainer = make_trainer(cfg.train.trainer)
        print(trainer)
    main()
