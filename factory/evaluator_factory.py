# Author: weiwei

from omegaconf import OmegaConf

from factory._factory import _factory

_evaluator_component = {
    'default': ['lib.evaluators', '.base_evaluator', 'BaseEvaluator'],
    'cifar10': ['lib.evaluators', '.cifar10', 'Evaluator'],
}


def make_evaluator(evaluator_cfg):
    evaluator_class = _factory(evaluator_cfg.name, _evaluator_component)
    evaluator = evaluator_class(**{k: v for k, v in OmegaConf.to_object(evaluator_cfg).items() if k != 'name'})
    return evaluator


if __name__ == '__main__':
    def main():
        from lib.config import cfg
        evaluator = make_evaluator(cfg.evaluator)
        print(evaluator)
    main()
