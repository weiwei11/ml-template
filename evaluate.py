# Author: weiwei
from factory.dataloader_factory import make_dataloader
from factory.dataset_factory import make_dataset
from factory.evaluator_factory import make_evaluator
from factory.network_factory import make_network
from factory.transform_factory import make_transform
from lib.config import cfg, args
import tqdm
import torch

from lib.train.recorder import write_summary
from lib.utils.net_utils import load_network
from omegaconf import OmegaConf


def evaluate_simple():
    torch.manual_seed(0)

    network = make_network(cfg.network).cuda()
    load_network(network, cfg.exp.model_dir, epoch=cfg.test.epoch)
    network.eval()

    transform = make_transform(cfg.test.transform, is_train=False)
    dataset = make_dataset(cfg.test.dataset, transform, is_train=False)
    data_loader = make_dataloader(cfg.test.dataloader, dataset, is_train=False)
    evaluator = make_evaluator(cfg.evaluator)
    for batch in tqdm.tqdm(data_loader):
        inp = batch['inp'].cuda()
        with torch.no_grad():
            output = network(inp)
        evaluator.evaluate(output, batch)
    results = evaluator.summarize()
    write_summary(cfg.exp.record_dir, results, OmegaConf.to_object(cfg.hparams), cfg.options.use_wandb)


if __name__ == '__main__':
    globals()['evaluate_'+args.type]()
