import torch
import tqdm

from factory.dataloader_factory import make_dataloader
from factory.dataset_factory import make_dataset
from factory.network_factory import make_network
from factory.transform_factory import make_transform
from factory.visualizer_factory import make_visualizer
from lib.config import cfg, args
from lib.utils.net_utils import load_network


def visualize_simple():
    torch.manual_seed(0)

    network = make_network(cfg.network).cuda()
    load_network(network, cfg.exp.model_dir, epoch=cfg.test.epoch)
    network.eval()

    transform = make_transform(cfg.test.transform, is_train=False)
    dataset = make_dataset(cfg.test.dataset, transform, is_train=False)
    data_loader = make_dataloader(cfg.test.dataloader, dataset, is_train=False)
    visualizer = make_visualizer(cfg.visualizer)
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            output = network(batch['inp'])
        visualizer.visualize(output, batch)


def visualize_train():
    network = make_network(cfg.network).cuda()
    load_network(network, cfg.exp.model_dir, epoch=cfg.test.epoch)
    network.eval()

    transform = make_transform(cfg.test.transform, is_train=False)
    dataset = make_dataset(cfg.train.dataset, transform, is_train=False)
    data_loader = make_dataloader(cfg.test.dataloader, dataset, is_train=False)
    visualizer = make_visualizer(cfg.visualizer)
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            output = network(batch['inp'], batch)
        visualizer.visualize_train(output, batch)


if __name__ == '__main__':
    globals()['visualize_'+args.type]()

