from torch.nn import DataParallel

from factory.dataloader_factory import make_dataloader
from factory.dataset_factory import make_dataset
from factory.evaluator_factory import make_evaluator
from factory.network_factory import make_network
from factory.optimizer_factory import make_optimizer
from factory.recorder_factory import make_recorder
from factory.scheduler_factory import make_lr_scheduler
from factory.trainer_factory import make_trainer
from factory.transform_factory import make_transform
from factory.wrapper_factory import make_wrapper
from lib.config import cfg, args
from lib.utils.net_utils import load_model, save_model, load_network
import torch.multiprocessing


def train(cfg, network):
    torch.multiprocessing.set_sharing_strategy('file_system')

    trainer = make_trainer(cfg.train.trainer)
    optimizer = make_optimizer(cfg.train.optimizer, network.cuda())
    scheduler = make_lr_scheduler(cfg.train.scheduler, optimizer)
    recorder = make_recorder(cfg.train.recorder)
    evaluator = make_evaluator(cfg.evaluator)

    # set_lr_scheduler(cfg, scheduler)

    train_transform = make_transform(cfg.train.transform, is_train=True)
    test_transform = make_transform(cfg.test.transform, is_train=False)
    train_set = make_dataset(cfg.train.dataset, train_transform, is_train=True)
    test_set = make_dataset(cfg.test.dataset, test_transform, is_train=False)
    train_loader = make_dataloader(cfg.train.dataloader, train_set, is_train=True)
    val_loader = make_dataloader(cfg.test.dataloader, test_set, is_train=False)
    # train_loader = make_data_loader(cfg, is_train=True, max_iter=100)

    net = make_wrapper(cfg.train.network_wrapper, network)
    trainer.fit(network, train_loader, val_loader, evaluator, recorder, optimizer, scheduler, net)

    return network


def test(cfg, network):
    trainer = make_trainer(cfg.train.trainer)

    val_transform = make_transform(cfg.test.transform, is_train=False)
    val_set = make_dataset(cfg.test.dataset, val_transform, is_train=False)
    val_loader = make_dataloader(cfg.test.dataloader, val_set, False)
    evaluator = make_evaluator(cfg.evaluator)
    epoch = load_network(network, cfg.train.trainer.model_dir, resume=cfg.train.trainer.resume, epoch=cfg.train.trainer.begin_epoch)

    net = make_wrapper(cfg.train.network_wrapper, network)
    net = net.cuda()
    net = DataParallel(net)
    trainer.val(epoch, net, val_loader, evaluator)


def main():
    network = make_network(cfg.network)
    if args.test:
        test(cfg, network)
    else:
        train(cfg, network)


if __name__ == "__main__":
    main()
