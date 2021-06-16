# Author: weiwei
import torch
import torch.utils.data
from omegaconf import OmegaConf

from factory._factory import _factory
from factory.collator_factory import make_collator
from factory.dataset_factory import make_dataset
from factory.transform_factory import make_transform
import numpy as np
import time


torch.multiprocessing.set_sharing_strategy('file_system')

_dataloader_component = {
    'default': ['torch.utils', '.data', 'DataLoader'],
}

_batch_sampler_component = {
    'default': ['torch.utils.data', '.sampler', 'BatchSampler'],
    'iteration': ['lib.datasets', '.samplers', 'IterationBaseBatchSampler'],
    'image_size': ['lib.datasets', '.samplers', 'ImageSizeBatchSampler'],
}


def make_dataloader(dataloader_cfg, dataset, is_train=True):
    # if is_train:
    #     dataloader_cfg = cfg.train.dataloader
    # else:
    #     dataloader_cfg = cfg.test.dataloader

    # config
    shuffle = dataloader_cfg.shuffle
    batch_size = dataloader_cfg.batch_size
    drop_last = dataloader_cfg.drop_last
    num_workers = dataloader_cfg.num_workers

    dataloader_class = _factory(dataloader_cfg.name, _dataloader_component)

    sampler = make_data_sampler(dataset, shuffle)
    batch_sampler = make_batch_data_sampler(dataloader_cfg, sampler, batch_size, drop_last, is_train)
    collator = make_collator(dataloader_cfg, is_train)
    data_loader = dataloader_class(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=collator,
        worker_init_fn=worker_init_fn
    )

    return data_loader


def make_data_sampler(dataset, shuffle):
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def make_batch_data_sampler(dataloader_cfg, sampler, batch_size, drop_last, is_train):
    batch_sampler_cfg = dataloader_cfg.batch_sampler
    batch_sampler_class = _factory(batch_sampler_cfg.name, _batch_sampler_component)

    batch_sampler = batch_sampler_class(sampler=sampler, batch_size=batch_size, drop_last=drop_last,
                                        **{k: v for k, v in OmegaConf.to_object(batch_sampler_cfg).items() if k != 'name'})

    # batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size, drop_last)

    # if max_iter != -1:
    #     batch_sampler = samplers.IterationBasedBatchSampler(batch_sampler, max_iter)
    #
    # strategy = cfg.train.batch_sampler if is_train else cfg.test.batch_sampler
    # if strategy == 'image_size':
    #     batch_sampler = samplers.ImageSizeBatchSampler(sampler, batch_size, drop_last, 256, 480, 640)

    return batch_sampler


def worker_init_fn(worker_id):
    np.random.seed(worker_id + (int(round(time.time() * 1000) % (2 ** 16))))


if __name__ == '__main__':
    def main():
        from lib.config import cfg
        is_train = True
        transforms = make_transform(cfg, is_train)
        dataset = make_dataset(cfg, transforms, is_train)
        print(make_dataloader(cfg, dataset, is_train))
        # print(make_dataloader(cfg, False))
    main()
