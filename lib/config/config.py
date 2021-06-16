import datetime

import argparse
import logging
import os
import sys
import time

from lib.utils.wind import file_util as wfile_util
from lib.utils.wind import git_info as wgit_info
from omegaconf import OmegaConf
from lib.config import resolver
# import resolver


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--template_file', default='configs/template.yaml', type=str)
    parser.add_argument('--cfg_file', default='configs/custom.yaml', type=str)
    parser.add_argument('--test', action='store_true', dest='test', default=False)
    parser.add_argument('--type', type=str, default="")
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return args


def parse_config(args):
    cfg_template = OmegaConf.load(args.template_file)
    cfg_custom = OmegaConf.load(args.cfg_file)
    cfg_cli = OmegaConf.from_cli(args.opts)
    cfg = OmegaConf.merge(cfg_template, cfg_custom, cfg_cli)
    OmegaConf.resolve(cfg)
    OmegaConf.set_readonly(cfg, True)
    print('---------------------------------------------')
    print(OmegaConf.to_yaml(cfg))
    print('---------------------------------------------')
    return cfg


def get_logger(cfg, logger_name='main-logger'):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter('[%(asctime)s][%(filename)s:%(lineno)d][%(levelname)s]:%(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(fmt)
    logger.addHandler(handler)

    file_name = time.strftime('%Y_%m_%d_%H_%M_%S.log')
    file_h = logging.FileHandler(os.path.join(cfg.exp.record_dir, file_name))
    file_h.setFormatter(fmt)
    logger.addHandler(file_h)
    print('log file: {}'.format(os.path.join(cfg.exp.record_dir, file_name)))
    return logger


def init_exp(cfg):
    if len(cfg.exp.task) == 0:
        raise ValueError('task must be specified')

    # assign the gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join([str(gpu) for gpu in cfg.options.gpus])

    # make dirs
    wfile_util.makedirs(cfg.exp.model_dir, verbose=True)
    wfile_util.makedirs(cfg.exp.record_dir, verbose=True)
    # wfile_util.makedirs(cfg.exp.result_dir, verbose=True)

    if not cfg.options.debug:
        # log the current config
        # wfile_util.makedirs(cfg.exp.record_dir, verbose=True)
        OmegaConf.save(cfg, os.path.join(cfg.exp.record_dir, 'config.yaml'), True)
        print(f'Current config write to {os.path.join(cfg.exp.record_dir, "config.yaml")}')
        # log the code difference
        with open(os.path.join(cfg.exp.record_dir, 'diff.txt'), 'w') as f:
            f.write(wgit_info.get_diff())
        print(f'Current code difference write {os.path.join(cfg.exp.record_dir, "diff.txt")}')

        if cfg.options.use_wandb:
            import wandb
            # project includes many experiments, and name is name of experiment, notes is detail of experiment
            # wandb.tensorboard.patch(root_logdir=os.path.join(cfg.record_dir, cfg.task), tensorboardX=True)
            wandb.init(project=cfg.exp.project, name=cfg.exp.name, tags=cfg.exp.tags,
                       job_type=cfg.exp.job_type, group=cfg.exp.group, notes=cfg.exp.notes, sync_tensorboard=False,
                       config=OmegaConf.to_object(cfg))


args = parse_argument()
cfg = parse_config(args)
init_exp(cfg)
logger = get_logger(cfg)
print = logger.info
