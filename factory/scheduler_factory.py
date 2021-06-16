from collections import Counter
from lib.utils.optimizer.lr_scheduler import WarmupMultiStepLR, MultiStepLR


def make_lr_scheduler(scheduler_cfg, optimizer):
    if scheduler_cfg.name == 'warmup':
        scheduler = WarmupMultiStepLR(optimizer, scheduler_cfg.milestones, scheduler_cfg.gamma, 1.0 / 3, 5, 'linear')
    elif scheduler_cfg.name == 'multi_step':
        scheduler = MultiStepLR(optimizer, milestones=scheduler_cfg.milestones, gamma=scheduler_cfg.gamma)
    else:
        raise ValueError('Scheduler name must be warmup or multi_step!')
    return scheduler


def set_lr_scheduler(cfg, scheduler):
    if cfg.train.warmup:
        scheduler.milestones = cfg.train.milestones
    else:
        scheduler.milestones = Counter(cfg.train.milestones)
    scheduler.gamma = cfg.train.gamma
