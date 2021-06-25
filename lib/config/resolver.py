# Author: weiwei
import sys
import time
from datetime import datetime
from lib.utils.wind import git_info as wgit_info
from omegaconf import OmegaConf


def datetime_now(fmt):
    return datetime.now().strftime(fmt)


OmegaConf.register_new_resolver('datetime_now', datetime_now, replace=True)
OmegaConf.register_new_resolver('cur_version', wgit_info.get_hash, replace=True)
OmegaConf.register_new_resolver('condition', lambda c, x1, x2: x1 if c else x2)
OmegaConf.register_new_resolver('is_debug', lambda: True if sys.gettrace() else False)


if __name__ == '__main__':
    cfg = OmegaConf.load('configs/template.yaml')
    print(cfg)
    print(OmegaConf.resolve(cfg))
    print(cfg)
    time.sleep(10)
    print(cfg)
