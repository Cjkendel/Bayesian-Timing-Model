from .exp_time_to_open import ExpTimeToOpen
from .normal_open_time import NormalOpenTime


def build_cluster(cfg):
    if cfg[0] == 'exp':
        return ExpTimeToOpen(cfg[1])
    if cfg[0] == 'norm':
        return NormalOpenTime(cfg[1], cfg[2])
    raise RuntimeError("Unknown distribution %s", cfg[0])


def build_clusters_from_config(config):
    return [build_cluster(cfg) for cfg in config]
