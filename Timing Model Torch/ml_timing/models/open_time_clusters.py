from models.open_time_dist import build_clusters_from_config
import torch


class OpenTimeClusters:
    def __init__(self, config):
        self.clusters = build_clusters_from_config(config)

    def log_likelihoods(self, t):
        # breakpoint()
        lp = torch.empty(t.seq_len(), len(self))
        for k in range(len(self)):
            lp[:, k] = self.clusters[k].log_prob(t.data)
        return lp

    def __len__(self):
        return len(self.clusters)
