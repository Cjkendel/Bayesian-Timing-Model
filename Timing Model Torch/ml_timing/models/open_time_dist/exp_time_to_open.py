from .base import Base
from torch.distributions.exponential import Exponential


class ExpTimeToOpen(Base):
    def __init__(self, rate):
        self.dist = Exponential(rate)

    def log_prob(self, times):
        dt = times[:, 0]
        return self.dist.log_prob(dt)
