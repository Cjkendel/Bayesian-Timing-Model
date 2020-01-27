from .base import Base
from torch.distributions.exponential import Exponential


class ExpTimeToOpen(Base):
    def __init__(self, rate):
        self.dist = Exponential(rate)

    def log_prob(self, times):
        dt = times[:, 1] - times[:, 0]
        dt.apply_(lambda x: x + 24 if x < 0 else x) # find more time effective way to compute
        return self.dist.log_prob(dt)
