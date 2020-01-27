from .base import Base
from torch.distributions.normal import Normal


class NormalOpenTime(Base):
    def __init__(self, loc, scale):
        self.dist = Normal(loc, scale)

    def log_prob(self, times):
        return self.dist.log_prob(times[:, 1])
