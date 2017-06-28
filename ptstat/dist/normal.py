import torch
from torch.autograd import Variable
import numpy as np
from ptstat.core import RandomVariable, _to_v, _kld_dispatch


class Normal(RandomVariable):
    """
    Normal(mu, diagonal_sd) rv.
    """
    def __init__(self, mu=0, sd=1, size=None, cuda=False):
        super(Normal, self).__init__()
        if size:
            assert len(size) == 2, str(size)
            mu = _to_v(mu, size, cuda)
            sd = _to_v(sd, size, cuda)
        else:
            assert len(mu.size()) == 2, str(mu.size())
            assert mu.size() == sd.size(), str(sd.size())
            assert torch.min(sd.data) >= 0, str(torch.min(sd.data))
        self._mu = mu
        self._sd = sd

    def _size(self):
        return self._mu.size()

    def _log_pdf(self, x):
        mu, sd = self._mu, self._sd
        constant = x.size(1) * np.log(2 * np.pi)
        return - 0.5 * (torch.sum(torch.log(sd ** 2), 1)
                        + torch.sum((x - mu) ** 2 / sd ** 2, 1)
                        + constant).squeeze()

    def _sample(self):
        mu, sd = self._mu, self._sd
        eps = Variable(mu.data.new().resize_as_(mu.data).normal_())
        return mu + sd * eps  # Pathwise gradient, OK.

    def _entropy(self):
        sd = self._sd
        return (torch.sum(torch.log(sd), 1) + 0.5 * sd.size()[1] * np.log(2 * np.pi * np.exp(1))).squeeze()


def _kld_normal(p, q):
    mu, sd = p._mu, p._sd
    qmu, qsd = q._mu, q._sd
    assert torch.max(qmu) == torch.min(qmu) == 0, "q not Normal(0, I) is not yet supported"
    assert torch.max(qsd) == torch.min(qsd) == 1, "q not Normal(0, I) is not yet supported"
    return - 0.5 * torch.sum(1 + torch.log(sd ** 2) - mu ** 2 - sd ** 2, 1).squeeze()

_kld_dispatch[(Normal, Normal)] = _kld_normal
