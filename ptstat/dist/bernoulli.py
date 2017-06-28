import torch
from ptstat.core import RandomVariable, _to_v


class Bernoulli(RandomVariable):
    """
    Vector of iid Bernoulli rvs, float type.
    """
    def __init__(self, p=0.5, p_min=1E-6, size=None, cuda=False):
        super(Bernoulli, self).__init__()
        if size:
            assert len(size) == 2, str(size)
            p = _to_v(p, size, cuda)
        else:
            assert len(p.size()) == 2, str(p.size())
            assert torch.max(p.data) <= 1, str(torch.max(p.data))
            assert torch.min(p.data) >= 0, str(torch.min(p.data))
        self._p = torch.clamp(p, p_min, 1 - p_min)

    def _size(self):
        return self._p.size()  # Type is Float.

    def _log_pdf(self, x):
        p = self._p
        return torch.sum(x * torch.log(p) + (1 - x) * torch.log(1 - p), 1).squeeze()

    def _sample(self):
        return self._p.bernoulli()

    def _entropy(self):
        p = self._p
        return - torch.sum(p * torch.log(p) + (1 - p) * torch.log(1 - p), 1).squeeze()
