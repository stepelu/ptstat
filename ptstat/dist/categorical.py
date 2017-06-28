import torch
from ptstat.core import RandomVariable, _to_v


class Categorical(RandomVariable):
    """
    Categorical over 0,...,N-1 with arbitrary probabilities, 1-dimensional rv, long type.
    """
    def __init__(self, p=None, p_min=1E-6, size=None, cuda=False):
        super(Categorical, self).__init__()
        if size:
            assert len(size) == 2, str(size)
            p = _to_v(1 / size[1], size, cuda)
        else:
            assert len(p.size()) == 2, str(p.size())
        assert torch.min(p.data) >= 0, str(torch.min(p.data))
        assert torch.max(torch.abs(torch.sum(p.data, 1) - 1)) <= 1E-5
        self._p = torch.clamp(p, p_min)

    def _size(self):
        return self._p.size()[0], 1  # Type is Long.

    def _log_pdf(self, x):
        return torch.log(self._p.gather(1, x)).squeeze()

    def _sample(self):
        return self._p.multinomial(1, True)

    def _entropy(self):
        return - torch.sum(self._p * torch.log(self._p), 1).squeeze()
