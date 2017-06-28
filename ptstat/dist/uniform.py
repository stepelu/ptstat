import torch
from torch.autograd import Variable
from ptstat.core import RandomVariable, _to_v


# TODO: Implement Uniform(a, b) constructor.
class Uniform(RandomVariable):
    """
    Uniform(0, 1) iid rv.
    """
    def __init__(self, size, cuda=False):
        super(Uniform, self).__init__()
        assert len(size) == 2, str(size)
        self._cuda = cuda
        self._p_size = size

    def _size(self):
        return self._p_size

    def _log_pdf(self, x):
        return self._entropy()

    def _sample(self):
        # TODO: Use CUDA random_ when implemented.
        y = Variable(torch.FloatTensor(*self._p_size).uniform_())
        if self._cuda:
            y = y.cuda()
        return y

    def _entropy(self):
        return _to_v(0, self._p_size[0], self._cuda)
