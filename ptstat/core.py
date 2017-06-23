
import torch
import numpy as np
from torch.autograd import Variable


# TODO:
# Remove Variable() everywhere when auto-promotion implemented.
# Make size() method indexable.


# [batch_size] -> [batch_size, num_classes].
def to_1hot(label, num_classes):
    assert len(label.size()) == 1, str(label.size())
    if label.is_cuda:
        y = torch.cuda.FloatTensor(label.size(0), num_classes).zero_()
    else:
        y = torch.zeros(label.size(0), num_classes)
    y.scatter_(1, label.data.unsqueeze(1), 1)
    return Variable(y)


# [batch_size, num_classes] -> [batch_size].
def to_label(one_hot):
    assert len(one_hot.size()) == 2, str(one_hot.size())
    _, y = torch.max(one_hot, 1).squeeze()
    return y


# [batch_size].
def label(batch, value, cuda):
    if cuda:
        return Variable(torch.cuda.LongTensor(batch).fill_(value))
    else:
        return Variable(torch.LongTensor(batch).fill_(value))


# First dimension is batch / independent samples.
# Second dimension is RV dimensionality (not identically distributed).
class RandomVariable:
    def _size(self):
        raise NotImplementedError("size is not implemented")

    def _log_pdf(self, x):
        raise NotImplementedError("log_pdf is not implemented")

    def _sample(self):
        raise NotImplementedError("sample is not implemented")

    def _entropy(self):
        raise NotImplementedError("entropy is not implemented")

    # [batch_size, rv_dimension]
    def size(self):
        return self._size()

    # [batch_size]
    def log_pdf(self, x):
        assert self.size() == x.size(), str(self.size()) + " ~ " + str(x.size())
        batch_log_pdf = self._log_pdf(x)
        assert batch_log_pdf.size() == (self.size()[0], ), str(batch_log_pdf.size()) + " ~ " + str((self.size()[0], ))
        return batch_log_pdf

    # [batch_size, rv_dimension]
    def sample(self):
        batch_samples = self._sample()
        assert self.size() == batch_samples.size(), str(self.size()) + " ~ " + str(batch_samples.size())
        return batch_samples

    # [batch_size]
    def entropy(self):
        batch_entropy = self._entropy()
        assert batch_entropy.size() == (self.size()[0], ), str(batch_entropy.size()) + " ~ " + str((self.size()[0], ))
        return batch_entropy


# Vector of iid Bernoulli rvs, float type.
class Bernoulli(RandomVariable):
    def __init__(self, p, p_min=1E-6):
        super(Bernoulli, self).__init__()
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


# Categorical over 0,...,N-1 with equal probabilities, 1-dimensional rv, long type.
class CategoricalUniform(RandomVariable):
    def __init__(self, batch_size, num_classes, cuda=False):
        super(CategoricalUniform, self).__init__()
        self._batch_size = batch_size
        self._num_classes = num_classes
        self._cuda = cuda

    def _size(self):
        return self._batch_size, 1  # Type is Long.

    def _log_pdf(self, x):
        batch_size, num_classes = self._batch_size, self._num_classes
        if self._cuda:
            return Variable(torch.cuda.FloatTensor(batch_size).fill_(-np.log(num_classes)))
        else:
            return Variable(torch.FloatTensor(batch_size).fill_(-np.log(num_classes)))

    def _sample(self):
        batch_size, num_classes = self._batch_size, self._num_classes
        if self._cuda:
            return Variable(torch.cuda.FloatTensor(batch_size, num_classes).fill_(1).multinomial(1, True))
        else:
            return Variable(torch.FloatTensor(batch_size, num_classes).fill_(1).multinomial(1, True))

    def _entropy(self):
        batch_size, num_classes = self._batch_size, self._num_classes
        batch_entropy = np.log(num_classes)
        if self._cuda:
            return Variable(torch.cuda.FloatTensor(batch_size).fill_(batch_entropy))
        else:
            return Variable(torch.FloatTensor(batch_size).fill_(batch_entropy))


# Categorical over 0,...,N-1 with arbitrary probabilities, 1-dimensional rv, long type.
class Categorical(RandomVariable):
    def __init__(self, p, p_min=1E-6):
        super(Categorical, self).__init__()
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


# Uniform(0, 1) iid rv.
class Uniform01(RandomVariable):
    def __init__(self, size, cuda=False):
        super(Uniform01, self).__init__()
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
        if self._cuda:
            return Variable(torch.cuda.FloatTensor(self._p_size[0]).fill_(0))
        else:
            return Variable(torch.FloatTensor(self._p_size[0]).fill_(0))


# Normal(0, I) rv.
class NormalUnit(RandomVariable):
    def __init__(self, size, cuda=False):
        super(NormalUnit, self).__init__()
        assert len(size) == 2, str(size)
        self._cuda = cuda
        self._p_size = size

    def _size(self):
        return self._p_size

    def _log_pdf(self, x):
        constant = x.size(1) * np.log(2 * np.pi)
        return - 0.5 * (torch.sum(x ** 2, 1) + constant).squeeze()

    def _sample(self):
        if self._cuda:
            return Variable(torch.cuda.FloatTensor(*self._p_size).normal_())
        else:
            return Variable(torch.FloatTensor(*self._p_size).normal_())

    def _entropy(self):
        entropy = 0.5 * self._p_size[1] * np.log(2 * np.pi * np.exp(1))
        if self._cuda:
            return Variable(torch.cuda.FloatTensor(self._p_size[0]).fill_(entropy))
        else:
            return Variable(torch.FloatTensor(self._p_size[0]).fill_(entropy))


# Normal(mu, diagonal_sd) rv.
class NormalDiagonal(RandomVariable):
    def __init__(self, mu, sd):
        super(NormalDiagonal, self).__init__()
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


def _kld_normal_diagonal_standard(p, _):
    mu, sd = p._mu, p._sd
    return - 0.5 * torch.sum(1 + torch.log(sd ** 2) - mu ** 2 - sd ** 2, 1).squeeze()


_kld_dispatch = {
    (NormalDiagonal, NormalUnit): _kld_normal_diagonal_standard,
}


# [batch_size]
def kld(p, q):
    assert p.size() == q.size()
    batch_kld = _kld_dispatch[(type(p), type(q))](p, q)
    assert batch_kld.size() == (p.size()[0], )
    assert torch.min(batch_kld.data) >= 0
    return batch_kld
