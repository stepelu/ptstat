
import torch
from torch.autograd import Variable

# TODO:
# Remove Variable() everywhere when auto-promotion implemented.
# Make size() method indexable.
# Option to remove asserts.
# Rename log_pdf to log_p ?


def _to_v(x, size=None, cuda=False):
    if isinstance(x, Variable):
        y = x
    elif torch.is_tensor(x):
        y = Variable(x)
    else:
        y = Variable(torch.cuda.FloatTensor([x])) if cuda else Variable(torch.FloatTensor([x]))
    if size:
        assert y.size() == (1, ), str(y.size())
        y = y.expand(size)
    return y


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


_kld_dispatch = {}


# [batch_size]
def kld(p, q):
    assert p.size() == q.size()
    batch_kld = _kld_dispatch[(type(p), type(q))](p, q)
    assert batch_kld.size() == (p.size()[0], )
    assert torch.min(batch_kld.data) >= 0
    return batch_kld
