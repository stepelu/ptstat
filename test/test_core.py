

import unittest
import numpy as np
import ptstat as stat
import torch
import numpy.testing as npt
from torch.autograd import Variable


# Make sure tests are deterministic:
cuda = False  # torch.cuda.is_available()
np.random.seed(1)
torch.manual_seed(1)
if cuda:
    torch.cuda.manual_seed(1)


def to_np(x):
    if cuda:
        return x.data.cpu().numpy()
    else:
        return x.data.numpy()


class TestRandomVariables(unittest.TestCase):
    def setUp(self):
        batch_size = 2
        rv_dimension = 5
        p = torch.normal(torch.zeros(batch_size, rv_dimension), torch.ones(batch_size, rv_dimension))
        p_pos = torch.abs(torch.normal(torch.zeros(batch_size, rv_dimension), torch.ones(batch_size, rv_dimension)))
        p_pos = torch.clamp(p_pos, 0.1, 0.9)
        if cuda:
            p = p.cuda()
            p_pos = p_pos.cuda()
        p = Variable(p)
        p_pos = Variable(p_pos)
        self.rv = [
            stat.Normal(size=(batch_size, rv_dimension), cuda=cuda),
            stat.Normal(p, p_pos),
            stat.Categorical(size=(batch_size, rv_dimension), cuda=cuda),
            stat.Categorical(p_pos / torch.sum(p_pos, 1).expand_as(p_pos)),
            stat.Bernoulli(size=(batch_size, rv_dimension), cuda=cuda),
            stat.Bernoulli(p_pos),
            stat.Uniform(size=(batch_size, rv_dimension), cuda=cuda)
        ]

    def test_entropy(self):
        mc_samples = 5000
        for rv in self.rv:
            entropy_avg = 0
            x_avg = 0
            for n in range(mc_samples):
                x = rv.sample()
                x_avg += x
                entropy_avg += - rv.log_pdf(x)
            entropy_avg /= mc_samples
            x_avg /= mc_samples
            npt.assert_allclose(to_np(entropy_avg), to_np(rv.entropy()), 0.02)


if __name__ == '__main__':
    unittest.main()
