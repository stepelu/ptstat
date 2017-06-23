# PtStat

Probabilistic Programming and Statistical Inference in PyTorch.

# Introduction

This project is being developed during my time at [Cogent Labs](https://www.cogent.co.jp/).

The documentation is still WIP, a brief API description is reported below. The [tests](test) might also be helpful.

The API might change quickly during this initial development period.

# API

The first dimension is the batch dimension, over which the samples are assumed to be independent.

```Python

# Random variables interface:

class RandomVariable:
    def size(self)        # --> (batch_size, rv_dimension)
        
    def log_pdf(self, x)  # --> [batch_size]

    def sample(self)      # --> [batch_size, rv_dimension]

    def entropy(self)     # --> [batch_size]


# Implemented random variables:

Bernoulli(p, p_min=1E-6)

CategoricalUniform(batch_size, num_classes, cuda=False)
Categorical(p, p_min=1E-6)

Uniform01(size, cuda=False)

NormalUnit(size, cuda=False)
NormalDiagonal(mu, sd)

# KL-Divergence:

def kld(rv_from, rv_to)  # --> [batch_size]

```

# Licensing

The code is released under the [MIT license](LICENSE).
