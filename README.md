# PtStat

Probabilistic Programming and Statistical Inference in PyTorch.

## Introduction

This project is being developed during my time at [Cogent Labs](https://www.cogent.co.jp/).

The documentation is still WIP, a brief API description is reported below. The [tests](test) might also be helpful.

The API might change quickly during this initial development period.

## API

The first dimension is the batch dimension, over which the samples are assumed to be independent.

```Python

# Random variables interface:

class RandomVariable:
    def size(self)        # --> (batch_size, rv_dimension)
        
    def log_pdf(self, x)  # --> [batch_size]

    def sample(self)      # --> [batch_size, rv_dimension]

    def entropy(self)     # --> [batch_size]


# Implemented random variables:

Normal(size=(batch_size, rv_dimension), cuda=cuda)
Normal(mu, sd)

Categorical(size=(batch_size, rv_dimension), cuda=cuda)
Categorical(p)

Bernoulli(size=(batch_size, rv_dimension), cuda=cuda)
Bernoulli(p)

Uniform(size=(batch_size, rv_dimension), cuda=cuda)

# KL-Divergence:

def kld(rv_from, rv_to)  # --> [batch_size]

```

## Changelog

### Version 0.2.0

+ removed specialized distributions => more flexible constructors
+ refactoring: distributions into multiple files

### Version 0.1.0

+ initial commit

## Licensing

The code is released under the [MIT license](LICENSE).
