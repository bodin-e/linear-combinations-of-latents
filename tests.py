
import numpy as np
import pytest
from scipy.stats import norm, uniform, logistic, beta
from lol import lol_gaussian, lol_iid, lol_scalar
from scipy.stats import kstest

np.random.seed(0)

def test_distribution_lol_gaussian():
  _test_lol_gaussian(batch_dimension=100000, num_seeds=100, num_dims=1)
  _test_lol_gaussian(batch_dimension=100000, num_seeds=10, num_dims=8)
  _test_lol_gaussian(batch_dimension=1, num_seeds=5, num_dims=3)

def test_distribution_lol_scalar():
  _test_lol_scalar(batch_dimension=100000, num_seeds=100, dist=logistic())
  _test_lol_scalar(batch_dimension=100000, num_seeds=100, dist=uniform())
  _test_lol_scalar(batch_dimension=100000, num_seeds=100, dist=beta(a=1, b=5))

def test_distribution_lol_iid():
  _test_lol_iid(batch_dimension=100000, num_seeds=100, num_dims=1, dist=logistic())
  _test_lol_iid(batch_dimension=100000, num_seeds=100, num_dims=8, dist=uniform())
  _test_lol_iid(batch_dimension=1, num_seeds=5, num_dims=3, dist=beta(a=1, b=5))

def _test_lol_scalar(batch_dimension, num_seeds, dist):
  w = np.random.normal(size=[batch_dimension, num_seeds])
  X = dist.rvs(size=[batch_dimension, num_seeds])
  output = lol_scalar(w, X, dist.cdf, dist.ppf)
  _validate_samples(samples=output, dist=dist)

def _test_lol_iid(batch_dimension, num_seeds, num_dims, dist):
  w = np.random.normal(size=[batch_dimension, num_seeds])
  X = dist.rvs(size=[batch_dimension, num_seeds, num_dims])
  output = lol_iid(w, X, dist.cdf, dist.ppf)
  for d in range(num_dims):
    _validate_samples(
      samples=output[:, d],
      dist=dist
    )

def _test_lol_gaussian(batch_dimension, num_seeds, num_dims):
  mean = np.random.randn(num_dims)
  std = np.random.gamma(scale=1, shape=1, size=num_dims)
  w = np.random.normal(size=[batch_dimension, num_seeds])
  X = np.random.normal(loc=mean, scale=std, size=[batch_dimension, num_seeds, num_dims])
  output = lol_gaussian(w, X, mean=mean)
  for d in range(num_dims):
    _validate_samples(
      samples=output[:, d],
      dist=norm(loc=mean[d], scale=std[d])
    )

def _validate_samples(samples, dist):
  result = kstest(
    samples,
    dist.cdf,
  )
  # Check that the probability of the sample coming from the same distribution is high enough to
  # reasonably be seen by chance.
  assert result.pvalue > 1e-5  # This case-case should fail by chance with probability 1e-5

if __name__ == "__main__":
  pytest.main()
