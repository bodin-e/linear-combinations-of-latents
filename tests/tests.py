
import numpy as np
import pytest
from scipy.stats import norm, uniform, logistic, beta
from lolatents import lol_torch
from lolatents import lol_numpy
import torch
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

def test_singularity_at_zero_handling():
  num_dims = 1
  batch_dimension = 10000
  num_seeds = 10
  mean = np.random.randn(num_dims)
  std = np.random.gamma(scale=1, shape=1, size=num_dims)
  w = np.zeros([batch_dimension, num_seeds])
  X = np.random.normal(loc=mean, scale=std, size=[batch_dimension, num_seeds, num_dims])
  output_numpy = lol_numpy.lol_gaussian(w, X, mean=mean)
  _validate_samples(
    samples=output_numpy.flatten(),
    dist=norm(loc=mean[0], scale=std[0])
  )
  output_torch = lol_torch.lol_gaussian(torch.tensor(w), torch.tensor(X), mean=torch.tensor(mean))
  # Check if the output is the same for both implementations
  assert np.allclose(output_numpy, output_torch.numpy(), atol=1e-5), "Outputs are not close enough"

def _test_lol_scalar(batch_dimension, num_seeds, dist):
  w = np.random.normal(size=[batch_dimension, num_seeds])
  X = dist.rvs(size=[batch_dimension, num_seeds])
  output_numpy = lol_numpy.lol_scalar(w, X, dist.cdf, dist.ppf)
  _validate_samples(samples=output_numpy, dist=dist)
  cdf = lambda x: torch.tensor(dist.cdf(x.numpy()), dtype=torch.float64)
  ppf = lambda x: torch.tensor(dist.ppf(x.numpy()), dtype=torch.float64)
  output_torch = lol_torch.lol_scalar(torch.tensor(w), torch.tensor(X), cdf, ppf)
  # Check if the output is the same for both implementations
  assert np.allclose(output_numpy, output_torch.numpy(), atol=1e-5), "Outputs are not close enough"

def _test_lol_iid(batch_dimension, num_seeds, num_dims, dist):
  w = np.random.normal(size=[batch_dimension, num_seeds])
  X = dist.rvs(size=[batch_dimension, num_seeds, num_dims])
  output_numpy = lol_numpy.lol_iid(w, X, dist.cdf, dist.ppf)
  for d in range(num_dims):
    _validate_samples(
      samples=output_numpy[:, d],
      dist=dist
    )
  cdf = lambda x: torch.tensor(dist.cdf(x.numpy()), dtype=torch.float64)
  ppf = lambda x: torch.tensor(dist.ppf(x.numpy()), dtype=torch.float64)
  output_torch = lol_torch.lol_iid(torch.tensor(w), torch.tensor(X), cdf, ppf)
  # Check if the output is the same for both implementations
  assert np.allclose(output_numpy, output_torch.numpy(), atol=1e-5), "Outputs are not close enough"

def _test_lol_gaussian(batch_dimension, num_seeds, num_dims):
  mean = np.random.randn(num_dims)
  std = np.random.gamma(scale=1, shape=1, size=num_dims)
  w = np.random.normal(size=[batch_dimension, num_seeds])
  X = np.random.normal(loc=mean, scale=std, size=[batch_dimension, num_seeds, num_dims])
  output_numpy = lol_numpy.lol_gaussian(w, X, mean=mean)
  for d in range(num_dims):
    _validate_samples(
      samples=output_numpy[:, d],
      dist=norm(loc=mean[d], scale=std[d])
    )
  output_torch = lol_torch.lol_gaussian(torch.tensor(w), torch.tensor(X), mean=torch.tensor(mean))
  # Check if the output is the same for both implementations
  assert np.allclose(output_numpy, output_torch.numpy(), atol=1e-5), "Outputs are not close enough"

def _validate_samples(samples, dist, min_pvalue=1e-5):
  result = kstest(samples, dist.cdf)
  # Check that the probability of the sample coming from the same distribution is high enough to
  # reasonably be seen by chance.
  # Note that we expect this test-case to fail by chance with probability min_pvalue if it is correct.
  assert result.pvalue >= min_pvalue

if __name__ == "__main__":
  pytest.main()
