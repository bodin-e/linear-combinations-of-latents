
import numpy as np
from typing import Callable, Union
from jaxtyping import Float, jaxtyped
from scipy.stats import norm
from beartype import beartype

@jaxtyped(typechecker=beartype)
def lol_gaussian(
  w: Float[np.ndarray, "batch_dimension num_seeds"],
  X: Union[
    Float[np.ndarray, "num_seeds num_dims"],
    Float[np.ndarray, "batch_dimension num_seeds num_dims"]
  ],
  mean: Float[np.ndarray, "num_dims"] = None,
  jitter=1e-10
) -> Float[np.ndarray, "batch_dimension num_dims"]:
  """
  Transforms linear combinations of iid samples from N(\bm{\mu}, \bm{\Sigma}). such that it produces
  samples the same distribution as the seeds, N(\bm{\mu}, \bm{\Sigma}).

  The transform minimises the quadratic Wasserstein distance between the original distribution
  of the linear combination and the target distribution (the distribution of the seeds).

  :param w: Weights of the linear combination.
  :param X: Seed samples.
  :param mean: Mean of the distribution. \bm{\mu}
  :param jitter: Small value to avoid division by zero.
  :return: Samples from the same distribution as the seeds.
  """
  batch_dimension, num_seeds = w.shape
  num_dims = X.shape[-1]
  should_broadcast_x = len(X.shape) == 2
  if should_broadcast_x:
    X = X[None, :, :]
  _check_larger_than_zero(batch_dimension, "batch_dimension")
  _check_larger_than_zero(num_seeds, "num_seeds")
  _check_larger_than_zero(num_dims, "num_dims")
  _check_larger_than_zero(jitter, "jitter")
  if mean is None:
    mean = np.zeros([num_dims], dtype=X.dtype)
  w = _handle_singularity_at_zero(w=w, jitter=jitter)
  alpha = np.sum(w, axis=1)
  beta = np.sum(w ** 2, axis=1)
  weighted_sum = np.einsum('bij,bi->bj', X, w) # y = X @ w
  alpha_broadcasted, beta_broadcasted = alpha[:, None], beta[:, None]
  mean_broadcasted = mean[None, :]
  transformed_samples = (
    (1 - alpha_broadcasted / np.sqrt(beta_broadcasted)) * mean_broadcasted +
    weighted_sum / np.sqrt(beta_broadcasted)
  )
  return transformed_samples

def _handle_singularity_at_zero(
  w: Float[np.ndarray, "batch_dimension num_seeds"],
  jitter: float
) -> Float[np.ndarray, "batch_dimension"]:
  """
  If all the weights are zero, the transformed samples will be zero. This singularity we can avoid
  by treating the weights as uniform, which semantically they are.
  :param w: Weights of the linear combination.
  :param jitter: Small value defining the smallest value denominator beta can take.
  :return: Weights with the singularity handled.
  """
  assert jitter > 0, "Jitter must be larger than zero."
  _, num_seeds = w.shape
  are_the_weights_all_zero = np.sum(w ** 2, axis=1) < jitter
  w = w.copy()
  # Setting all the weights (when they are all zero) to 1. This will be normalised by beta.
  w[are_the_weights_all_zero] = 1.
  return w

@jaxtyped(typechecker=beartype)
def lol_iid(
  w: Float[np.ndarray, "batch_dimension num_seeds"],
  X: Union[
    Float[np.ndarray, "num_seeds num_dims"],
    Float[np.ndarray, "batch_dimension num_seeds num_dims"]
  ],
  cdf: Callable[[np.ndarray], np.ndarray],
  inverse_cdf: Callable[[np.ndarray], np.ndarray],
  jitter = 1e-10
) -> Float[np.ndarray, "batch_dimension num_dims"]:
  """
  Transforms linear combinations of iid samples from a distribution such that it produces samples
  from the same distribution.

  Assumes that the distribution is independent across dimensions and that each dimension (element)
  follows the same distribution. If they do not, use the lol_scalar function directly instead
  parameterised by the respective cdf and inverse_cdf functions.

  The transform minimises the quadratic Wasserstein distance between the original distribution
  of the linear combination and the target distribution (the distribution of the seeds).

  :param w: Weights of the linear combination.
  :param X: Seed samples.
  :param cdf: CDF of the distribution the seeds are sampled from.
  :param inverse_cdf: Inverse CDF of the distribution the seeds are sampled from.
  :return: Samples from the same distribution as the seeds.
  """
  batch_dimension, num_seeds = w.shape
  num_dims = X.shape[-1]
  _check_larger_than_zero(batch_dimension, "batch_dimension")
  _check_larger_than_zero(num_seeds, "num_seeds")
  _check_larger_than_zero(num_dims, "num_dims")
  _check_larger_than_zero(jitter, "jitter")
  unit_uniform_seed_samples = cdf(X)
  unit_gaussian_seed_samples = norm.ppf(unit_uniform_seed_samples)
  unit_gaussian_samples = lol_gaussian(w=w, X=unit_gaussian_seed_samples, jitter=jitter)
  unit_uniform_samples = norm.cdf(unit_gaussian_samples)
  samples = inverse_cdf(unit_uniform_samples)
  return samples

@jaxtyped(typechecker=beartype)
def lol_scalar(
  w: Float[np.ndarray, "batch_dimension num_seeds"],
  X: Union[
    Float[np.ndarray, "num_seeds"],
    Float[np.ndarray, "batch_dimension num_seeds"]
  ],
  cdf: Callable[[np.ndarray], np.ndarray],
  inverse_cdf: Callable[[np.ndarray], np.ndarray],
  jitter = 1e-10
) -> Float[np.ndarray, "batch_dimension"]:
  """
  Transforms linear combinations of iid samples from a distribution such that it produces samples
  from the same distribution.

  The transform minimises the quadratic Wasserstein distance between the original distribution
  of the linear combination and the target distribution (the distribution of the seeds).

  :param w: Weights of the linear combination.
  :param X: Seed samples.
  :param cdf: CDF of the distribution the seeds are sampled from.
  :param inverse_cdf: Inverse CDF of the distribution the seeds are sampled from.
  :return: Samples from the same distribution as the seeds.
  """
  batch_dimension, num_seeds = w.shape
  _check_larger_than_zero(batch_dimension, "batch_dimension")
  _check_larger_than_zero(num_seeds, "num_seeds")
  _check_larger_than_zero(jitter, "jitter")
  unit_uniform_seed_samples = cdf(X)
  unit_gaussian_seed_samples = norm.ppf(unit_uniform_seed_samples)
  unit_gaussian_samples = lol_gaussian(
    w=w,
    X=np.expand_dims(unit_gaussian_seed_samples, axis=-1),
    jitter=jitter
  ).squeeze(axis=-1)
  unit_uniform_samples = norm.cdf(unit_gaussian_samples)
  samples = inverse_cdf(unit_uniform_samples)
  return samples

def _check_larger_than_zero(value, name):
  if value <= 0:
    raise ValueError(f"{name} must be larger than zero.")