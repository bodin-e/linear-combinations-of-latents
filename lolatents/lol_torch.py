
import torch
from torch.distributions import Normal
from typing import Callable, Union
from jaxtyping import Float, jaxtyped
from beartype import beartype

@jaxtyped(typechecker=beartype)
def lol_iid(
  w: Float[torch.Tensor, "batch_dimension num_seeds"],
  X: Union[
    Float[torch.Tensor, "num_seeds num_dims"],
    Float[torch.Tensor, "batch_dimension num_seeds num_dims"]
  ],
  cdf: Callable[[torch.Tensor], torch.Tensor],
  inverse_cdf: Callable[[torch.Tensor], torch.Tensor],
  jitter: float = 1e-10
) -> Float[torch.Tensor, "batch_dimension num_dims"]:
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
  normal = Normal(0.0, 1.0)
  unit_uniform_seed_samples = cdf(X)
  unit_gaussian_seed_samples = normal.icdf(unit_uniform_seed_samples)
  unit_gaussian_samples = lol_gaussian(w=w, X=unit_gaussian_seed_samples, jitter=jitter)
  unit_uniform_samples = normal.cdf(unit_gaussian_samples)
  samples = inverse_cdf(unit_uniform_samples)
  return samples

@jaxtyped(typechecker=beartype)
def lol_gaussian(
  w: Float[torch.Tensor, "batch_dimension num_seeds"],
  X: Union[
    Float[torch.Tensor, "num_seeds num_dims"],
    Float[torch.Tensor, "batch_dimension num_seeds num_dims"]
  ],
  mean: Float[torch.Tensor, "num_dims"] = None,
  jitter: float = 1e-10
) -> Float[torch.Tensor, "batch_dimension num_dims"]:
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
  should_broadcast_x = X.dim() == 2
  if should_broadcast_x:
    X = X.unsqueeze(0)
  _check_larger_than_zero(batch_dimension, "batch_dimension")
  _check_larger_than_zero(num_seeds, "num_seeds")
  _check_larger_than_zero(num_dims, "num_dims")
  _check_larger_than_zero(jitter, "jitter")
  if mean is None:
    mean = torch.zeros(num_dims, dtype=X.dtype, device=X.device)
  w = _handle_singularity_at_zero(w=w, jitter=jitter)
  alpha = torch.sum(w, dim=1)
  beta = torch.sum(w ** 2, dim=1)
  weighted_sum = torch.sum(X * w.unsqueeze(-1), dim=1)
  alpha_broadcasted = alpha[:, None]
  beta_broadcasted = beta[:, None]
  mean_broadcasted = mean[None, :]
  transformed_samples = (
    (1 - alpha_broadcasted / torch.sqrt(beta_broadcasted)) * mean_broadcasted +
    weighted_sum / torch.sqrt(beta_broadcasted)
  )
  return transformed_samples

@jaxtyped(typechecker=beartype)
def _handle_singularity_at_zero(
  w: Float[torch.Tensor, "batch_dimension num_seeds"],
  jitter: float
) -> Float[torch.Tensor, "batch_dimension num_seeds"]:
  """
  If all the weights are zero, the transformed samples will be zero. This singularity we can avoid
  by treating the weights as uniform, which semantically they are.
  :param w: Weights of the linear combination.
  :param jitter: Small value defining the smallest value denominator beta can take.
  :return: Weights with the singularity handled.
  """
  _check_larger_than_zero(jitter, "jitter")
  beta = torch.sum(w ** 2, dim=1)
  zero_mask = beta < jitter
  w = w.clone()
  w[zero_mask] = 1.0
  return w

@jaxtyped(typechecker=beartype)
def lol_scalar(
  w: Float[torch.Tensor, "batch_dimension num_seeds"],
  X: Union[
    Float[torch.Tensor, "num_seeds"],
    Float[torch.Tensor, "batch_dimension num_seeds"]
  ],
  cdf: Callable[[torch.Tensor], torch.Tensor],
  inverse_cdf: Callable[[torch.Tensor], torch.Tensor],
  jitter: float = 1e-10
) -> Float[torch.Tensor, "batch_dimension"]:
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
  normal = Normal(0.0, 1.0)
  unit_uniform_seed_samples = cdf(X)
  unit_gaussian_seed_samples = normal.icdf(unit_uniform_seed_samples)
  unit_gaussian_samples = lol_gaussian(
    w=w,
    X=unit_gaussian_seed_samples.unsqueeze(-1),
    jitter=jitter
  ).squeeze(-1)
  unit_uniform_samples = normal.cdf(unit_gaussian_samples)
  samples = inverse_cdf(unit_uniform_samples)
  return samples

def _check_larger_than_zero(value, name: str):
  if value <= 0:
    raise ValueError(f"{name} must be larger than zero.")
