# Linear combinations of latents in generative models: subspaces and beyond

This repository provides an implementation of **LOL** (Linear combinations Of Latent variables), as presented in the paper:

[**Linear Combinations of Latents in Generative Models: Subspaces and Beyond**](https://arxiv.org/pdf/2408.08558)

![Alt text](images/cars_subspace.png)

## Abstract
Sampling from generative models has become a crucial tool for applications like data synthesis and augmentation. 
Diffusion, Flow Matching and Continuous Normalizing Flows have shown effectiveness across various modalities, 
and rely on latent variables for generation. 
For experimental design or creative applications that require more control over the generation process, 
it has become common to manipulate the latent variable directly. 
However, existing approaches for performing such manipulations (e.g. interpolation or forming low-dimensional representations) 
only work well in special cases or are network or data-modality specific. 
We propose Linear combinations of Latent variables (LOL) as a general-purpose method to form linear combinations of latent 
variables that adhere to the assumptions of the generative model. As LOL is easy to implement and naturally addresses 
the broader task of forming any linear combinations, e.g. the construction of subspaces of the latent space, 
LOL dramatically simplifies the creation of expressive low-dimensional representations of high-dimensional objects.


## Installation
```sh
pip install lolatents
```

## Usage
To see the LOL method in use, please see the provided example notebooks:

- **[`example_interpolation.ipynb`](example_interpolation.ipynb)** – Demonstrates simple interpolation.
- **[`example_subspace.ipynb`](example_subspace.ipynb)** – Illustrates how latent subspaces can be defined. 
- **[`example_grid.ipynb`](example_subspace.ipynb)** – Similar as above but directly using linear combination weights.

```sh
from lolatents.lol_numpy import lol_iid
# from lolatents.lol_torch import lol_iid # for PyTorch

latents = lol_iid(
  w=w_grid_including_remaining,
  X=seed_latents,
  cdf=latent_distribution.cdf,
  inverse_cdf=latent_distribution.ppf
)
```

