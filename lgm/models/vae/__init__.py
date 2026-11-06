"""This module contains functionalities for variational autoencoders.

This implementation puts completeness/correctness over simplicity. We provide a "general"
implementation that closely maps onto the theoretical framework, and the most common form with
Gaussian prior/posterior as a special case. This is important: `VAEGauss` is **not** "the" VAE. It's
just one specific implementation!

REFERENCES
Original paper: https://arxiv.org/abs/1312.6114
beta-VAE: https://openreview.net/pdf?id=Sy2fzU9gl
sigma-VAE: https://arxiv.org/abs/2006.13202
Continuous Bernoulli (bad :'<): https://arxiv.org/abs/1907.06845
"""
from .fun import check_code_usage, per_dim_latent_walk, show_latent_distributions
from .model import GaussianPosterior, VAE, VAEGauss
from .training import VAETrainingModule
