"""This module implements the most well-known VAE variant with Gaussian posterior & prior."""
from typing import ClassVar, override

import torch
from torch import nn
from torch.distributions import Distribution, Normal

from lgm.layers import EncoderDecoderConfig
from lgm.models.autoencoder import Likelihood
from lgm.ops import sum_except
from lgm.types import ScalarFloat, TabularBatchFloat
from .base import PosteriorFamily, VAE


class VAEGauss(VAE):
    """Wrapper for VAE with Gaussian posterior and Standard Gaussian prior."""
    @override
    def __init__(self,
                 likelihood: Likelihood,
                 enc_dec_config: EncoderDecoderConfig,
                 latent_zero_init: bool = False):
        """Configure the module.

        Parameters:
            latent_zero_init: If `True`, final weight layer of the encoder is initialized with 0
                              weights and bias. This causes means and log variances to be 0, leading
                              to perfect standard Gaussian posteriors at initialization. This can
                              sometimes stabilize otherwise unstable training early on.
            Remaining: See `lgm.models.autoencoder.Autoencoder`.
        """
        super().__init__(likelihood=likelihood,
                         enc_dec_config=enc_dec_config,
                         posterior=GaussianPosterior())

        if latent_zero_init:
            if self.encoder.head_style == "conv":
                layer_index = 0
            else:
                layer_index = -1
            final_layer = self.encoder.head[layer_index]
            nn.init.zeros_(final_layer.weight)
            nn.init.zeros_(final_layer.bias)

    def kl_divergence(self,
                      parameters: tuple[TabularBatchFloat, TabularBatchFloat],
                      samples_per_q: int = 1) -> ScalarFloat:
        """Closed-form KL-Divergence for Gaussian posterior and Standard Gaussian prior.
        
        Note that `samples_per_q` is ignored, as there is no sampling here. The argument is included
        to keep the interface consistent with the general version.
        """
        means, log_variances = parameters
        raise NotImplementedError("For the Gaussian VAE, there is a special case for the "
            "KL-divergence! See the notebook for the formula. Be mindful that we get the "
            "log-variances above, and you need the actual variances (without log), as well. Be sure "
            "to correctly *sum* over data dimensions, but *average* over the batch dimensions. "
            "The sum_except function (already imported above!) should be helpful here.")
        # NOTE in case you are struggling with this one, you can leave it blank, and simply use
        # the sample-based approximate KL divergence from the base class.
        return 0.5 * sum_except(means**2 - 1 + torch.exp(log_variances) - log_variances).mean()


class GaussianPosterior(PosteriorFamily):
    """Gaussian posterior family.
    
    We parameterize distributions using mean and log variance. The encoder thus needs to return two
    values per latent dimension, and raw latent outputs should be unconstrained real numbers.
    """
    n_parameters: ClassVar[int] = 2

    @override
    def from_parameters(self,
                        parameters) -> Distribution:
        mean, log_variance = parameters
        return Normal(mean, torch.exp(0.5 * log_variance))

    @override
    def reparameterized_sample(self,
                               parameters,
                               samples_per_q: int = 1) -> TabularBatchFloat:
        mean, log_variance = parameters
        stddev = torch.exp(0.5 * log_variance)
        if samples_per_q > 1:
            mean = torch.repeat_interleave(mean, samples_per_q, dim=0)
            stddev = torch.repeat_interleave(stddev, samples_per_q, dim=0)

        raise NotImplementedError("How do we do the 'reparameterization trick' for a Normal "
                                  "distribution?")
        return None

    @override
    def get_prior(self) -> Distribution:
        return Normal(loc=0., scale=1.)
