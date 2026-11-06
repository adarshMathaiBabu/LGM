"""This module implements a generic VAE model.

This is in itself fully functional, although more specialized versions (e.g. with analytic KL
divergence) are possible.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, override

import torch
from torch.distributions import Distribution

from lgm.layers import EncoderDecoderConfig
from lgm.models.autoencoder import Autoencoder, Likelihood
from lgm.ops import sum_except
from lgm.types import DataBatchFloat, ScalarFloat, TabularBatchFloat


class VAE(Autoencoder):
    """Wrapper for VAE with arbitrary posterior and prior."""
    @override
    def __init__(self,
                 likelihood: Likelihood,
                 enc_dec_config: EncoderDecoderConfig,
                 posterior: PosteriorFamily,
                 prior: Distribution | None = None):
        """Configure the module.

        Parameters:
            posterior: Represents the family for the posterior distribution. We cannot create a
                single `Distribution` instance here, as the parameters depend dynamically on the
                encoder outputs.
            prior: `torch.Distribution` instance representing `p(z)`, i.e. the prior distribution on
                the latent space. If not given, will get it from the `posterior.get_prior()` method.
                Note, this is expected to be a univariate distribution, e.g.
                `Normal(loc=0., scale=1.)`, *not* a multivariate one like `MultivariateNormal`. This
                slightly reduces our flexibility by having all dimensions be the same distribution,
                as well as imposing independence between dimensions, but that's okay. If `None`,
                the prior is chosen from the posterior family.
            Remaining: See `lgm.models.autoencoder.Autoencoder`.
        """
        super().__init__(likelihood=likelihood,
                         enc_dec_config=enc_dec_config,
                         latent_multiplier=posterior.n_parameters)
        if prior is None:
            prior = posterior.get_prior()
        self.posterior = posterior
        self.prior = prior
        self.prior_dim = self.decoder.latent_units

    @override
    def forward(self,
                inputs: DataBatchFloat,
                map_to_expected: bool = False) -> DataBatchFloat:
        """Apply the VAE.

        Note that this call is *stochastic* in that we take samples from the encoder distribution.
        Each time you apply this method to the same inputs, you will get different results!
        
        Parameters:
            inputs: Batch of inputs.
            map_to_expected: If `True,` model outputs (distribution parameters) are mapped to
                expected values. Only with this flag will you get "true" reconstructions.

        Returns:
            Reconstruction parameters or expected values, depending on `map_to_expected`.
        """
        reconstructions = self.decoder(self.sample_codes(inputs))
        if map_to_expected:
            reconstructions = self.expected_value(reconstructions)
        return reconstructions

    def encoding_parameters(self,
                            inputs: DataBatchFloat) -> tuple[TabularBatchFloat, ...]:
        """Get encoding distribution parameters and split into corresponding parts.

        The encoder is applied, returning *all* parameters for the posterior distributions at each
        latent dimension. The output is then split into equal-sized parts along the feature axis,
        one per distribution parameter. For example, in the Gaussian posterior case, there would be
        two parts: Means and log variances (or however the variances are parameterized).
        
        Paramters:
            inputs: Batch of data space inputs.

        Returns:
            Tuple with one element per distribution parameter, each of shape
            `(batch, self.prior_dim)`.
        """
        raise NotImplementedError("Whoops! We forgot to implement the encoder. The autoencoder has "
            "a self.encoder attribute, so that could probably be of use here. However, it just "
            "returns a bunch of numbers, and we need to separate the different parameters, e.g. "
            "means and variances for a Gaussian q(z|x). Maybe we could SPLIT them somehow? Or turn "
            "the outputs into multiple CHUNKs? If only torch had functions for that... NOTE that "
            "the self.posterior object has an n_parameters field that could be useful to figure "
            "out how many chunks we need!")
        return None

    def sample_codes(self,
                     inputs: DataBatchFloat) -> TabularBatchFloat:
        """Shortcut for self.posterior.reparameterized_sample(self.encoding_parameters(inputs)).
        
        This combines getting posterior parameters (i.e. encoding) and sampling.
        """
        return self.posterior.reparameterized_sample(self.encoding_parameters(inputs))

    def kl_divergence(self,
                      parameters: tuple[TabularBatchFloat, ...],
                      samples_per_q: int = 1) -> ScalarFloat:
        """Generic KL divergence implementation.
        
        For VAE training, we require `E_q[log q - log p]`, i.e. the expected value over `q(z|x)`
        of the difference of posterior/prior log probabilities. This can be estimated via Monte
        Carlo sampling, just as we are averaging an expected value for the reconstruction loss.

        Note that we could simply re-use the samples we need to take anyway to put them into the
        decoder for the reconstruction loss. However, that would complicate the interface for this
        function, as this generic version requires samples (for computation) *and* parameters (to
        get the posterior), whereas more specific VAE versions (like `VAEGauss`) can compute the
        divergence based on parameters only. Thus we take new samples here just so that the function
        only requires `parameters` as input.
        """
        posterior_samples = self.posterior.reparameterized_sample(parameters, samples_per_q)
        posterior_distribution = self.posterior.from_parameters(parameters)
        return sum_except(posterior_distribution.log_prob(posterior_samples)
                          - self.prior.log_prob(posterior_samples)).mean()

    def generate(self, 
                 n_samples: int) -> DataBatchFloat:
        """Generate data samples via decoding samples from latent prior.
        
        Parameters:
            n_samples: Number of samples to generate.

        Returns:
            Batch of generated samples.
        """
        device = next(self.parameters()).device
        raise NotImplementedError("How do we sample from a VAE? I suppose we need to get samples "
            "from the prior and put them into the decoder... Note that the VAE has a self.prior_dim "
            "attribute to keep track of the latent space size!")
        codes = None
        return self.expected_value(self.decoder(codes))


class PosteriorFamily(ABC):
    """Parameterizes posterior distribution depending on given parameters.
    
    The parameters are usually returned by the VAE encoder.
    """
    n_parameters: ClassVar[int]

    @abstractmethod
    def from_parameters(self,
                        parameters: tuple[TabularBatchFloat, ...]) -> Distribution:
        """Create concrete posterior distribution conditioned on input `x` from parameters.
        
        Parameters:
            parameters: Distribution parameters. These should be as output by a
                `VAE.encoding_parameters()` call.

        Returns:
            `torch.Distribution` object for the posterior using the given parameters.
        """

    @abstractmethod
    def reparameterized_sample(self,
                               parameters: tuple[TabularBatchFloat, ...],
                               samples_per_q: int = 1) -> TabularBatchFloat:
        """Draw posterior samples using the reparameterization trick.
        
        This is crucial for keeping differentiability in the VAE.

        Parameters:
            parameters: As for the `from_parameters()` method.
            samples_per_q: Take this many samples from each distribution. If larger than 1, total
                batch size is increased by this factor, and output will first contain all samples
                for the first element in the `parameters` batch, then all samples for the second,
                etc.

        Returns:
            Batch of samples from `q(z|x)`.
        """

    @abstractmethod
    def get_prior(self) -> Distribution:
        """Get a corresponding simple prior distribution.
        
        For example, for a Gaussian posterior, this could be the Standard Gaussian.
        """
