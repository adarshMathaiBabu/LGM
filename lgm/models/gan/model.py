"""GAN Network implementation.

Generator and Discriminator are implemented based on encoder/decoder networks. This means you can
actually compare GANs quite nicely to VAEs -- both have an encoder/decoder component, and by
matching the architectures as much as possible, you can see the difference that the *framework*
makes for the generated samples, rather than architectural details.

We also have a `GAN` class that simply acts as a wrapper handling a generator and a discriminator.
This means we can pass a single `model` object to the lightning module, and so we can re-use our
`TrainingModuleBase` yet again.
"""
from torch import nn
from torch.distributions import Distribution

from lgm.layers import DecoderConfig, DecoderNet, EncoderConfig, EncoderNet
from lgm.types import DataBatchFloat, TabularBatchFloat


class Generator(DecoderNet):
    """GAN generator that is actually a thin wrapper around `DecoderNet`."""
    def __init__(self,
                 config: DecoderConfig,
                 noise_distribution: Distribution):
        """Configure the network.

        Parameters:
            config: `DecoderConfig` instance to configure the network.
            noise_distribution: `torch` distribution object to sample latent codes from. Should be
                univariate, as we explicitly sample the correct number of units. If the
                distribution is multivariate, this would result in too many units/dimensions being
                sampled. Strictly speaking, this is a limitation, as it does not allow for sampling
                different dimensions with different distributions, which would be possible in
                principle. However, things like that are very rarely done, so we will be fine with
                using the same distribution for all dimensions.  
                Note that, since training does not need to backpropagate through
                the sampling process, pretty much anything is permissible here. You could even try
                things like Bernoulli noise (random binary numbers).
        """
        super().__init__(config)
        self.noise_distribution = noise_distribution

    def generate(self,
                 n_samples: int) -> DataBatchFloat:
        return self(self.sample_noise(n_samples))
        

    def sample_noise(self,
                     n_samples: int) -> TabularBatchFloat:
        """Shortcut to sample from noise distribution on correct device."""
        device = next(self.parameters()).device
        return self.noise_distribution.sample((n_samples, self.latent_units)).to(device)


class Discriminator(EncoderNet):
    def __init__(self,
                 config: EncoderConfig,
                 sd_layer: bool = False):
        """GAN discriminator returning hidden features (for feature matching).
        
        This is just a wrapper around `EncoderNet`, but we require `return_all=True` to allow for
        feature matching, and a single output for binary classification.

        Parameters:
            config: `EncoderConfig` instance to configure the network.  
                Note that `out_dim` **must** be 1, as the discriminator performs binary
                classification.  
                Also, `return_all` **must** be `True` for feature matching.  
                TODO forcibly override and warn?
            sd_layer: If `True`, add a "Standard deviation layer" as proposed for Progressive
                Growing GAN right at the start of the head.
        """
        if config.out_dim != 1:
            raise ValueError(f"Provided out_dim {config.out_dim}, but GAN discriminator requires a "
                             "value of 1!")
        if not config.return_all:
            raise ValueError("Discriminator requires return_all to be True for feature matching. "
                             "Please set this in the config parameter!")
        
        super().__init__(config, sd_layer=sd_layer)


class GAN(nn.Module):
    """Wrapper class for Discriminator and Generator."""
    def __init__(self,
                 discriminator: Discriminator,
                 generator: Generator):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator

    def generate(self,
                 n_samples: int) -> DataBatchFloat:
        """Wrap `generator.generate()` for convenience."""
        return self.generator.generate(n_samples)
