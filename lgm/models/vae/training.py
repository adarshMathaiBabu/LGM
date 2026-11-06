"""This module implements a lightning module for training a variational autoencoder."""
from collections.abc import Callable
from typing import Literal, override

import torch

from lgm.models.autoencoder.training import AETrainingModuleBase
from lgm.optimizers import OptimizationConfig
from lgm.types import DataBatchFloat, LabelBatch, ScalarFloat
from lgm.utils import rescale_likelihood
from .model import VAE


class VAETrainingModule(AETrainingModuleBase[VAE]):
    """Lightning module for training a VAE.
    
    We minimize a loss corresponding to maximizing the ELBO, a lower bound on the log-likelihood.
    In practice, this combines the typical autoencoder reconstruction loss with a KL-divergence.
    It's crucial that the reconstruction loss represents a proper likelihood on the data. This is
    why we already practiced this with regular autoencoders! Without a proper likelihood, you're
    training something, but it's *not* a VAE.  
    We also implement the beta-VAE framework, which amounts to simply multiplying the KL-loss by
    `beta`. This is usually chosen `> 1`, but in principle there is nothing stopping you from using
    numbers below 1. You should definitely experiment with this, as finding a good value here is
    usually required to get the best possible model.  
    Finally, we also implement taking multiple "samples per q". Keep in mind that we are technically
    approximating expected values over q(z|x) in both the reconstruction loss and KL-divergence. The
    standard approach is to take only a single sample per input. However, it is technically valid to
    take multiple samples each, which may improve the approximation, thus reducing variance and
    possibly speeding up training in terms of steps needed to converge. However, it also massively
    increases time taken per step, as we are effectively multiplying the batch size for the decoder.
    In practice, this is rarely worth it, but we include it for completion's sake.
    """
    @override
    def __init__(self,
                 model: VAE,
                 optimization_config: OptimizationConfig,
                 scale_multiplier: float = 256.,
                 reconstruction_loss_fn: Callable[[DataBatchFloat,
                                                   DataBatchFloat],
                                                   ScalarFloat] | None = None,
                 beta: float = 1.,
                 samples_per_q: int = 1):
        """Configure the module.

        Parameters:
            model: `VAE` instance.
            beta: KL-Loss weight for Beta-VAE.
            samples_per_q: How many samples to take from each q(z|x). See class docs for more
                details.
            Remaining: See `lgm.models.autoencoder.Autoencoder`.
        """
        super().__init__(model=model,
                         optimization_config=optimization_config,
                         scale_multiplier=scale_multiplier,
                         reconstruction_loss_fn=reconstruction_loss_fn)
        self.beta = beta
        self.samples_per_q = samples_per_q

    @override
    def core_step(self,
                  tag: Literal["train", "val", "test"],
                  batch: tuple[DataBatchFloat, LabelBatch],
                  batch_index: int) -> ScalarFloat:
        """Compute reconstructions and corresponding loss, as well as KL-divergence.

        Parameters:
            See: `TrainingModuleBase`.

        Returns:
            Scalar loss for the batch -- reconstruction loss plus weighted KL-divergence.
        """
        input_batch, _ = batch
        parameters = self.model.encoding_parameters(input_batch)
        raise NotImplementedError("What do we put in the decoder? It should be *samples* from the "
            "posterior q(z|x). Note that the self.model object (the VAE) has a posterior attribute. "
            "This one might have the correct function to do the sampling :)")
        output_batch = self.model.decoder(None)

        if self.samples_per_q > 1:
            # replicate target data, since we get multiple reconstructions for each input
            input_batch = torch.repeat_interleave(input_batch, self.samples_per_q, dim=0)


        reconstruction_loss = self.reconstruction_loss_fn(output_batch, input_batch)
        kl_loss = self.model.kl_divergence(parameters, self.samples_per_q)
        raise NotImplementedError("What is the full VAE loss, based on reconstruction loss and KL"
            "divergence? Remember to also implement the beta-VAE framework, as specified in the"
            "assignment text.")
        full_loss = None

        # the below is just for show :) the loss above is sufficient for training!
        # it just tends to be a more interpretable rescaling of the loss above
        with torch.inference_mode():
            neg_elbo = self.model.likelihood.nll(output_batch, input_batch) + kl_loss
            neg_elbo = rescale_likelihood(neg_elbo,
                                          input_batch,
                                          self.scale_multiplier,
                                          self.trainer.datamodule.normalizer.log_inverse_jacobian)

        self.log_metrics(tag,
                         {"reconstruction_loss": reconstruction_loss,
                          "kl_loss": kl_loss,
                          "vae_loss": full_loss,
                          "negative_elbo": neg_elbo})
        return full_loss

    @override
    def generate(self,
                 n_samples: int) -> DataBatchFloat:
        # must override this since we are inheriting from autoencoder, and that one overrides it to
        # be non-functional.
        return self.model.generate(n_samples)
