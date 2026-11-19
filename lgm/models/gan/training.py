"""Implementation of several lightning modules for training GANs."""
from abc import ABC, abstractmethod
from typing import Literal, override

import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import nn

from lgm.data import UniformDequantize
from lgm.models import TrainingModuleBase
from lgm.optimizers import OptimizationConfig
from lgm.types import DataBatchFloat, LabelBatch, ScalarFloat
from .model import Discriminator, GAN


class GANTrainingModuleBase(TrainingModuleBase[GAN], ABC):
    """Base lightning module for GANs.
    
    This is not functional, as it does not implement `core_step()`. There are in fact various ways
    to implement GAN training, from the "classic" alternating training of D and G, to a more
    optimized "parallel" version that avoids having to call both models twice, as well as variants
    in-between. Different variants may handle the "divison of labor" between `core_step()` and
    `train_step()` differently. As such, some override the standard `train_step()`, while others
    only implement `core_step()`. However, they all use so-called "manual optimization", meaning
    that computing gradients, as well as applying the optimizer and scheduler, must be done
    manually.  
    It is recommended to add a `GenerationLogging` callback for subjective evaluation of sample
    quality.
    """
    @override
    def __init__(self,
                 model: GAN,
                 optimization_config: OptimizationConfig,
                 d_loss_fn: nn.Module,
                 feature_matching: float = 0.,
                 label_smoothing: float = 0.):
        """Configure the module.

        Parameters:
            model: `GAN` instance.
            optimization_config: See `TrainingModuleBase`.
            d_loss_fn: Discriminator loss function. Typical choices include `BCEWithLogitsLoss()`
                for the "classic" GAN (this will implement the "flipped" loss with better generator
                gradients), or `nn.MSELoss()` for (one version of) LS-GAN.
            feature_matching: Coefficient for feature matching loss added to standard generator
                loss. 0 turns off feature matching.
            label_smoothing: Pass value > 0 to turn on one-sided label smoothing. This moves labels
                away from 1, e.g. smoothing of 0.1 uses labels of 0.9. This *only* affects the true
                labels for discriminator training. Neither the fake labels, nor the "flipped" 1
                labels for generator training, are affected. Only makes sense when using standard
                cross-entropy loss for D; it's supposed to reduce saturation in the loss function
                here.
        """
        super().__init__(model=model,
                         optimization_config=optimization_config)
        self.loss_fn = d_loss_fn
        self.feature_matching = feature_matching
        self.label_mult = 1 - label_smoothing
        self.dequantizer = UniformDequantize()
        self.automatic_optimization = False

    @abstractmethod
    def core_step(self,
                  tag: Literal["train", "val", "test"],
                  batch: tuple[DataBatchFloat, LabelBatch],
                  batch_index: int):
        """Compute G and D classification losses, as well as feature matching loss.

        Parameters:
            See: `TrainingModuleBase`.

        Returns:
            Depending on the subclass, scalar D, G losses for the batch, or nothing.
        """


class GANTrainingModuleParallel(GANTrainingModuleBase):
    """Parallel GAN step training D and G at the same time.
    
    This requires running G and D only once to update both, compared to the classic sequential step,
    which runs both models twice. On the flipside, it can have higher peak memory requirements, as
    graphs for the entire process have to be kept for a longer time. Still, in practice, we found
    little difference in this regard, while the speedup can be noticeable.

    Note that this technically results in different training dynamics from the sequential step. In
    the simplest way, you could interpret this one as G being trained with respect to D being one
    step "behind", compared to the sequential step, as we don't update D first in the parallel one.
    Still, there may be more complex consequences to this seemingly simple shift. In practice, we
    found both versions to perform similarly.
    """
    @override
    def core_step(self,
                  tag: Literal["train", "val", "test"],
                  batch: tuple[DataBatchFloat, LabelBatch],
                  batch_index: int) -> tuple[ScalarFloat, ScalarFloat]:
        input_batch, _ = batch

        # bonus question, compare to the sequential module below. why are we NOT using no_grad here?
        generated_batch = self.model.generate(input_batch.shape[0])
        generated_batch = self.dequantizer(generated_batch)
        input_batch = self.dequantizer(input_batch)

        batch_size = input_batch.shape[0]  # TODO separate real, fake batch size
        full_batch = torch.cat((input_batch, generated_batch))
        d_output, features = self.model.discriminator(full_batch)

        real_labels = torch.full((batch_size, 1), self.label_mult, device=input_batch.device)
        fake_labels = torch.zeros((batch_size, 1), device=input_batch.device)
        full_labels = torch.cat((real_labels, fake_labels))
        fool_labels = torch.ones(batch_size, 1, device=input_batch.device)

        d_loss = self.loss_fn(d_output, full_labels)
        g_loss = self.loss_fn(d_output[batch_size:], fool_labels)
        metrics = {"discriminator_loss": d_loss,
                   "generator_loss": g_loss}

        if self.feature_matching:
            feature_match_loss = 0
            features_flat = [feature for level_feature in features for feature in level_feature]
            for feature in features_flat:
                real_feature, fake_feature = torch.chunk(feature, 2, dim=0)
                feature_difference = real_feature.mean(dim=0) - fake_feature.mean(dim=0)
                feature_match_loss += (feature_difference**2).mean()
            generator_loss_full = g_loss + self.feature_matching * feature_match_loss
            metrics["feature_matching_loss"] = feature_match_loss
        else:
            generator_loss_full = g_loss

        self.log_metrics(tag, metrics)

        return d_loss, generator_loss_full

    @override
    def training_step(self,
                      batch: tuple[DataBatchFloat, LabelBatch],
                      batch_index: int):
        d_loss, g_loss = self.core_step("train", batch, batch_index)

        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        optimizer.zero_grad()
        self.manual_backward(d_loss, inputs=list(self.model.discriminator.parameters()),
                             retain_graph=True)
        self.manual_backward(g_loss, inputs=list(self.model.generator.parameters()))
        optimizer.step()
        scheduler.step()


class GANTrainingModuleSequential(GANTrainingModuleBase):
    """Classic GAN step training D and G in an alternating fashion.

    This first trains D by generating a batch of data, computing the classification loss, and
    finally updating the weights.  
    Afterwards, a *new* batch is generated for training G, using the updated D, i.e. D also has to
    be run again. In particular, for feature matching, the real data *also* has to be put through
    the updated D again.
    
    TODO refactor to use proper `training_step` rather than relying on `tag`?
    Problem with that would be that we can no longer rely on a simple `core_step`, as D training
    happens in the middle of it.
    """
    @override
    def core_step(self,
                  tag: Literal["train", "val", "test"],
                  batch: tuple[DataBatchFloat, LabelBatch],
                  batch_index: int):
        input_batch, _ = batch
        batch_size = input_batch.shape[0]
        d_optimizer, g_optimizer = self.optimizers()
        d_scheduler, g_scheduler = self.lr_schedulers()

        # train D
        # don't need to toggle_optimizer since we wrap G in no_grad
        with torch.no_grad():
            # bonus question, why are we using no_grad?
            generated_batch = None
            generated_batch = self.dequantizer(generated_batch)
        input_batch = self.dequantizer(input_batch)
        
        full_batch = torch.cat((input_batch, generated_batch))
        d_output, _ = self.model.discriminator(full_batch)
        real_labels = None
        fake_labels = None
        full_labels = torch.cat((real_labels, fake_labels))
        d_loss = self.loss_fn(d_output, full_labels)

        if tag == "train":
            d_optimizer.zero_grad()
            self.manual_backward(d_loss)
            d_optimizer.step()
            d_scheduler.step()

        # train G
        self.toggle_optimizer(g_optimizer)
        generated_batch2 = self.generate(batch_size)
        generated_batch2 = self.dequantizer(generated_batch2)
        full_batch2 = torch.cat((input_batch, generated_batch2))
        d_output2, features = self.model.discriminator(full_batch2)
        fool_labels = None

        g_loss = self.loss_fn(d_output2[batch_size:], fool_labels)
        metrics = {"discriminator_loss": d_loss,
                   "generator_loss": g_loss}

        if self.feature_matching:
            feature_match_loss = 0
            features_flat = [feature for level_feature in features for feature in level_feature]
            for feature in features_flat:
                real_feature, fake_feature = torch.chunk(feature, 2, dim=0)
                feature_difference = None
                feature_match_loss += (feature_difference**2).mean()
            generator_loss_full = g_loss + self.feature_matching * feature_match_loss
            metrics["feature_matching_loss"] = feature_match_loss
        else:
            generator_loss_full = g_loss

        if tag == "train":
            pass
        self.untoggle_optimizer(g_optimizer)
        self.log_metrics(tag, metrics)
    
    @override
    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer_d = self.optimization_config.optimizer.build(self.model.discriminator.parameters())
        optimizer_g = self.optimization_config.optimizer.build(self.model.generator.parameters())

        scheduler_d = self.optimization_config.scheduler.build(
            optimizer_d,
            self.trainer.estimated_stepping_batches)
        scheduler_g = self.optimization_config.scheduler.build(
            optimizer_g,
            self.trainer.estimated_stepping_batches)

        config_d = {"scheduler": scheduler_d,
                    "interval": "step",
                    "frequency": 1}
        config_g = {"scheduler": scheduler_g,
                    "interval": "step",
                    "frequency": 1}
        
        return [optimizer_d, optimizer_g], [config_d, config_g]
