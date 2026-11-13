"""Functionalities for additional investigations into GANs."""
import torch

from lgm.ops import interleave, sum_except
from lgm.types import DataBatchFloat, TabularBatchFloat
from lgm.visualization import ImagePlotter
from .model import Generator


def optimization_inference(target_data: DataBatchFloat,
                           generator: Generator,
                           n_steps: int,
                           plotter: ImagePlotter,
                           regularization: float = 0.) -> TabularBatchFloat:
    """Do inference via optimization for GANs.
    
    This means that given some target data, we try to find points in the latent space that, once
    generated from, match the target data as much as possible. This is not guaranteed to actually
    find good matches for all inputs, and in fact will often result in garbage.

    Parameters:
        target_data: Batch of inputs we want to approximate.
        generator: GAN generator.
        n_steps: Number of gradient steps we take for optimization.
        plotter: `ImagePlotter` object to plot the optimized images.
        regularization: There is no guarantee that the optimized noise actually represents good
            samples from the noise function. If the true noise is standard normal, we can infer the
            expected norm of a sample. Using this, we can quantify the difference between that norm
            and the actual norm of the optimized noise, and use that as a regularizer.  
            Note that this is only valid for standard normal noise functions. Also, this can
            severely reduce the chance of finding a good match in the noise space.
    """
    optimized_noise = generator.sample_noise(target_data.shape[0])
    optimized_noise.requires_grad = True
    noise_optimizer = torch.optim.Adam([optimized_noise], lr=1.)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(noise_optimizer, n_steps)

    noise_dim = torch.prod(torch.tensor(optimized_noise.shape[1:]))
    non_batch_data_dims = tuple(range(1, target_data.dim()))

    # note, as we are optimizing batched variables directly, we sum over batch axis.
    # otherwise, if we averaged instead, larger batches would reduce the per-element gradient.
    for step in range(n_steps + 1):
        outputs = generator(optimized_noise)
        diff = (outputs - target_data).abs().mean(dim=non_batch_data_dims).sum()
        ideal_norm = torch.sqrt(noise_dim)
        actual_norms = torch.sqrt(sum_except(optimized_noise**2))
        diff = diff + regularization * ((ideal_norm - actual_norms)**2).sum()

        diff.backward(inputs=[optimized_noise])
        noise_optimizer.step()
        noise_optimizer.zero_grad()
        schedule.step()

        if not step % (n_steps // 10):
            n_rows = 8  # TODO don't harcode
            combined = interleave(target_data[:n_rows**2 // 2], outputs[:n_rows**2 // 2])
            plotter.image_grid(combined,
                               title=f"After Step {step + 1}, diff {diff.item():.5f}",
                               n_rows=n_rows)
    return optimized_noise.detach()
