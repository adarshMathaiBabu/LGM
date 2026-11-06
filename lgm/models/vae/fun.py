"""This module implements various utilities to investigate VAEs and their latent space."""
from collections.abc import Callable

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from .model import VAE, VAEGauss
from ...types import DataBatchFloat, ImageBatchFloat, LabelBatch, TabularBatchFloat, TabularFloat


def show_latent_distributions(model: VAE,
                              dataloader: DataLoader[tuple[DataBatchFloat, LabelBatch]],
                              n_rows: int,
                              figsize: tuple[int, int] = (12, 12)):
    """Display histograms of encoder output values over a dataset.

    Note that codes are sampled from the encocder output parameters, so there is some randomness
    involved.
    
    Parameters:
        model: Any VAE-type model that supports the respective encoding/sampling functions.
        dataloader: Provides the dataset to get distributions for.
        n_rows: How many rows to plot. You are responsible for making sure this divides the number
            of dimensions.
        figsize: Total size of the plot.
    """
    device = next(model.parameters()).device
    all_codes = []
    with torch.inference_mode():
        for input_batch, _ in dataloader:
            batch_encoded = model.sample_codes(input_batch.to(device))
            all_codes.append(batch_encoded)
    all_codes = torch.cat(all_codes, dim=0)

    if all_codes.dim() != 2:
        raise ValueError("Codes must be 1D vectors.")
    code_dim = all_codes.shape[1]
    n_cols = code_dim // n_rows
    plt.figure(figsize=figsize)
    for ind, codes_dim in enumerate(all_codes.T):
        plt.subplot(n_rows, n_cols, ind+1)
        plt.hist(codes_dim.cpu().numpy(), bins=100)
    plt.show()


def check_code_usage(model: VAEGauss,
                     dataloader: DataLoader[tuple[DataBatchFloat,
                                                  LabelBatch]]) -> TabularBatchFloat:
    """For each code dimension, see how much it diverges from a standard normal.
    
    This is a proxy for how much information this dimension contains. A code dimension that is
    always near standard normal for all inputs (indicated by low KL) contains little to no
    information about what the input was. The larger the average divergence, the more informative
    this dimension tends to be. This is basically a more formal version of our very fun
    `per_dim_latent_walk`.

    TODO generalize to non-Gauss

    Parameters:
        model: VAE model with encoder that returns Gaussian parameters.
        dataloader: Provides the dataset to get distributions for.
    
    Returns:
        Matrix showing KL-divergence for each input in the dataloader and each dimension.
    """
    def kl_loss_gauss_per_dim(means: DataBatchFloat,
                              log_variances: DataBatchFloat) -> DataBatchFloat:
        means = means.view(means.shape[0], -1)
        log_variances = log_variances.view(log_variances.shape[0], -1)

        return 0.5 * (means**2 - 1 + torch.exp(log_variances) - log_variances)
    
    device = next(model.parameters()).device
    all_klds = []
    with torch.inference_mode():
        for input_batch, _ in dataloader:
            means, logvars = model.encoding_parameters(input_batch.to(device))
            all_klds.append(kl_loss_gauss_per_dim(means, logvars))
    return torch.cat(all_klds, dim=0)


def per_dim_latent_walk(generator: Callable[[TabularBatchFloat], ImageBatchFloat],
                        anchor_code: TabularFloat,
                        range_lim: float,
                        threshold: float,
                        n_steps: int = 100) -> list[float]:
    """Fix all but one dimension of the latent space and "walk" along that one.
    
    Dimensions where a meaningful difference has occured are plotted.

    Parameters:
        generator: Despite the name, any callable (so can also be an `nn.Module`) that takes a
            latent code input and returns images. So for example a VAE decoder + `expected_value`
            chain, or a GAN generator object.
        anchor_code: The "base code" we walk from. This must be on the correct device for the
            generator already! Since we can't guarantee that generator is a `Module`, we cannot rely
            on inferring the device from it.
        range_lim: Will walk from `-range_lim` to `range_lim` in each dimension.
        threshold: We compute the average (i.e. per-pixel) absolute difference between the start and
            end points of the walk. If that is above threshold/n_steps, we count that as a "used"
            dimension, else we skip it. Thus, this parameter, can be taken as the necessary
            difference between the start and end points. The optimal value likely depends on the
            dataset.
        n_steps: How many steps to take on the walk. MUST be a square, and there is currently no
            check that it's not a square. Whoops!
    """
    threshold = threshold / n_steps
    if anchor_code.dim() != 1:
        raise ValueError("anchor_code must be 1d vector.")
    differences = []
    for latent_dim in range(anchor_code.shape[0]):
        code_repeated = torch.tile(anchor_code, [n_steps, 1])
        value_range = torch.linspace(-range_lim, range_lim, n_steps)
        for dim in range(len(code_repeated)):
            code_repeated[dim, latent_dim] = value_range[dim]

        with torch.inference_mode():
            interpolated_images = generator(code_repeated).cpu().numpy()
        avg_abs_diff = (np.abs(interpolated_images[0] - interpolated_images[-1]).sum()
                        / np.prod(interpolated_images[0].shape))
        differences.append(avg_abs_diff)

        if avg_abs_diff < threshold:
            print(f"Dim {latent_dim} below threshold (avg_abs_difference {avg_abs_diff})")
        else:
            print(avg_abs_diff)
            plt.figure(figsize=(5, 5))
            for ind in range(n_steps):
                plt.subplot(int(np.sqrt(n_steps)), int(np.sqrt(n_steps)), ind+1)
                plt.imshow(interpolated_images[ind].transpose(1, 2, 0),
                           vmin=0,
                           vmax=1,
                           cmap="Greys")
                plt.axis("off")
            plt.show()
    return differences
