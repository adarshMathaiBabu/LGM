"""CNN "level" implementation. See `CNNLevel` class docs for more information."""
from __future__ import annotations

from typing import override

from torch import nn

from lgm.types import DataBatchFloat
from .blocks import BlockConfig, ConvBlockPrenorm
from .core import INTERPOLATION_MAP, UpsampleAA


TAKE_SKIPS_ALL = 1
TAKE_SKIPS_EXCEPT_FIRST = 2
TAKE_SKIPS_FIRST_ONLY = 3


class CNNLevel(nn.Module):
    """One level of a CNN consists of a number of convolutional blocks as well as up-/down-sampling.
    
    All blocks at a level output the same image size and number of filters.
    """
    @override
    def __init__(self,
                 n_blocks: int,
                 previous_filters: int,
                 current_filters: int,
                 stride: int,
                 direction: str,
                 block_config: BlockConfig,
                 use_attention: bool = False,
                 return_all: bool = False,
                 take_skips: int = 0):
        """Configure the level and create component blocks.
        
        Parameters:
            n_blocks: How many blocks the level should consist of.
            previous_filters: Number of channels of the input to this level.
            current_filters: Number of channels on this level.
            stride: Up- or down-sampling factor applied by the first block. NOTE: In case of
                downsampling, the stride is applied at the start of the first block. In case of
                upsampling, bilinear upsampling is added at the *end* of the level.
            direction: One of `"up"` or `"down"`. `"down"` would usually be for encoder-type models,
                `"up"` for decoder/generator-like.
            block_config: `BlockConfig` object to config the blocks.
            use_attention: Whether to insert attention blocks after the residuals at this level.
            return_all: See doc of `CNNBody`. If `True,` return a list of tensors of all the block
                outputs in this level.
            take_skips: See doc of `CNNBody`. If `> 0`, forward must receive a list of tensors in
                `skip_inputs`. A value of `TAKE_SKIPS_ALL` will assume *all* blocks get a skip
                input. A value of `TAKE_SKIPS_EXCEPT_FIRST` will build the *first block* in the
                level to *not* get skip inputs! A value of `TAKE_SKIPS_FIRST_ONLY` will assume *only
                the first block* to take skips.  
                Usually you want to set this to `EXCEPT_FIRST` for the first level in the decoder of
                a U-Net-style model, and `ALL` for the remaining levels. `FIRST_ONLY` is intended
                for input skips for StyleGAN-like architectures.  
                NOTE that layers that don't receive skip inputs must get `None` provided in their
                `forward` call.
        """
        super().__init__()
        self.blocks = nn.ModuleList()
        for layer_ind in range(n_blocks):
            do_take_skips = skip_helper(take_skips, layer_ind)
            self.blocks.append(ConvBlockPrenorm(
                previous_filters=previous_filters if layer_ind == 0 else current_filters,
                current_filters=current_filters,
                config=block_config,
                stride=stride if (layer_ind == 0 and direction == "down") else 1,
                skip_dim=current_filters if do_take_skips else 0,
                use_attention=use_attention))
        if direction == "up" and stride > 1:
            self.upsample = UpsampleAA(scale_factor=stride,
                                       mode=INTERPOLATION_MAP[block_config.data_format])
        else:
            self.upsample = None
        self.return_all = return_all

    @override
    def forward(self,
                inputs: DataBatchFloat,
                skip_inputs: list[DataBatchFloat] | None = None)\
                    -> DataBatchFloat | tuple[DataBatchFloat, list[DataBatchFloat]]:
        """Compute level outputs.
        
        Parameters:
            inputs: Input features for this level.
            skip_inputs: Optional list of additional inputs to each block. If given, must be one per
                block, although certain blocks may receive `None`. Blocks MUST be configured with
                the correct `skip_dim`.
            conditioning: Optional conditioning information passed to each block. Blocks MUST be
                configured with the correct `cond_dim`.

        Returns:
            Either just the output features, or, if `return_all` is `True``, a tuple of output
            features and a list of all block outputs for this level.
        """
        outputs = inputs
        if self.return_all:
            all_level_outputs = []
        for ind, block in enumerate(self.blocks):
            skip_input = None if skip_inputs is None else skip_inputs[ind]
            outputs = block(outputs, skip_input)
            if self.return_all:
                all_level_outputs.append(outputs)
            
        if self.upsample is not None:
            outputs = self.upsample(outputs)
        return (outputs, all_level_outputs) if self.return_all else outputs


def skip_helper(take_skips: int,
                layer_index: int) -> bool:
    """Helper function to figure out whether a layer on a level should take skips.
    
    Parameters:
        layer_index: Index of the layer within its level.

    Returns:
        `True` if skips should be taken, `False` otherwise.
    """
    # for unet skips, first layer of first level of decoder should not take skips
    if take_skips == TAKE_SKIPS_EXCEPT_FIRST:
        return layer_index != 0
    # for input skips, only the first layer of each level (EXCEPT THE FIRST LEVEL -- NOT HANDLED 
    # HERE) should take skips
    if take_skips == TAKE_SKIPS_FIRST_ONLY:
        return layer_index == 0
    # for unet skips, levels beyond the first take skips in all layers
    if take_skips == TAKE_SKIPS_ALL:
        return True
    # else
    return False
