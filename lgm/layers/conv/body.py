"""CNN "body" implementation. See `CNNBody` and `BodyConfig` class docs for more information."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import override

import torch
from torch import nn

from lgm.types import DataBatchFloat, ScalarFloat, TabularBatchFloat
from lgm.layers.norms import n_groups_convert, norm_selector
from .blocks import BlockConfig
from .core import INTERPOLATION_MAP, DownsampleAA, UpsampleAA, conv_nd
from .level import TAKE_SKIPS_ALL, TAKE_SKIPS_EXCEPT_FIRST, TAKE_SKIPS_FIRST_ONLY, CNNLevel


class CNNBody(nn.Module):
    """A sequence of CNN "levels", which in turn consist of sequences of "blocks".
    
    Each level can have a different number of filters, and optional down-/up-sampling.
    """
    @override
    def __init__(self,
                 direction: str,
                 body_config: BodyConfig,
                 block_config: BlockConfig,
                 return_all: bool = False,
                 take_skips: bool = False):
        """Configure the body and create levels.
    
        Parameters:
            direction: One of `"up"` or `"down"`. `"down"` would usually be for encoder-type models,
                `"up"` for decoder/generator-like.
            body_config: `BodyConfig` object.
            block_config: `BlockConfig` to config the blocks.
            return_all: If `True`, additionally to the final output, will return a list of lists of
                tensors, containing the outputs of all blocks at all levels. Useful for feature
                matching in GANs, skip connections in U-Net architectures (encoder part), or
                "output skip" designs such as in StyleGAN2.
            take_skips: If `True`, each block expects an additional input besides the output of the
                previous block. These will have to be provided in the forward call as a list of
                lists of tensors `skip_inputs`, such as returned by another `CNNBody` with
                `return_all=True`. Useful for skip-connections in U-Net architectures (decoder part)
                or so-called "input skips" as proposed for StyleGAN2.  
                NOTE A naive implementation will duplicate the input for the first block of the
                first level. This is because it receives the same "regular" input as the skip input.
                We avoid this by special arguments to the `CNNLevel` class. Also NOTE that you are
                responsible for making sure no skip input is passed to the first block of the first
                level when actually calling the model! Else you will get a shape mismatch.  
                NOTE that we automatically derive the desired skips via the direction argument. For
                `'down'`, we use input skips, where the first block of each level receives a skip
                input. For `'up'`, we use U-Net decoder skips as described above.  
                Also NOTE that this will simply assume *the exact same dimensionality* for the skip
                inputs as the current block dimension. That is, we assume "mirrored" encoder/decoder
                designs, and that the skip inputs come from the encoder and go into the decoder.
                This is NOT a general-purpose skip connection implementation.
        """
        super().__init__()

        self.levels = nn.ModuleList()
        previous_filters = body_config.level_filters[0]
        for level_ind in range(len(body_config.level_filters)):
            if direction == "down":
                skip_kind = 0 if level_ind == 0 else TAKE_SKIPS_FIRST_ONLY
            elif direction == "up":
                skip_kind = TAKE_SKIPS_EXCEPT_FIRST if level_ind == 0 else TAKE_SKIPS_ALL

            current_filters = body_config.level_filters[level_ind]
            self.levels.append(CNNLevel(body_config.blocks_per_level[level_ind],
                                        previous_filters,
                                        current_filters,
                                        stride=body_config.level_strides[level_ind],
                                        direction=direction,
                                        block_config=block_config,
                                        use_attention=body_config.level_attentions[level_ind],
                                        return_all=return_all,
                                        take_skips=skip_kind*take_skips))
            previous_filters = current_filters

        self.return_all = return_all
        self.final_normact = body_config.final_normact
        if self.final_normact:
            norm_groups = n_groups_convert(block_config.norm_groups, body_config.level_filters[-1])
            self.final_norm = norm_selector(block_config.norm, block_config.data_format)(
                norm_groups,
                body_config.level_filters[-1])
            self.final_activation = block_config.activation()
            # TODO use dropout here?

        if body_config.level_residual:
            self.interpolators = nn.ModuleList([])
            data_format = block_config.data_format
            # NOTE these are 1x1 convs, so no depthwise
            conv_fn = conv_nd(data_format)
            previous_filters = body_config.level_filters[0]

            for level_ind, stride in enumerate(body_config.level_strides):
                current_filters = body_config.level_filters[level_ind]
                interpolator = nn.Sequential(conv_fn(in_channels=previous_filters, 
                                                     out_channels=current_filters,
                                                     kernel_size=1))
                if direction == "down" and stride > 1:
                    interpolator.append(DownsampleAA(scale_factor=stride,
                                                     mode=INTERPOLATION_MAP[data_format]))
                elif direction == "up" and stride > 1:
                    interpolator.append(UpsampleAA(scale_factor=stride,
                                                   mode=INTERPOLATION_MAP[data_format]))

                self.interpolators.append(interpolator)
                previous_filters = current_filters

            self.residual_factor: ScalarFloat
            if block_config.rescale_residual:
                self.register_buffer("residual_factor", 1/torch.sqrt(torch.tensor(2.)))
            else:
                self.register_buffer("residual_factor", torch.tensor(1.))
        else:
            self.interpolators = None

    @override
    def forward(self,
                inputs: DataBatchFloat,
                skip_inputs: list[list[DataBatchFloat]] | None = None)\
                    -> DataBatchFloat | tuple[DataBatchFloat, list[list[DataBatchFloat]]]:
        """Apply the body.
        
        Parameters:
            inputs: Input features.
            skip_inputs: Optional list of additional inputs to each level. If given, must be one per
                level, each of which also must receive a list. Blocks MUST be configured with the
                correct `skip_dim`.

        Returns:
            Either just the output features, or, if `return_all` is `True`, a tuple of output
            features and a list of list all level outputs.
        """
        outputs = inputs
        if self.return_all:
            all_level_outputs = []

        for ind, level in enumerate(self.levels):
            skip_input = None if skip_inputs is None else skip_inputs[ind]
            level_inputs = outputs  # save for residual

            if self.return_all:
                outputs, all_block_outputs = level(outputs, skip_inputs=skip_input)
                all_level_outputs.append(all_block_outputs)
            else:
                outputs = level(outputs, skip_inputs=skip_input)

            if self.interpolators is not None:
                outputs = self.residual_factor * (outputs + self.interpolators[ind](level_inputs))

        if self.final_normact:
            outputs = self.final_activation(self.final_norm(outputs))
        return (outputs, all_level_outputs) if self.return_all else outputs


@dataclass(frozen=True)
class BodyConfig:
    """Helper object to bundle parameters for `CNNBody`.
    
    In practice, `CNNBody` is often called as a chain like `EncoderNet -> DownCNN -> CNNBody`.
    Carrying arguments all the way through is cumbersome. Thus, we bundle them in a config, as
    `**kwargs` are not an option (see `BlockConfig` docs).

    NOTE that this does not include all arguments to `CNNBody`, as some are generally set depending
    on context by the class(es) above. Also NOTE that certain classes may override some attributes 
    to make sure they work in context.

    Parameters:
        blocks_per_level: How many blocks each level consists of. Can be `int`, then all levels will
            use the same number of blocks. Or an iterable like `filters` and `strides.`
        level_filters: Iterable (e.g. list) with number of filters for each level.
        level_strides: Stride (up- or downsampling factor) at each level. If not given, will use
            stride of 1 for the first level, and 2 for all others. This is a sensible default for
            encoder-style models working on images, causing the first level to work at the original
            input resolution, and each successive level at a reduced one. Take care to explictly
            invert this for decoder-style models!  
            For models on e.g. raw audio, you will likely want to use larger strides, such as 4.
        level_attentions: Optional iterable of bools. Each level where this is `True` will have
            multi-head self-attention blocks interspersed after each convolutional  block. Commonly
            used in diffusion models at lower resolutions.  
            NOTE be careful with this -- activating at high resolutions will result in massive
            memory and runtime increase. Usually recommended at `16x16` at most, especially for the
            kind of models/data we use in this course.
        final_normact: If `True`, apply a final sequence of normalization & activation after the
            last level. This can improve performance somewhat. But note that this can sometimes be
            superfluous in models where we apply multiple `CNNBodys` in sequence, e.g. U-Nets. Here,
            the first `Body` would end with a final `norm-act`, and the second body would also start
            with `norm-act`. In such cases, you could set this to `False` for the first `Body`.
        level_residual: If `True`, uses residual connections between levels as proposed for
            StyleGAN2. This is NOT the same as the residual convolutional blocks!
    """
    blocks_per_level: int | Iterable[int]
    level_filters: Iterable[int]
    level_strides: Iterable[int] | None = None
    level_attentions: Iterable[bool] | None = None
    final_normact: bool = True
    level_residual: bool = False

    @override
    def __post_init__(self):
        """Validate arguments and bring to common form.
        
        Raises:
            ValueError: If any of the per-level lists have different lengths.
        """
        if self.level_strides is None:
            object.__setattr__(self, "level_strides",
                               [1] + [2] * (len(self.level_filters) - 1))
        if self.level_attentions is None:
            object.__setattr__(self, "level_attentions",
                               [False] * len(self.level_filters))
        if isinstance(self.blocks_per_level, int):
            object.__setattr__(self, "blocks_per_level",
                               [self.blocks_per_level] * len(self.level_filters))

        lengths = {"blocks": len(self.blocks_per_level),
                   "filters": len(self.level_filters),
                   "strides": len(self.level_strides),
                   "attentions": len(self.level_attentions)}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Expected all configuration lists to have the same length, got "
                             f"{lengths}.")
