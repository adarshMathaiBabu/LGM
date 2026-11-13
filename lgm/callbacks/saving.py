"""Callbacks that do some kind of model saving stuff."""
from pathlib import Path
from typing import override

import torch
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities import rank_zero_info
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from lgm.types import DataBatchFloat, LabelBatch


class BestCheckpoint(ModelCheckpoint):
    """Wrapper around ModelCheckpoint that changes the "last" checkpoint name.
    
    `ModelCheckpoint` supports a feature where it will create a copy of the last checkpoint saved,
    and call it "last". This one simply renames that to "best". The idea is to have two checkpoint
    callbacks: One simply saves the latest checkpoints, the other saves only the best by some
    metric. By renaming "last" here, we can use `save_last` for both callbacks, while avoiding
    naming confusion.
    """
    CHECKPOINT_NAME_LAST = "best"


class EMAModel(Callback):
    """Stores an exponential moving average of model parameters, and saves weights at the end.
    
    This does NOT save a proper checkpoint like the `ModelCheckpoint` callback. It merely saves the
    `state_dict`. You would first have to restore the model through other means (e.g. from a full
    checkpoint), and then use `model.load_state_dict(torch.load(path, weights_only=True))` to
    replace the weights by the EMA.  
    The folder to store the `state_dict` is simply grabbed from the first `ModelCheckpoint` instance
    it can find. If you are not running a `ModelCheckpoint` callback, this will not save anything.
    """
    @override
    def __init__(self,
                 decay: float = 0.999,
                 include_buffers: bool = True):
        """Configure the callback.
        
        Parameters:
            decay: Decay rate for EMA.
            include_buffers: If `True`, include buffers in the EMA. Most importantly, this includes
                things like the running averages for Batchnorm. Since these are already moving
                averages over time, it's a bit superfluous to apply EMA to them usually.  
                Unfortunately, using `False` most likely prevents the buffers from being saved in
                the EMA model *at all*, i.e. it will use whatever values the buffers were
                initialized to.
        """
        self.decay = decay
        self.ema_model = None
        self.include_buffers = include_buffers

    @override
    def on_fit_start(self,
                     trainer: Trainer,
                     pl_module: LightningModule):
        self.ema_model = AveragedModel(pl_module,
                                       multi_avg_fn=get_ema_multi_avg_fn(self.decay),
                                       use_buffers=self.include_buffers)

    @override
    def on_train_batch_end(self,
                           trainer: Trainer,
                           pl_module: LightningModule,
                           outputs,
                           batch: tuple[DataBatchFloat, LabelBatch],
                           batch_idx: int):
        self.ema_model.update_parameters(pl_module)

    @override
    def on_fit_end(self,
                   trainer: Trainer,
                   pl_module: LightningModule):
        if not trainer.is_global_zero:
            return

        checkpoint_callbacks = [cb for cb in trainer.callbacks
                                if isinstance(cb, ModelCheckpoint)]

        if not checkpoint_callbacks:
            rank_zero_info("EMACallback did not find ModelCheckpoint callback. EMA checkpoint will "
                           "NOT be saved!")
            return

        checkpoint_dir = checkpoint_callbacks[0].dirpath
        path = Path(checkpoint_dir) / "ema_weights.pt"
        torch.save(self.ema_model.module.state_dict(), path)
