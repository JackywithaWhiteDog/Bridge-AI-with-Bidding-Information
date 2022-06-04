from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import Trainer, LightningModule

class EarlyStoppingWarmup(EarlyStopping):
    def __init__(self, warmup: int=10, **kwargs):
        super(EarlyStoppingWarmup, self).__init__(**kwargs)
        self.warmup = warmup

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        pass

    def on_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.current_epoch >= self.warmup:
            super()._run_early_stopping_check(trainer)