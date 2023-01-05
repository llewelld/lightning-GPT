import time

import lightning as L
import torch
from lightning.pytorch import Callback
from lightning.pytorch.utilities import rank_zero_info


class CUDAMetricsCallback(Callback):
    def on_train_epoch_start(self, trainer: "L.Trainer", pl_module: "L.LightningModule"):  # type: ignore
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(self.root_gpu(trainer))
        torch.cuda.synchronize(self.root_gpu(trainer))
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:  # type: ignore
        torch.cuda.synchronize(self.root_gpu(trainer))
        max_memory = torch.cuda.max_memory_allocated(self.root_gpu(trainer)) / 2**20
        epoch_time = time.time() - self.start_time

        max_memory = trainer.strategy.reduce(max_memory)
        epoch_time = trainer.strategy.reduce(epoch_time)

        rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
        rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")

    def root_gpu(self, trainer: "L.Trainer") -> int:  # type: ignore
        return trainer.strategy.root_device.index
