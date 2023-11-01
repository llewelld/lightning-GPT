from typing import Any, Tuple
from urllib.request import urlopen

import torch
from lightning import LightningModule, Trainer
from lightning.app import CloudCompute, LightningApp
from torch.utils.data import DataLoader

from lightning_gpt import bench, data, models


class FSDPMinGPTBench(bench.Bench):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.num_workers = 4
        self.batch_size = 64
        self.max_epochs = 5
        self.precision = 16
        self.model_type = "gpt2"
        self.num_runs = 5

    def create(self) -> Tuple[models.FSDPMinGPT, DataLoader]:
        torch.set_float32_matmul_precision("high")

        with urlopen("https://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt") as fo:
            text = fo.read()

        dataset = data.CharDataset(text, block_size=128)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)

        model = models.FSDPMinGPT(
            vocab_size=dataset.vocab_size,
            block_size=dataset.block_size,
            model_type=self.model_type,
        )

        return model, dataloader

    def train(self, model: LightningModule, dataloader: DataLoader) -> float:
        self._check_precision()
        trainer = Trainer(
            fast_dev_run=True,
            max_epochs=self.max_epochs,
            gradient_clip_val=1.0,
            accelerator="cuda",
            devices="auto",
            precision=self.precision,  # type: ignore
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
            logger=False,
            use_distributed_sampler=False,
            strategy="fsdp_native",
        )

        trainer.fit(model, dataloader)
        final_loss = trainer.fit_loop.running_loss.last().item()
        return final_loss

    def run(self) -> None:
        model, dataloader = self.create()

        self.run_benchmark(name="nocompile", fn=self.train, args=(model, dataloader), num_runs=self.num_runs)

        model, dataloader = self.create()
        model = torch.compile(model)

        self.run_benchmark("compile", self.train, args=(model, dataloader), num_runs=self.num_runs)


app = LightningApp(
    bench.BenchRun(
        FSDPMinGPTBench,
        num_nodes=2,
        cloud_compute=CloudCompute("gpu-fast"),
    )
)
