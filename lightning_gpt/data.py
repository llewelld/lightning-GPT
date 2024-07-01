import math
import random
from typing import Tuple, Union

import torch
from lightning.pytorch.utilities import rank_zero_info
from torch.utils.data import Dataset
from collections import OrderedDict

class CharDataset(Dataset):
    def __init__(self, data: str, block_size: int):
        chars = list(OrderedDict.fromkeys(data))
        data_size, vocab_size = len(data), len(chars)
        rank_zero_info("data has %d characters, %d unique." % (data_size, vocab_size))

        assert vocab_size < 256, "The vocabulary exceeds byte-size storage"

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = bytes(self.stoi[s] for s in data)

    def __len__(self) -> int:
        return math.ceil(len(self.data) / (self.block_size + 1))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # we're actually going to "cheat" and pick a spot in the dataset at random
        i = random.randint(0, len(self.data) - (self.block_size + 1))
        dix = list(self.data[i : i + self.block_size + 1])
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

    def to_tokens(self, message: str, device: Union[str, torch.device]) -> torch.Tensor:
        return torch.tensor([self.stoi[s] for s in message], dtype=torch.long)[None, ...].to(device)

    def from_tokens(self, tokens: torch.Tensor) -> str:
        return "".join([self.itos[int(i)] for i in tokens])
