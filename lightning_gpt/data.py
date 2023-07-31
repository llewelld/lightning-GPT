import math
import random
import os
from collections import OrderedDict
from typing import Tuple, Union

import torch
from lightning.pytorch.utilities import rank_zero_info
from torch.utils.data import Dataset

# Restrict the data to a given maximum size
# Set to None for no maximum
MAX_DATA = 4573338

class CharDataset(Dataset):
    alphabet_file = '../../../datasets/pile/alphabet.bin'
    order = ''
    char_to_byte_lookup = {}
    byte_to_char_lookup = ''
    conversion = {
        '\t': ' ',
        '\n': ' ',
        '`': '\'',
        ' ': ' ',
        '´': '\'',
        '‐': '-',
        '–': '-',
        '—': '-',
        '‘': '\'',
        '’': '\'',
        '“': '"',
        '”': '"',
        '−': '-',
    }
    files_in = ''
    block_size = 0
    blocks = 0
    file_handles = []
    file_size = 0
    blocks_per_file = 0
    vocab_size = 0

    def input_alphabet(self, file_in):
        with open(file_in, 'rb') as f:
           data = f.read()
           self.order = data.decode(encoding='utf-8')

        for pos in range(min(len(self.order), 256)):
            char = self.order[pos:pos + 1]
            self.char_to_byte_lookup[char] = pos
            self.byte_to_char_lookup += char 

        for key, value in self.conversion.items():
            if self.char_to_byte_lookup.get(value, None) != None:
                self.char_to_byte_lookup[key] = self.char_to_byte_lookup[value]
            else:
                rank_zero_info('Skipping, as not in alphabet: {} ({})'.format(value, ord(value)))

        rank_zero_info('Alphabet read from: {}'.format(file_in))

    def char_to_byte(self, char):
        return self.char_to_byte_lookup.get(char, None)

    def byte_to_char(self, byte):
        assert byte < 256
        assert byte > 0
        return self.byte_to_char_lookup[byte:byte + 1]

    def tokenise(self, text):
        return bytes(map(lambda char: self.char_to_byte_lookup[char], filter(lambda char: char in self.char_to_byte_lookup, text)))

    def detokenise(self, tokenised):
        return ''.join(map(lambda byte: self.byte_to_char_lookup[byte:byte + 1], tokenised))

    def __init__(self, files_in: str, block_size: int):
        # Set up variables
        self.files_in = files_in
        self.block_size = block_size

        # Load in the alphabet
        self.input_alphabet(self.alphabet_file)
        self.vocab_size = min(len(self.order), 256)

        # Size up the input data
        found = None
        count = 0
        total = None

        while found == None:
            file_in = self.files_in.format(count)
            try:
                with open(file_in, 'rb') as f:
                    f.seek(0, os.SEEK_END)
                    size = f.tell()
                    if self.file_size == 0:
                        assert size % self.block_size == 0
                        self.file_size = size
                        self.blocks_per_file = size // self.block_size
                    self.blocks += size // self.block_size
                    if size < self.file_size:
                        if size == 0:
                            found = count
                        else:
                            found = count + 1
            except FileNotFoundError:
                found = count
            count += 1
        self.blocks = int(self.blocks) + 1

        rank_zero_info('Number of files found: {}'.format(found))
        rank_zero_info('Total number of blocks: {}'.format(self.blocks))

        if MAX_DATA and (self.blocks * self.block_size > MAX_DATA):
            self.blocks = MAX_DATA // self.block_size
            rank_zero_info('Restricted to bytes: {}'.format(MAX_DATA))
            rank_zero_info('Adjusted number of blocks: {}'.format(self.blocks))

        # We can't open the files until processing has started
        self.file_handles = [None] * found

    def __len__(self) -> int:
        return self.blocks

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Grab a block from one of the files
        file_count = idx // self.blocks_per_file
        block_count = idx % self.blocks_per_file
        assert idx < self.blocks

        # Open the file if it isn't already and keep hold of the handle
        if self.file_handles[file_count] == None:
            self.file_handles[file_count] = open(self.files_in.format(file_count), 'rb')

        self.file_handles[file_count].seek(self.block_size * block_count, os.SEEK_SET)
        data = self.file_handles[file_count].read(self.block_size)

        x = torch.tensor([int(x) for x in data[:-1]], dtype=torch.long)
        y = torch.tensor([int(x) for x in data[1:]], dtype=torch.long)

        return x, y

    def to_tokens(self, message: str, device: Union[str, torch.device]) -> torch.Tensor:
        return torch.tensor(map(lambda char: self.char_to_byte_lookup[char], message), dtype=torch.long)[None, ...].to(device)

    def from_tokens(self, tokens: torch.Tensor) -> str:
        return ''.join(map(lambda byte: self.byte_to_char_lookup[int(byte):int(byte) + 1], tokens))

