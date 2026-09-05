"""Data: a memmapped tokenized shard, sampled deterministically.

The shard is a flat uint16 token file (nanoGPT convention — GPT-2 BPE ids
fit in uint16). np.memmap keeps it on disk; each batch touches only the
pages it reads, so shard size doesn't affect memory.

Batches are drawn at deterministic offsets seeded by (seed, step,
dp_rank). That does two jobs with zero communication:

  - every rank inside one DP replica (all its TP ranks, all its pipeline
    stages) computes the identical batch, so the first stage's tokens and
    the last stage's targets agree without a broadcast;
  - different DP replicas draw disjoint-in-expectation batches, which is
    all data parallelism needs.
"""

import numpy as np
import torch

from .parallel import DPContext, DP_SINGLE


class TokenShard:
    def __init__(self, path: str, block_size: int, batch_size: int,
                 seed: int = 0, dp: DPContext = DP_SINGLE):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        if len(self.data) < block_size + 1:
            raise ValueError(f"shard {path} shorter than one block")
        self.block_size = block_size
        self.batch_size = batch_size
        self.seed = seed
        self.dp = dp

    def get_batch(self, step: int):
        g = torch.Generator().manual_seed(
            self.seed * 1000003 + step * 1009 + self.dp.rank
        )
        ix = torch.randint(len(self.data) - self.block_size - 1, (self.batch_size,), generator=g)
        x = torch.stack(
            [torch.from_numpy(self.data[i : i + self.block_size].astype(np.int64)) for i in ix]
        )
        y = torch.stack(
            [torch.from_numpy(self.data[i + 1 : i + 1 + self.block_size].astype(np.int64)) for i in ix]
        )
        return x, y


class SyntheticShard:
    """Random tokens with the same interface and the same determinism
    contract, for smoke tests and benchmarking without a dataset."""

    def __init__(self, vocab_size: int, block_size: int, batch_size: int,
                 seed: int = 0, dp: DPContext = DP_SINGLE):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.batch_size = batch_size
        self.seed = seed
        self.dp = dp

    def get_batch(self, step: int):
        g = torch.Generator().manual_seed(
            self.seed * 1000003 + step * 1009 + self.dp.rank
        )
        x = torch.randint(self.vocab_size, (self.batch_size, self.block_size), generator=g)
        y = torch.randint(self.vocab_size, (self.batch_size, self.block_size), generator=g)
        return x, y
