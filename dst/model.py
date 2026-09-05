"""GPT-2, written from scratch, tensor-parallel by construction.

Layout per block, following Megatron:

  attention: q/k/v projections are column-parallel sharded by heads (a
    column split of the projection is exactly a head split, since GeLU-free
    attention math never mixes entries across heads), output projection is
    row-parallel. One all-reduce forward (f-bar on the proj output), the
    matching all-reduces backward (f on the q/k/v inputs).

  mlp: c_fc column-parallel (GeLU is elementwise, a column split never
    mixes entries), c_proj row-parallel. Same one-forward-one-backward
    all-reduce pattern.

LayerNorms, embeddings, and biases-after-reduce are replicated. Their
gradients come out identical on every rank with no extra sync because f's
backward all-reduce restores the full gradient at each TP boundary — the
correctness suite asserts this.

The LM head is column-parallel over a padded vocab (padded to a multiple
of 128 regardless of tp, so initialization draws are tp-invariant) with
gather_output=True; padding logits are sliced off before the loss so they
get zero gradient. Weight tying with wte is deliberately skipped: the head
holds a vocab shard while the embedding is replicated.

Every submodule takes a TPContext; built with parallel.SINGLE this is
plain single-process PyTorch.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .parallel import TPContext
from .layers import ColumnParallelLinear, RowParallelLinear

VOCAB_PAD_MULTIPLE = 128


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0


def _pad_vocab(vocab_size: int) -> int:
    return ((vocab_size + VOCAB_PAD_MULTIPLE - 1) // VOCAB_PAD_MULTIPLE) * VOCAB_PAD_MULTIPLE


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig, tp: TPContext, generator: torch.Generator):
        super().__init__()
        if config.n_head % tp.world != 0:
            raise ValueError(f"n_head {config.n_head} not divisible by tp {tp.world}")
        self.tp = tp
        self.n_head_local = config.n_head // tp.world
        self.head_dim = config.n_embd // config.n_head
        # GPT-2 scales residual-path projections by 1/sqrt(2L).
        proj_std = 0.02 / math.sqrt(2 * config.n_layer)
        self.q = ColumnParallelLinear(config.n_embd, config.n_embd, tp, generator=generator)
        self.k = ColumnParallelLinear(config.n_embd, config.n_embd, tp, generator=generator)
        self.v = ColumnParallelLinear(config.n_embd, config.n_embd, tp, generator=generator)
        self.proj = RowParallelLinear(
            config.n_embd, config.n_embd, tp, init_std=proj_std, generator=generator
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        h, d = self.n_head_local, self.head_dim
        q = self.q(x).view(B, T, h, d).transpose(1, 2)
        k = self.k(x).view(B, T, h, d).transpose(1, 2)
        v = self.v(x).view(B, T, h, d).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, h * d)
        return self.dropout(self.proj(y))


class MLP(nn.Module):
    def __init__(self, config: GPTConfig, tp: TPContext, generator: torch.Generator):
        super().__init__()
        proj_std = 0.02 / math.sqrt(2 * config.n_layer)
        self.c_fc = ColumnParallelLinear(config.n_embd, 4 * config.n_embd, tp, generator=generator)
        self.c_proj = RowParallelLinear(
            4 * config.n_embd, config.n_embd, tp, init_std=proj_std, generator=generator
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x), approximate="tanh")))


class Block(nn.Module):
    def __init__(self, config: GPTConfig, tp: TPContext, generator: torch.Generator):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config, tp, generator)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config, tp, generator)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig, tp: TPContext, generator: torch.Generator = None):
        super().__init__()
        if generator is None:
            generator = torch.Generator().manual_seed(0)
        self.config = config
        self.tp = tp
        self.padded_vocab = _pad_vocab(config.vocab_size)

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.wte.weight.data.normal_(0.0, 0.02, generator=generator)
        self.wpe.weight.data.normal_(0.0, 0.02, generator=generator)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config, tp, generator) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = ColumnParallelLinear(
            config.n_embd,
            self.padded_vocab,
            tp,
            bias=False,
            gather_output=True,
            generator=generator,
        )

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)[..., : self.config.vocab_size]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return logits, loss
