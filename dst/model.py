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

SP has one cost the conjugate operators don't cover: replicated params
that consume sequence-sharded activations (LayerNorm weights/biases,
row-parallel biases added after the reduce-scatter) see only their rank's
sequence chunk in backward, so their gradients come out partial. They are
tagged at construction and allreduce_sequence_parallel_grads() must run
after backward, before the optimizer step. (Embedding grads are NOT
affected: the scatter's backward all-gathers, restoring full gradients.)

With config.sequence_parallel set, the regions between TP blocks —
LayerNorm, dropout, residual adds — run on activations sharded along the
sequence axis: the embedding output is scattered along seq, every
column-parallel entry becomes g (all-gather seq) and every row-parallel
exit becomes g-bar (reduce-scatter seq), so nothing outside the matmuls
ever holds a full b*s*h tensor. Inside the TP region the sequence is full
— attention needs every key and value. The math is identical to plain TP;
only where activations live changes.

Every submodule takes a TPContext; built with parallel.SINGLE this is
plain single-process PyTorch.
"""

import math
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .parallel import TPContext
from .layers import ColumnParallelLinear, RowParallelLinear
from .ops import scatter_along_seq

VOCAB_PAD_MULTIPLE = 128


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    sequence_parallel: bool = False


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
        in_mode = "sequence" if config.sequence_parallel else "replicated"
        self.q = ColumnParallelLinear(
            config.n_embd, config.n_embd, tp, input_mode=in_mode, generator=generator
        )
        self.k = ColumnParallelLinear(
            config.n_embd, config.n_embd, tp, input_mode=in_mode, generator=generator
        )
        self.v = ColumnParallelLinear(
            config.n_embd, config.n_embd, tp, input_mode=in_mode, generator=generator
        )
        self.proj = RowParallelLinear(
            config.n_embd,
            config.n_embd,
            tp,
            output_mode=in_mode,
            init_std=proj_std,
            generator=generator,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Under sequence parallelism x arrives sequence-sharded; the q/k/v
        # projections all-gather it, so T comes from their output, not x.
        B = x.shape[0]
        h, d = self.n_head_local, self.head_dim
        q = self.q(x)
        T = q.shape[1]
        q = q.view(B, T, h, d).transpose(1, 2)
        k = self.k(x).view(B, T, h, d).transpose(1, 2)
        v = self.v(x).view(B, T, h, d).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, h * d)
        return self.dropout(self.proj(y))


class MLP(nn.Module):
    def __init__(self, config: GPTConfig, tp: TPContext, generator: torch.Generator):
        super().__init__()
        proj_std = 0.02 / math.sqrt(2 * config.n_layer)
        mode = "sequence" if config.sequence_parallel else "replicated"
        self.c_fc = ColumnParallelLinear(
            config.n_embd, 4 * config.n_embd, tp, input_mode=mode, generator=generator
        )
        self.c_proj = RowParallelLinear(
            4 * config.n_embd,
            config.n_embd,
            tp,
            output_mode=mode,
            init_std=proj_std,
            generator=generator,
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
            input_mode="sequence" if config.sequence_parallel else "replicated",
            generator=generator,
        )

        if config.sequence_parallel:
            for m in self.modules():
                if isinstance(m, nn.LayerNorm):
                    for p in m.parameters():
                        p.sequence_parallel_replicated = True

    def finalize_grads(self) -> None:
        """All-reduce the partial gradients of replicated params that live in
        sequence-sharded regions. Call after backward, before the optimizer
        step. No-op without sequence parallelism."""
        if not (self.config.sequence_parallel and self.tp.enabled):
            return
        for p in self.parameters():
            if getattr(p, "sequence_parallel_replicated", False) and p.grad is not None:
                dist.all_reduce(p.grad, group=self.tp.group)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        if self.config.sequence_parallel:
            if T % self.tp.world != 0:
                raise ValueError(f"seq len {T} not divisible by tp {self.tp.world}")
            x = scatter_along_seq(x, self.tp)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)[..., : self.config.vocab_size]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return logits, loss
