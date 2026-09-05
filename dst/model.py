"""GPT-2, written from scratch, tensor-parallel by construction.

Layout per block, following Megatron:

  attention: q/k/v projections are column-parallel sharded by heads (a
    column split of the projection is exactly a head split, since
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

Initialization is invariant to the parallel decomposition in BOTH axes.
TP: sharded layers draw the full matrix from a generator and keep a slice.
PP: every component (each block, each embedding, the head) draws from its
own generator, seeded by (base seed, component id), so a pipeline stage
that builds only its own layers gets bit-identical weights to the full
model. That is what lets every correctness suite compare against a plain
single-process reference with no weight copying.

SP has one cost the conjugate operators don't cover: replicated params
that consume sequence-sharded activations (LayerNorm weights/biases,
row-parallel biases added after the reduce-scatter) see only their rank's
sequence chunk in backward, so their gradients come out partial. They are
tagged at construction and finalize_grads() must run after backward,
before the optimizer step. (Embedding grads are NOT affected: the
scatter's backward all-gathers, restoring full gradients.)

With config.sequence_parallel set, the regions between TP blocks —
LayerNorm, dropout, residual adds — run on activations sharded along the
sequence axis: the embedding output is scattered along seq, every
column-parallel entry becomes g (all-gather seq) and every row-parallel
exit becomes g-bar (reduce-scatter seq), so nothing outside the matmuls
ever holds a full b*s*h tensor. Inside the TP region the sequence is full
— attention needs every key and value. The math is identical to plain TP;
only where activations live changes. Pipeline boundaries under SP move
the seq-sharded tensor, so p2p volume divides by tp as well.

GPT is the whole model on one process group (tp only). GPTStage is one
pipeline stage of it: a contiguous slice of blocks, plus embeddings on
the first stage and ln_f + head + loss on the last. Every submodule takes
a TPContext; built with parallel.SINGLE this is plain PyTorch.
"""

import math
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .parallel import TPContext, PPContext, PP_SINGLE
from .layers import ColumnParallelLinear, RowParallelLinear
from .ops import scatter_along_seq
from .recompute import recompute

VOCAB_PAD_MULTIPLE = 128

# Component ids for per-component init generators (pp-invariant init).
_COMP_WTE = 0
_COMP_WPE = 1
_COMP_HEAD = 2
_COMP_BLOCK0 = 10


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    sequence_parallel: bool = False
    recompute_attention: bool = False  # selective recompute of core attention


def _pad_vocab(vocab_size: int) -> int:
    return ((vocab_size + VOCAB_PAD_MULTIPLE - 1) // VOCAB_PAD_MULTIPLE) * VOCAB_PAD_MULTIPLE


def _gen(seed: int, component: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed * 1000003 + component)


def core_attention(q, k, v):
    """softmax(QK^T / sqrt(d)) V with a causal mask, materialized the way
    pre-FlashAttention kernels do it: the s x s attention matrix and its
    softmax exist as real tensors (activations scaling as s^2*b*a). That is
    deliberate — selective recompute exists to discard exactly these, so the
    framework must actually allocate them. Deterministic and RNG-free, which
    recompute() requires."""
    d = q.shape[-1]
    att = (q @ k.transpose(-2, -1)) / math.sqrt(d)
    T = att.shape[-1]
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=att.device), diagonal=1)
    att = att.masked_fill(mask, float("-inf"))
    att = F.softmax(att, dim=-1)
    return att @ v


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig, tp: TPContext, generator: torch.Generator):
        super().__init__()
        if config.n_head % tp.world != 0:
            raise ValueError(f"n_head {config.n_head} not divisible by tp {tp.world}")
        self.config = config
        self.tp = tp
        self.n_head_local = config.n_head // tp.world
        self.head_dim = config.n_embd // config.n_head
        # GPT-2 scales residual-path projections by 1/sqrt(2L).
        proj_std = 0.02 / math.sqrt(2 * config.n_layer)
        mode = "sequence" if config.sequence_parallel else "replicated"
        self.q = ColumnParallelLinear(
            config.n_embd, config.n_embd, tp, input_mode=mode, generator=generator
        )
        self.k = ColumnParallelLinear(
            config.n_embd, config.n_embd, tp, input_mode=mode, generator=generator
        )
        self.v = ColumnParallelLinear(
            config.n_embd, config.n_embd, tp, input_mode=mode, generator=generator
        )
        self.proj = RowParallelLinear(
            config.n_embd,
            config.n_embd,
            tp,
            output_mode=mode,
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
        if self.config.recompute_attention:
            y = recompute(core_attention, q, k, v)
        else:
            y = core_attention(q, k, v)
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


def _make_embeddings(config: GPTConfig, seed: int):
    wte = nn.Embedding(config.vocab_size, config.n_embd)
    wpe = nn.Embedding(config.block_size, config.n_embd)
    wte.weight.data.normal_(0.0, 0.02, generator=_gen(seed, _COMP_WTE))
    wpe.weight.data.normal_(0.0, 0.02, generator=_gen(seed, _COMP_WPE))
    return wte, wpe


def _make_head(config: GPTConfig, tp: TPContext, padded_vocab: int, seed: int):
    return ColumnParallelLinear(
        config.n_embd,
        padded_vocab,
        tp,
        bias=False,
        gather_output=True,
        input_mode="sequence" if config.sequence_parallel else "replicated",
        generator=_gen(seed, _COMP_HEAD),
    )


def _tag_sequence_parallel_replicated(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.LayerNorm):
            for p in m.parameters():
                p.sequence_parallel_replicated = True


def _finalize_grads(module: nn.Module, config: GPTConfig, tp: TPContext) -> None:
    if not (config.sequence_parallel and tp.enabled):
        return
    for p in module.parameters():
        if getattr(p, "sequence_parallel_replicated", False) and p.grad is not None:
            dist.all_reduce(p.grad, group=tp.group)


class GPT(nn.Module):
    """The full model on one set of ranks (tensor parallelism only)."""

    def __init__(self, config: GPTConfig, tp: TPContext, seed: int = 0):
        super().__init__()
        self.config = config
        self.tp = tp
        self.padded_vocab = _pad_vocab(config.vocab_size)

        self.wte, self.wpe = _make_embeddings(config, seed)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            Block(config, tp, _gen(seed, _COMP_BLOCK0 + i)) for i in range(config.n_layer)
        )
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = _make_head(config, tp, self.padded_vocab, seed)
        if config.sequence_parallel:
            _tag_sequence_parallel_replicated(self)

    def finalize_grads(self) -> None:
        """All-reduce the partial gradients of replicated params that live in
        sequence-sharded regions. Call after backward, before the optimizer
        step. No-op without sequence parallelism."""
        _finalize_grads(self, self.config, self.tp)

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


class GPTStage(nn.Module):
    """One pipeline stage: a contiguous slice of blocks, plus embeddings on
    the first stage and ln_f + LM head + loss on the last.

    forward() takes token ids on the first stage and the hidden-state
    activation (sequence-sharded under SP) everywhere else; it returns the
    hidden state on non-last stages and (logits, loss) on the last. The
    1F1B schedule in dst/pipeline.py moves activations and their gradients
    between stages.

    With the default arguments the stage is the contiguous slice implied by
    pp. The interleaved schedule instead passes an explicit layer_range and
    virtual-first/last flags to build one model chunk (see GPTChunks).
    """

    def __init__(
        self,
        config: GPTConfig,
        tp: TPContext,
        pp: PPContext = PP_SINGLE,
        seed: int = 0,
        *,
        layer_range=None,
        is_first: bool = None,
        is_last: bool = None,
    ):
        super().__init__()
        self.config = config
        self.tp = tp
        self.pp = pp
        self.padded_vocab = _pad_vocab(config.vocab_size)

        if layer_range is None:
            if config.n_layer % pp.world != 0:
                raise ValueError(f"n_layer {config.n_layer} not divisible by pp {pp.world}")
            per_stage = config.n_layer // pp.world
            layer_range = (pp.rank * per_stage, (pp.rank + 1) * per_stage)
            is_first, is_last = pp.is_first, pp.is_last
        self.first_layer = layer_range[0]
        self.is_first_stage = is_first
        self.is_last_stage = is_last

        if is_first:
            self.wte, self.wpe = _make_embeddings(config, seed)
            self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            Block(config, tp, _gen(seed, _COMP_BLOCK0 + self.first_layer + i))
            for i in range(layer_range[1] - layer_range[0])
        )
        if is_last:
            self.ln_f = nn.LayerNorm(config.n_embd)
            self.lm_head = _make_head(config, tp, self.padded_vocab, seed)
        if config.sequence_parallel:
            _tag_sequence_parallel_replicated(self)

    def finalize_grads(self) -> None:
        _finalize_grads(self, self.config, self.tp)

    def ref_param_name(self, name: str) -> str:
        """Map a stage-local parameter name to the full GPT's name."""
        if name.startswith("blocks."):
            parts = name.split(".")
            parts[1] = str(self.first_layer + int(parts[1]))
            return ".".join(parts)
        return name

    def forward(self, x: torch.Tensor, targets: torch.Tensor = None):
        if self.is_first_stage:
            B, T = x.shape
            pos = torch.arange(T, device=x.device)
            x = self.wte(x) + self.wpe(pos)
            if self.config.sequence_parallel:
                if T % self.tp.world != 0:
                    raise ValueError(f"seq len {T} not divisible by tp {self.tp.world}")
                x = scatter_along_seq(x, self.tp)
            x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        if not self.is_last_stage:
            return x
        x = self.ln_f(x)
        logits = self.lm_head(x)[..., : self.config.vocab_size]
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return logits, loss


class GPTChunks(nn.Module):
    """The v model chunks one rank holds under the interleaved schedule.

    Virtual stage k = c * p + r lives on rank r = k % p as chunk c = k // p,
    so chunk c on rank r holds layers [(c*p + r) * L/(p*v), ...). A
    microbatch runs rank 0..p-1 through chunk 0, wraps back to rank 0 for
    chunk 1, and so on: p2p becomes a ring. Embeddings live in chunk 0 of
    rank 0; head and loss in chunk v-1 of rank p-1.
    """

    def __init__(self, config: GPTConfig, tp: TPContext, pp: PPContext, v: int, seed: int = 0):
        super().__init__()
        p = pp.world
        if config.n_layer % (p * v) != 0:
            raise ValueError(f"n_layer {config.n_layer} not divisible by p*v {p * v}")
        per = config.n_layer // (p * v)
        self.config = config
        self.tp = tp
        self.pp = pp
        self.v = v
        self.chunks = nn.ModuleList()
        for c in range(v):
            first = (c * p + pp.rank) * per
            self.chunks.append(
                GPTStage(
                    config,
                    tp,
                    pp,
                    seed,
                    layer_range=(first, first + per),
                    is_first=(pp.rank == 0 and c == 0),
                    is_last=(pp.rank == p - 1 and c == v - 1),
                )
            )

    def finalize_grads(self) -> None:
        _finalize_grads(self, self.config, self.tp)

    def ref_param_name(self, name: str) -> str:
        _, c, rest = name.split(".", 2)
        return self.chunks[int(c)].ref_param_name(rest)
