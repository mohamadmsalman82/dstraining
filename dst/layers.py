"""Tensor-parallel linear layers.

Both layers draw the FULL weight matrix from the caller's seeded generator
and keep only their shard. Every rank draws the same numbers in the same
order, so shards are consistent across ranks and — more usefully — a model
built at tp=2 holds exact slices of the weights a tp=1 model holds under
the same seed. Initialization is invariant to the parallel degree, which
is what lets the correctness suite compare against a plain single-process
reference without any weight copying.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .parallel import TPContext
from .ops import (
    copy_to_tp_region,
    reduce_from_tp_region,
    gather_from_tp_region,
    gather_along_seq,
    reduce_scatter_along_seq,
)


class ColumnParallelLinear(nn.Module):
    """Y = XA with A split by columns: A = [A_1 | A_2 | ...].

    Each rank computes X @ A_i independently. Output stays sharded along
    the last dim unless gather_output is set. In torch's Linear convention
    (weight is [out, in], y = x @ W.T) a column split of A is a row split
    of the weight tensor.

    input_mode selects the entry operator:
      "replicated" — input is full on every rank; apply f (identity fwd,
                     all-reduce bwd). Plain tensor parallelism.
      "sequence"   — input arrives sequence-sharded; apply g (all-gather
                     fwd, reduce-scatter bwd). Sequence parallelism.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tp: TPContext,
        *,
        bias: bool = True,
        gather_output: bool = False,
        input_mode: str = "replicated",
        init_std: float = 0.02,
        generator: torch.Generator = None,
    ):
        super().__init__()
        if out_features % tp.world != 0:
            raise ValueError(f"out_features {out_features} not divisible by tp {tp.world}")
        assert input_mode in ("replicated", "sequence")
        self.tp = tp
        self.gather_output = gather_output
        self.input_mode = input_mode
        self.in_features = in_features
        self.out_features = out_features
        out_local = out_features // tp.world

        full = torch.empty(out_features, in_features)
        full.normal_(0.0, init_std, generator=generator)
        shard = full[tp.rank * out_local : (tp.rank + 1) * out_local]
        self.weight = nn.Parameter(shard.clone())
        self.bias = nn.Parameter(torch.zeros(out_local)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_mode == "sequence":
            x = gather_along_seq(x, self.tp)
        else:
            x = copy_to_tp_region(x, self.tp)
        y = F.linear(x, self.weight, self.bias)
        if self.gather_output:
            y = gather_from_tp_region(y, self.tp)
        return y


class RowParallelLinear(nn.Module):
    """Z = YB with B split by rows: Z = Y_1 B_1 + Y_2 B_2 + ...

    Input arrives already sharded along the last dim (the output of a
    column-parallel layer). Each rank computes its partial product; the
    partials are summed across ranks. The bias is added after the sum.

    output_mode selects the exit operator:
      "replicated" — f-bar: all-reduce fwd, identity bwd; output is full
                     on every rank. Plain tensor parallelism.
      "sequence"   — g-bar: reduce-scatter fwd, all-gather bwd; output
                     lands sequence-sharded. Sequence parallelism.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tp: TPContext,
        *,
        bias: bool = True,
        output_mode: str = "replicated",
        init_std: float = 0.02,
        generator: torch.Generator = None,
    ):
        super().__init__()
        if in_features % tp.world != 0:
            raise ValueError(f"in_features {in_features} not divisible by tp {tp.world}")
        assert output_mode in ("replicated", "sequence")
        self.output_mode = output_mode
        self.tp = tp
        self.in_features = in_features
        self.out_features = out_features
        in_local = in_features // tp.world

        full = torch.empty(out_features, in_features)
        full.normal_(0.0, init_std, generator=generator)
        shard = full[:, tp.rank * in_local : (tp.rank + 1) * in_local]
        self.weight = nn.Parameter(shard.clone())
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        if self.bias is not None and output_mode == "sequence":
            # The bias is added to a sequence-sharded output, so its grad on
            # each rank covers only that rank's sequence chunk; it must be
            # all-reduced across the TP group after backward.
            self.bias.sequence_parallel_replicated = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight)
        if self.output_mode == "sequence":
            y = reduce_scatter_along_seq(y, self.tp)
        else:
            y = reduce_from_tp_region(y, self.tp)
        if self.bias is not None:
            y = y + self.bias
        return y
