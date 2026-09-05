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
from .ops import copy_to_tp_region, reduce_from_tp_region, gather_from_tp_region


class ColumnParallelLinear(nn.Module):
    """Y = XA with A split by columns: A = [A_1 | A_2 | ...].

    Each rank computes X @ A_i independently (input is replicated, f makes
    the backward correct). Output stays sharded along the last dim unless
    gather_output is set. In torch's Linear convention (weight is [out, in],
    y = x @ W.T) a column split of A is a row split of the weight tensor.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tp: TPContext,
        *,
        bias: bool = True,
        gather_output: bool = False,
        init_std: float = 0.02,
        generator: torch.Generator = None,
    ):
        super().__init__()
        if out_features % tp.world != 0:
            raise ValueError(f"out_features {out_features} not divisible by tp {tp.world}")
        self.tp = tp
        self.gather_output = gather_output
        self.in_features = in_features
        self.out_features = out_features
        out_local = out_features // tp.world

        full = torch.empty(out_features, in_features)
        full.normal_(0.0, init_std, generator=generator)
        shard = full[tp.rank * out_local : (tp.rank + 1) * out_local]
        self.weight = nn.Parameter(shard.clone())
        self.bias = nn.Parameter(torch.zeros(out_local)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = copy_to_tp_region(x, self.tp)
        y = F.linear(x, self.weight, self.bias)
        if self.gather_output:
            y = gather_from_tp_region(y, self.tp)
        return y


class RowParallelLinear(nn.Module):
    """Z = YB with B split by rows: Z = Y_1 B_1 + Y_2 B_2 + ...

    Input arrives already sharded along the last dim (the output of a
    column-parallel layer). Each rank computes its partial product and
    f-bar sums them: one all-reduce forward, identity backward. The bias
    is added after the reduce, once, identically on every rank.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tp: TPContext,
        *,
        bias: bool = True,
        init_std: float = 0.02,
        generator: torch.Generator = None,
    ):
        super().__init__()
        if in_features % tp.world != 0:
            raise ValueError(f"in_features {in_features} not divisible by tp {tp.world}")
        self.tp = tp
        self.in_features = in_features
        self.out_features = out_features
        in_local = in_features // tp.world

        full = torch.empty(out_features, in_features)
        full.normal_(0.0, init_std, generator=generator)
        shard = full[:, tp.rank * in_local : (tp.rank + 1) * in_local]
        self.weight = nn.Parameter(shard.clone())
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight)
        y = reduce_from_tp_region(y, self.tp)
        if self.bias is not None:
            y = y + self.bias
        return y
