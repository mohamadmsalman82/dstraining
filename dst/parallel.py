"""Process-group setup and parallel topology.

A TPContext describes one tensor-parallel group: the NCCL/gloo process
group handle, this rank's position inside it, and its size. Layers take a
TPContext at construction and never touch global state, so a model built
with SINGLE (world=1) is plain PyTorch — that is what the correctness
suite compares against.

A PPContext describes this rank's pipeline stage and its neighbors as
GLOBAL ranks (p2p ops address global ranks; no group handle needed).

Rank layout: rank = pp_rank * tp_size + tp_rank. TP varies fastest, so a
TP group is consecutive ranks (same node over NVLink under standard
torchrun placement — TP traffic is the bandwidth-hungry kind), and a
pipeline neighbor is the corresponding TP rank tp_size away.
"""

import os
import sys
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class TPContext:
    group: Optional[object]  # ProcessGroup, or None when world == 1
    rank: int
    world: int

    @property
    def enabled(self) -> bool:
        return self.world > 1


# A null context: tp of size 1, no collectives ever issued.
SINGLE = TPContext(group=None, rank=0, world=1)


@dataclass(frozen=True)
class PPContext:
    rank: int  # stage index
    world: int  # number of stages
    prev_rank: Optional[int]  # global rank holding the previous stage
    next_rank: Optional[int]  # global rank holding the next stage
    # Ring neighbors for the interleaved schedule: same as prev/next but
    # wrapping around, so always defined (self for world == 1).
    ring_prev_rank: int = 0
    ring_next_rank: int = 0

    @property
    def is_first(self) -> bool:
        return self.rank == 0

    @property
    def is_last(self) -> bool:
        return self.rank == self.world - 1


PP_SINGLE = PPContext(rank=0, world=1, prev_rank=None, next_rank=None)

# Data parallelism needs the same (group, rank, world) triple as TP.
DPContext = TPContext
DP_SINGLE = SINGLE


def init_distributed(backend: Optional[str] = None) -> None:
    """Initialize torch.distributed from torchrun's environment variables.

    NCCL on CUDA machines, gloo on CPU-only machines (the two code paths
    are otherwise identical, so all correctness work runs on CPU).
    """
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    if backend == "gloo" and sys.platform == "darwin":
        # Keep gloo on loopback; macOS reverse-DNS lookups can hang it.
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0")

    # torchrun's TCP rendezvous also hangs on macOS DNS; scripts/launch_local.sh
    # sets DST_INIT_METHOD to a FileStore path instead, which needs no sockets.
    init_method = os.environ.get("DST_INIT_METHOD")
    if init_method:
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            rank=int(os.environ["RANK"]),
            world_size=int(os.environ["WORLD_SIZE"]),
        )
    else:
        dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        # LOCAL_RANK is torchrun's per-node rank; correct under multi-node
        # where global rank % device_count would collide.
        local = int(os.environ.get("LOCAL_RANK", dist.get_rank() % torch.cuda.device_count()))
        torch.cuda.set_device(local)


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def make_tp_context(tp_size: Optional[int] = None) -> TPContext:
    """Partition the world into tensor-parallel groups of consecutive ranks.

    Consecutive ranks (0..tp-1, tp..2tp-1, ...) form a TP group because TP
    traffic is the bandwidth-hungry kind and consecutive ranks land on the
    same node under standard torchrun placement. dist.new_group must be
    called by every rank for every group, hence the full loop.
    """
    world = dist.get_world_size()
    rank = dist.get_rank()
    if tp_size is None:
        tp_size = world
    if world % tp_size != 0:
        raise ValueError(f"world size {world} not divisible by tp size {tp_size}")

    if tp_size == 1:
        return SINGLE

    ctx = None
    for start in range(0, world, tp_size):
        ranks = list(range(start, start + tp_size))
        group = dist.new_group(ranks)
        if rank in ranks:
            ctx = TPContext(group=group, rank=rank - start, world=tp_size)
    assert ctx is not None
    return ctx


def make_topology(tp_size: int = 1, pp_size: int = 1, dp_size: int = 1):
    """Decompose the world into (tp, pp, dp) contexts.

    rank = pp_rank * (tp * dp) + dp_rank * tp + tp_rank — Megatron's order:
    TP innermost (consecutive ranks, node-local, bandwidth-hungry), DP
    groups striding by tp within a stage, pipeline neighbors striding by
    tp * dp (across nodes, tiny p2p traffic).
    """
    world = dist.get_world_size()
    rank = dist.get_rank()
    if tp_size * pp_size * dp_size != world:
        raise ValueError(f"tp {tp_size} * pp {pp_size} * dp {dp_size} != world {world}")

    tp = make_tp_context(tp_size)

    stride = tp_size * dp_size
    pp_rank = rank // stride
    pp = PPContext(
        rank=pp_rank,
        world=pp_size,
        prev_rank=rank - stride if pp_rank > 0 else None,
        next_rank=rank + stride if pp_rank < pp_size - 1 else None,
        ring_prev_rank=rank - pp_rank * stride + ((pp_rank - 1) % pp_size) * stride,
        ring_next_rank=rank - pp_rank * stride + ((pp_rank + 1) % pp_size) * stride,
    )

    if dp_size == 1:
        return tp, pp, DP_SINGLE

    dp = None
    dp_rank = (rank // tp_size) % dp_size
    for pp_r in range(pp_size):
        for tp_r in range(tp_size):
            ranks = [pp_r * stride + d * tp_size + tp_r for d in range(dp_size)]
            group = dist.new_group(ranks)
            if rank in ranks:
                dp = DPContext(group=group, rank=dp_rank, world=dp_size)
    assert dp is not None
    return tp, pp, dp
