"""Process-group setup and tensor-parallel topology.

A TPContext describes one tensor-parallel group: the NCCL/gloo process
group handle, this rank's position inside it, and its size. Layers take a
TPContext at construction and never touch global state, so a model built
with SINGLE (world=1) is plain PyTorch — that is what the correctness
suite compares against.
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
        torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())


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
