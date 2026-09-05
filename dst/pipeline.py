"""1F1B pipeline schedule.

The naive schedule (all forwards, then all backwards) holds every
microbatch's activations at once. 1F1B caps that: after a warmup of
(p - 1 - stage) forwards, each stage alternates one-forward-one-backward,
so at most (p - stage) microbatches are ever in flight, independent of m.
The bubble fraction stays (p-1)/m — raising m shrinks it until memory
limits you (interleaving improves on this and comes next).

Autograd does not span processes, so each stage keeps (input, output)
pairs for its in-flight microbatches. Backward for a microbatch receives
d(loss)/d(output) from the next stage, runs autograd over the local
segment, and sends d(loss)/d(input) to the previous stage. The last
stage's backward starts from its own loss (scaled by 1/m so gradients
accumulated over microbatches equal the full-batch gradient).

All p2p transfers go through _exchange, which posts the receives and
sends of one schedule step as a single batch_isend_irecv. That is what
makes the steady state deadlock-free: when stage i sends a forward
activation to i+1 while i+1 sends a gradient back, both pairs are in
flight in the same batch, so neither blocking send can wait on a recv
that hasn't been posted. Activation shapes are static per config —
(micro_batch, seq[/tp under SP], n_embd) — so no shape handshake is
needed.

Works with tp >= 1 inside each stage: all TP ranks of a stage run the
schedule in lockstep and exchange with their counterpart TP rank in the
neighbor stage. Under SP the boundary tensor is sequence-sharded, so p2p
volume divides by tp too.
"""

from collections import deque
from typing import Optional

import torch
import torch.distributed as dist

from .model import GPTStage, GPTChunks
from .parallel import PPContext


def _exchange(recvs, sends):
    """Post all receives and sends of one schedule step as a single batch.
    recvs/sends: lists of (tensor, global peer rank)."""
    ops = [dist.P2POp(dist.irecv, t, peer) for t, peer in recvs]
    ops += [dist.P2POp(dist.isend, t.contiguous(), peer) for t, peer in sends]
    if ops:
        for req in dist.batch_isend_irecv(ops):
            req.wait()


class Pipeline1F1B:
    def __init__(self, stage: GPTStage, pp: PPContext, micro_batch_size: int, seq_len: int):
        cfg = stage.config
        self.stage = stage
        self.pp = pp
        seq_local = seq_len // stage.tp.world if cfg.sequence_parallel else seq_len
        self.act_shape = (micro_batch_size, seq_local, cfg.n_embd)
        self.micro_batch_size = micro_batch_size
        # p2p buffers must match the activation dtype (bf16 under mixed
        # precision); the params' dtype is the activations' dtype here.
        self.act_dtype = next(stage.parameters()).dtype

    # -- p2p helpers ------------------------------------------------------

    def _recv_forward(self) -> Optional[torch.Tensor]:
        if self.pp.is_first:
            return None
        x = torch.empty(self.act_shape, dtype=self.act_dtype)
        _exchange([(x, self.pp.prev_rank)], [])
        return x

    def _recv_backward(self) -> Optional[torch.Tensor]:
        if self.pp.is_last:
            return None
        g = torch.empty(self.act_shape, dtype=self.act_dtype)
        _exchange([(g, self.pp.next_rank)], [])
        return g

    def _send_forward(self, out: torch.Tensor) -> None:
        if not self.pp.is_last:
            _exchange([], [(out.detach(), self.pp.next_rank)])

    def _send_backward(self, in_grad: Optional[torch.Tensor]) -> None:
        if not self.pp.is_first:
            _exchange([], [(in_grad, self.pp.prev_rank)])

    def _send_forward_recv_backward(self, out: torch.Tensor) -> Optional[torch.Tensor]:
        if self.pp.is_last:
            return None
        g = torch.empty(self.act_shape, dtype=self.act_dtype)
        _exchange([(g, self.pp.next_rank)], [(out.detach(), self.pp.next_rank)])
        return g

    def _send_backward_recv_forward(self, in_grad: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if self.pp.is_first:
            return None
        x = torch.empty(self.act_shape, dtype=self.act_dtype)
        _exchange([(x, self.pp.prev_rank)], [(in_grad, self.pp.prev_rank)])
        return x

    # -- compute steps ----------------------------------------------------

    def _forward_step(self, x, micro_idx, micro_targets, n_micro, losses):
        """Returns (input_tensor, output_tensor) to stash for backward."""
        if self.pp.is_first:
            inp = None
            x = micro_idx
        else:
            inp = x.requires_grad_(True)
        if self.pp.is_last:
            _, loss = self.stage(x, micro_targets)
            loss = loss / n_micro
            losses.append(loss.detach())
            return inp, loss
        return inp, self.stage(x)

    def _backward_step(self, inp, out, out_grad) -> Optional[torch.Tensor]:
        torch.autograd.backward(out, grad_tensors=out_grad)
        return inp.grad if inp is not None else None

    # -- the schedule -----------------------------------------------------

    def train_step(self, idx: torch.Tensor, targets: torch.Tensor):
        """One full batch: split into microbatches along the batch dim, run
        1F1B, leave accumulated gradients on the stage's params. Returns the
        mean microbatch loss on the last stage, None elsewhere. The caller
        zeroes grads, then calls this, then stage.finalize_grads(), then the
        optimizer."""
        B = idx.shape[0]
        if B % self.micro_batch_size != 0:
            raise ValueError(f"batch {B} not divisible by micro batch {self.micro_batch_size}")
        n_micro = B // self.micro_batch_size
        micro_idx = idx.split(self.micro_batch_size)
        micro_tgt = targets.split(self.micro_batch_size)

        p, s = self.pp.world, self.pp.rank
        n_warmup = min(p - 1 - s, n_micro)
        n_steady = n_micro - n_warmup

        in_flight = deque()  # (input, output) pairs, oldest first
        losses = []
        fwd_i = 0  # next microbatch to run forward

        for _ in range(n_warmup):
            x = self._recv_forward()
            pair = self._forward_step(x, micro_idx[fwd_i], micro_tgt[fwd_i], n_micro, losses)
            self._send_forward(pair[1])
            in_flight.append(pair)
            fwd_i += 1

        if n_steady > 0:
            x = self._recv_forward()
        for i in range(n_steady):
            pair = self._forward_step(x, micro_idx[fwd_i], micro_tgt[fwd_i], n_micro, losses)
            fwd_i += 1
            out_grad = self._send_forward_recv_backward(pair[1])
            in_flight.append(pair)
            inp, out = in_flight.popleft()
            in_grad = self._backward_step(inp, out, out_grad)
            if i < n_steady - 1:
                x = self._send_backward_recv_forward(in_grad)
            else:
                self._send_backward(in_grad)

        for _ in range(n_warmup):
            out_grad = self._recv_backward()
            inp, out = in_flight.popleft()
            in_grad = self._backward_step(inp, out, out_grad)
            self._send_backward(in_grad)

        assert not in_flight and fwd_i == n_micro
        if self.pp.is_last:
            return torch.stack(losses).sum()
        return None


class PipelineInterleaved:
    """Interleaved 1F1B (Megatron's interleaved schedule).

    Each rank holds v non-contiguous model chunks; virtual stage k = c*p + r
    lives on rank r as chunk c, so a microbatch traverses the ranks v times
    and p2p becomes a ring (rank p-1 wraps forward activations back to rank
    0 at each chunk boundary). The bubble shrinks from (p-1)/m to
    (p-1)/(m*v), paid for in v times as many, v times smaller p2p messages.

    Schedule, following Megatron: tnm = m*v steps per direction, warmup of
    min(2*(p-1-r) + (v-1)*p, tnm) forwards, then 1F1B, then cooldown.
    Step i runs chunk (i % (p*v)) // p forward; backward steps run the
    chunks in reverse. Requires m % p == 0 (Megatron's constraint: chunk
    rotation happens in groups of p microbatches).

    Two implementation choices differ from the non-interleaved schedule:

    Channels. The ring wrap means one rank pair can carry several message
    streams in the same direction (rank p-1 sends both forward-wrap
    activations and backward grads to rank 0), and every activation has the
    same shape, so untagged FIFO matching could silently pair a recv with
    the wrong stream. NCCL has no p2p tags; the portable equivalent is a
    separate process group per (direction, chunk), which this class creates
    as communication channels.

    Send discipline. Sends are fire-and-forget isends (buffers held until
    the step ends), receives block. Fire-and-forget makes causal
    consistency of the step order sufficient for deadlock-freedom — no
    per-step combined-op flag machinery — at the cost of not bounding send
    buffering by the schedule. In-flight activations per rank stay bounded
    by the warmup depth regardless (peak_in_flight records the observed
    maximum so tests can assert it).
    """

    def __init__(self, model: GPTChunks, pp: PPContext, micro_batch_size: int, seq_len: int):
        cfg = model.config
        self.model = model
        self.pp = pp
        self.v = model.v
        seq_local = seq_len // model.tp.world if cfg.sequence_parallel else seq_len
        self.act_shape = (micro_batch_size, seq_local, cfg.n_embd)
        self.micro_batch_size = micro_batch_size
        self.act_dtype = next(model.parameters()).dtype

        world = dist.get_world_size()
        everyone = list(range(world))
        self.fwd_ch = [dist.new_group(everyone) for _ in range(self.v)]
        self.bwd_ch = [dist.new_group(everyone) for _ in range(self.v)]

        self.ring_next = pp.ring_next_rank
        self.ring_prev = pp.ring_prev_rank

        self.peak_in_flight = 0

    def _is_vfirst(self, c: int) -> bool:
        return self.pp.rank == 0 and c == 0

    def _is_vlast(self, c: int) -> bool:
        return self.pp.rank == self.pp.world - 1 and c == self.v - 1

    def _recv(self, peer: int, group) -> torch.Tensor:
        x = torch.empty(self.act_shape, dtype=self.act_dtype)
        dist.recv(x, src=peer, group=group)
        return x

    def _isend(self, t: torch.Tensor, peer: int, group, pending) -> None:
        t = t.detach().contiguous()
        pending.append((dist.isend(t, dst=peer, group=group), t))

    def train_step(self, idx: torch.Tensor, targets: torch.Tensor):
        """Same contract as Pipeline1F1B.train_step."""
        B = idx.shape[0]
        if B % self.micro_batch_size != 0:
            raise ValueError(f"batch {B} not divisible by micro batch {self.micro_batch_size}")
        m = B // self.micro_batch_size
        p, r, v = self.pp.world, self.pp.rank, self.v
        if m % p != 0:
            raise ValueError(f"interleaving needs microbatches {m} divisible by stages {p}")
        micro_idx = idx.split(self.micro_batch_size)
        micro_tgt = targets.split(self.micro_batch_size)

        tnm = m * v
        warmup = min(2 * (p - 1 - r) + (v - 1) * p, tnm)

        queues = [deque() for _ in range(v)]  # (input, output) per chunk
        pending = []  # (isend work, buffer) pairs kept alive
        losses = []
        feed_i = 0  # next data microbatch into the virtual-first chunk
        tgt_i = 0  # next target microbatch at the virtual-last chunk

        def chunk_of(step: int, forward: bool) -> int:
            c = (step % (p * v)) // p
            return c if forward else v - 1 - c

        def fwd_one(step: int) -> None:
            nonlocal feed_i, tgt_i
            c = chunk_of(step, True)
            chunk = self.model.chunks[c]
            if self._is_vfirst(c):
                inp, x = None, micro_idx[feed_i]
                feed_i += 1
            else:
                inp = self._recv(self.ring_prev, self.fwd_ch[c]).requires_grad_(True)
                x = inp
            if self._is_vlast(c):
                _, loss = chunk(x, micro_tgt[tgt_i])
                tgt_i += 1
                out = loss / m
                losses.append(out.detach())
            else:
                out = chunk(x)
                dest_c = c if r < p - 1 else c + 1
                self._isend(out, self.ring_next, self.fwd_ch[dest_c], pending)
            queues[c].append((inp, out))
            self.peak_in_flight = max(self.peak_in_flight, sum(len(q) for q in queues))

        def bwd_one(step: int) -> None:
            c = chunk_of(step, False)
            if self._is_vlast(c):
                out_grad = None
            else:
                src_c = c if r < p - 1 else c + 1
                out_grad = self._recv(self.ring_next, self.bwd_ch[src_c])
            inp, out = queues[c].popleft()
            torch.autograd.backward(out, grad_tensors=out_grad)
            if not self._is_vfirst(c):
                self._isend(inp.grad, self.ring_prev, self.bwd_ch[c], pending)

        for i in range(warmup):
            fwd_one(i)
        for j in range(tnm - warmup):
            fwd_one(warmup + j)
            bwd_one(j)
        for j in range(tnm - warmup, tnm):
            bwd_one(j)

        for work, _ in pending:
            work.wait()
        assert all(not q for q in queues) and feed_i in (0, m)

        if self.pp.is_last:
            return torch.stack(losses).sum()
        return None
