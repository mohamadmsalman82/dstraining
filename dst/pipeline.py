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

from .model import GPTStage
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

    # -- p2p helpers ------------------------------------------------------

    def _recv_forward(self) -> Optional[torch.Tensor]:
        if self.pp.is_first:
            return None
        x = torch.empty(self.act_shape)
        _exchange([(x, self.pp.prev_rank)], [])
        return x

    def _recv_backward(self) -> Optional[torch.Tensor]:
        if self.pp.is_last:
            return None
        g = torch.empty(self.act_shape)
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
        g = torch.empty(self.act_shape)
        _exchange([(g, self.pp.next_rank)], [(out.detach(), self.pp.next_rank)])
        return g

    def _send_backward_recv_forward(self, in_grad: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if self.pp.is_first:
            return None
        x = torch.empty(self.act_shape)
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
