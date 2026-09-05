"""bf16 training with fp32 master weights.

The model's params, activations, and gradients live in bf16; the
optimizer state and the authoritative weights live in fp32. Each step:

  1. bf16 grads are upcast into the master params' .grad
  2. Adam steps the fp32 masters
  3. masters are cast back down into the bf16 model params

Why: bf16 has fp32's range but only 8 mantissa bits, so a weight update
of lr * grad ~ 1e-4 of the weight underflows if applied in bf16 — the
update vanishes and training stalls. Applying updates in fp32 and
rounding once keeps the accumulated signal. bf16 needs no loss scaling
(that is an fp16 problem: fp16's narrow exponent underflows small grads;
bf16's doesn't).

The wrapper is optimizer-agnostic plumbing around torch.optim.Adam —
the from-scratch boundary of this project is the parallelism, not Adam's
update rule.
"""

from typing import Iterable

import torch


class MasterWeightOptimizer:
    def __init__(self, params: Iterable[torch.nn.Parameter], **adam_kwargs):
        self.model_params = [p for p in params if p.requires_grad]
        self.master_params = [
            p.detach().clone().float().requires_grad_(False) for p in self.model_params
        ]
        self.inner = torch.optim.Adam(self.master_params, **adam_kwargs)

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.model_params:
            p.grad = None if set_to_none else torch.zeros_like(p)

    def step(self) -> None:
        for mp, master in zip(self.model_params, self.master_params):
            master.grad = None if mp.grad is None else mp.grad.float()
        self.inner.step()
        for mp, master in zip(self.model_params, self.master_params):
            mp.data.copy_(master)  # one bf16 rounding per step, of the fp32 truth
