# dstraining

A transformer training framework implementing tensor, sequence, pipeline, and
data parallelism from scratch, benchmarked against Megatron-LM.

**What "from scratch" means here.** Every collective call and all of the
parallelism logic is written in this repo. The allowed primitives are autograd,
`torch.matmul`/cuBLAS, and raw NCCL/gloo collectives via `torch.distributed`.
Not used: `DistributedDataParallel`, FSDP, `torch.distributed.pipelining`,
DeepSpeed, Megatron, or any other parallelism library. No custom CUDA — nothing
here needs a kernel; the matmuls are cuBLAS and the collectives are NCCL, and a
kernel would expand surface area without adding insight.

## Milestones

- [x] **Tensor parallelism** with a numerical correctness suite
- [x] **Sequence parallelism**
- [x] **1F1B pipeline parallelism** (composes with TP and SP)
- [x] **Interleaved pipeline schedule**
- [x] **Selective activation recompute**
- [x] **Data parallelism on top** — full composition verified: dp2 × pp2 × tp2,
      interleaved, with SP, on 8 processes
- [x] **bf16 + fp32 master weights**, memmapped data, `train.py` composing
      everything (loss falls on real tokens through the full stack)
- [ ] Benchmark against Megatron-LM (needs GPUs)

## How tensor parallelism works here

The MLP computes `Z = GeLU(X·A)·B`. Split `A` by columns so each rank computes
`GeLU(X·Aᵢ)` independently — GeLU is elementwise, a column split never mixes
entries. Split `B` by rows so `Z = Y₁B₁ + Y₂B₂`, one all-reduce. Attention
splits the same way, by heads: the q/k/v projections are column-parallel (a
column split is exactly a head split), the output projection is row-parallel.

Correctness comes from one pair of conjugate operators (`dst/ops.py`):

| | forward | backward |
|---|---|---|
| `f` (copy_to_tp_region) | identity | all-reduce |
| `f̄` (reduce_from_tp_region) | all-reduce | identity |

A column-parallel linear applies `f` to its input, a row-parallel linear
applies `f̄` to its output, and every gradient — including those of replicated
parameters like LayerNorms and embeddings, which need no extra sync — comes out
right with no other bookkeeping. The correctness suite asserts this.

Initialization is invariant to the parallel degree: each sharded layer draws
the full weight matrix from a shared seeded generator and keeps its slice, so a
tp=4 model holds exact slices of the tp=1 model's weights. That is what lets
the suite compare a TP model against a plain single-process reference with no
weight copying, and it is why the training loss trajectory is identical
(within fp32 noise) at tp=1, 2, and 4.

## How sequence parallelism works here

LayerNorm and dropout can't split along the hidden dimension, so under plain
TP every rank redundantly holds the full `b·s·h` activation in those regions.
Sequence parallelism (`GPTConfig.sequence_parallel`) shards them along the
sequence axis instead, swapping the conjugate pair:

| | forward | backward |
|---|---|---|
| `g` (gather_along_seq) | all-gather over seq | reduce-scatter over seq |
| `ḡ` (reduce_scatter_along_seq) | reduce-scatter over seq | all-gather over seq |

Since an all-reduce *is* a reduce-scatter followed by an all-gather, total
bandwidth is unchanged — the memory is free. Column-parallel layers enter via
`g`, row-parallel layers exit via `ḡ`, the embedding output is scattered along
seq, and inside the TP region the sequence stays full (attention needs every
key). The suite verifies with forward hooks that every LayerNorm really sees
`[B, T/tp, C]`.

The one cost the conjugate operators don't cover: replicated params that
consume sequence-sharded activations (LayerNorm weights/biases, row-parallel
biases added after the reduce-scatter) get partial gradients — each rank only
sees its sequence chunk. `GPT.finalize_grads()` all-reduces exactly those
grads across the TP group after backward; the suite's gradient checks fail
without it. (Embedding grads are unaffected: the scatter's backward
all-gathers, restoring full gradients.)

gloo has no native reduce-scatter, so the CPU path emulates it with
all-reduce + slice (same result, an all-gather's worth of extra bandwidth);
NCCL uses the real `reduce_scatter_tensor`.

## How pipeline parallelism works here

The model splits by layer into `GPTStage`s: embeddings on the first stage,
`ln_f` + LM head + loss on the last, a contiguous slice of blocks on each.
Ranks lay out as `rank = pp_rank · tp_size + tp_rank`, so a TP group is
consecutive ranks (one node over NVLink on real hardware) and a pipeline
neighbor is the corresponding TP rank `tp_size` away.

The schedule (`dst/pipeline.py`) is 1F1B: after a warmup of `p − 1 − stage`
forwards, each stage alternates one-forward-one-backward, capping in-flight
activations at `p − stage` microbatches regardless of `m`. The bubble
fraction is `(p−1)/m`; interleaving (next milestone) improves it to
`(p−1)/(m·v)`.

Autograd doesn't span processes, so each stage stashes `(input, output)`
pairs for its in-flight microbatches; backward receives `∂loss/∂output` from
downstream, runs autograd over the local segment, and ships `∂loss/∂input`
upstream. Per-microbatch losses are scaled by `1/m`, so the accumulated
gradients equal the full-batch gradient — the suite checks them to ~1e-8.
Each schedule step's sends and receives are posted as one
`batch_isend_irecv`, which is what keeps the steady state deadlock-free
(a blocking send can't wait on a recv that hasn't been posted). Activation
shapes are static per config, so there's no shape handshake; under SP the
boundary tensors are sequence-sharded, so p2p volume divides by `tp` too.

Init is invariant to the pipeline split as well: every component (each
block, each embedding, the head) draws from its own generator seeded by
`(base seed, component id)`, so a stage that builds only its layers gets
bit-identical weights to the full model — same trick that makes TP init
slice-exact, extended to the layer axis.

## The interleaved schedule

Instead of one contiguous block of layers, each rank holds `v`
non-contiguous chunks: virtual stage `k = c·p + r` lives on rank `r` as
chunk `c`, so a microbatch traverses the ranks `v` times and p2p becomes a
ring — rank `p−1` wraps forward activations back to rank 0 at every chunk
boundary. The bubble shrinks from `(p−1)/m` to `(p−1)/(m·v)`, paid for in
`v` times as many, `v` times smaller point-to-point messages. The step
order follows Megatron: `m·v` steps per direction, warmup of
`2(p−1−r) + (v−1)p` forwards, chunk `(step mod pv) ÷ p` per forward step
and the reverse for backward, with `m` divisible by `p`.

Two implementation choices worth noting (`PipelineInterleaved`):

- **Channels.** The ring wrap means one rank pair can carry several message
  streams in the same direction — rank `p−1` sends rank 0 both forward-wrap
  activations and backward grads, all identically shaped — so untagged FIFO
  p2p matching could silently pair a recv with the wrong stream. NCCL has
  no p2p tags; the portable equivalent is a separate process group per
  (direction, chunk), used as a channel.
- **Send discipline.** Sends are fire-and-forget isends, receives block.
  That makes causal consistency of the step order sufficient for
  deadlock-freedom, with no per-step combined-op flag machinery. The suite
  asserts that in-flight activations per rank still stay bounded by the
  warmup depth — the property that separates 1F1B from naive
  all-forward-then-all-backward.

## Data parallelism

The last axis is the simple one: each DP replica holds a full copy of its
(tp, pp)-sharded model, takes its own slice of the global batch, and after
backward every gradient is all-reduced and averaged across the DP group
(`dst/dp.py`). Ranks lay out as `rank = pp·(tp·dp) + dp·tp + tp_rank`, so
TP stays innermost and node-local, DP groups stride by `tp` within a
stage, and pipeline neighbors stride by `tp·dp`. What DDP adds beyond this
is performance engineering — grad bucketing and overlapping the
all-reduce with backward — which lands with the GPU phase.

The DP suite checks the composition of everything at once: the flagship
config is `dp2 × pp2 × tp2` with interleaving and sequence parallelism on
8 processes, where DP-averaged losses, every gradient, and a 5-step Adam
trajectory match the full-batch single-process reference.

## Selective recompute

Core attention — `softmax(QKᵀ/√d)·V` — is written by hand and materializes
the `s×s` attention matrix the way pre-FlashAttention kernels do. That is
deliberate: those tensors are the `s²·b·a`-scaling activations selective
recompute exists to discard, so the framework must actually allocate them
(SDPA would silently never create them and there'd be nothing to measure).

`recompute()` (`dst/recompute.py`) is a 30-line custom autograd Function,
not `torch.utils.checkpoint`: forward runs the region under `no_grad`
saving only q/k/v (alive in the surrounding graph anyway), backward
re-runs it and backpropagates through the fresh subgraph. Requires the
region to be deterministic and RNG-free, which core attention here is.

The test measures what autograd actually stashes via
`saved_tensors_hooks`: with `recompute_attention` on, saved-for-backward
bytes drop by the analytic `2·s²·b·a` attention-matrix term (4,210,688
observed vs 4,194,304 predicted for the test config), gradients are
bitwise identical, and the region runs exactly twice per layer per
microbatch. The TP suite's `--recompute` flag puts recompute on the
parallel model only, so every comparison against the full-activation
reference cross-validates it under TP and SP too.

## Mixed precision and training

`MasterWeightOptimizer` (`dst/precision.py`): bf16 params/activations/
grads, fp32 master weights and Adam state. A weight update of `lr·grad`
~1e-4 of the weight underflows bf16's 8 mantissa bits — applied in fp32
and rounded once per step, the signal accumulates. bf16 needs no loss
scaling (that's fp16's narrow-exponent problem). The test suite
demonstrates the failure mode live: the same model with plain-bf16 Adam
lands measurably behind, while bf16+masters tracks the fp32 trajectory to
a ~0.006 final-loss gap.

`train.py` composes every axis behind flags; data is a memmapped uint16
token shard (`scripts/prepare_openwebtext.py`, nanoGPT convention) or
`--data synthetic`. Batches are drawn at offsets seeded by
`(seed, step, dp_rank)`: every rank of one DP replica computes the
identical batch with zero communication (first stage needs the tokens,
last stage the targets), while DP replicas draw independent data.

```
scripts/launch_local.sh 8 train.py --dp 2 --pp 2 --tp 2 --chunks 2 \
    --sp --recompute --bf16 --micro 2 --steps 40 --data <shard.bin>
```

runs everything at once; on GPUs, launch the same file with torchrun.

## Layout

```
dst/parallel.py   process groups and topology (TPContext, PPContext)
dst/ops.py        f/f̄ (TP), g/ḡ + scatter (SP), vocab gather — all collectives
dst/layers.py     ColumnParallelLinear, RowParallelLinear
dst/model.py      GPT-2 from scratch: GPT, GPTStage (one stage), GPTChunks (v chunks)
dst/pipeline.py   1F1B and interleaved schedules, p2p activation/gradient exchange
dst/recompute.py  selective activation recompute (custom autograd Function)
dst/dp.py         data-parallel gradient all-reduce
dst/precision.py  bf16 + fp32 master weights
dst/data.py       memmapped token shard, deterministic batch draws
train.py          training entrypoint composing every axis
tests/            numerical correctness suites
scripts/          launchers, data prep
```

## Running the correctness suite

On a CUDA machine:

```
torchrun --standalone --nproc_per_node=2 tests/test_tp_correctness.py
```

On a CPU-only machine (including macOS, where torchrun's TCP rendezvous can
hang on DNS — the local launcher uses a FileStore instead):

```
scripts/launch_local.sh 2 tests/test_tp_correctness.py
```

Add `--sp` to run the same suite with sequence parallelism enabled, and
`--recompute` to enable selective recompute on the parallel model only.
The recompute memory/equivalence test is single-process:
`python3 tests/test_recompute.py`.

The pipeline suite composes all three axes; TP degree × PP degree must
equal the process count:

```
scripts/launch_local.sh 2 tests/test_pp_correctness.py --pp 2 --micro 1
scripts/launch_local.sh 4 tests/test_pp_correctness.py --pp 4 --micro 1
scripts/launch_local.sh 4 tests/test_pp_correctness.py --pp 2 --tp 2 --micro 1 --sp
scripts/launch_local.sh 2 tests/test_pp_correctness.py --pp 2 --chunks 2 --micro 1
scripts/launch_local.sh 4 tests/test_pp_correctness.py --pp 4 --chunks 2 --layers 8 --micro 1
scripts/launch_local.sh 4 tests/test_pp_correctness.py --pp 2 --tp 2 --chunks 2 --micro 1 --sp
```

It checks the pipelined microbatched loss, every accumulated gradient, and
a 5-step Adam trajectory against the full-batch single-process reference.

The DP suite composes all four axes:

```
scripts/launch_local.sh 2 tests/test_dp_correctness.py --dp 2
scripts/launch_local.sh 4 tests/test_dp_correctness.py --dp 2 --tp 2 --sp
scripts/launch_local.sh 4 tests/test_dp_correctness.py --dp 2 --pp 2 --micro 1
scripts/launch_local.sh 8 tests/test_dp_correctness.py --dp 2 --pp 2 --tp 2 --chunks 2 --micro 1 --sp
```

For the TP suite, TP degree = number of processes. The suite checks, in fp32:
conjugacy of `f`/`f̄` (and `g`/`ḡ` under `--sp`) on a toy split,
shard-vs-slice init identity, sequence sharding of LayerNorm regions,
forward logits vs the reference and across ranks, every gradient shard vs
the sliced reference gradient, and a 10-step Adam trajectory that must
track the reference and fall.

## Notes and deliberate scope cuts (so far)

- q/k/v are three separate column-parallel linears rather than one fused
  projection; fusing needs Megatron's interleaved per-head weight layout and
  buys only matmul batching. Revisit when profiling.
- The LM head is column-parallel over a vocab padded to a multiple of 128
  (fixed, so init stays tp-invariant) with logits gathered before the loss.
  Vocab-parallel cross-entropy — which avoids materializing full logits —
  comes later.
- No embedding/LM-head weight tying: the head holds a vocab shard while the
  embedding is replicated.
- Dropout defaults to 0. Replicated regions need shared RNG state across TP
  ranks and parallel regions need per-rank state; under SP the dropout
  regions are sequence-sharded (disjoint chunks), which makes per-rank RNG
  correct there — but the shared-state discipline for the plain-TP path is
  still unimplemented, so keep dropout at 0 for now.
