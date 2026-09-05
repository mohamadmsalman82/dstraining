# dstraining: the full write-up

*How a distributed training framework got built from scratch, verified on a
laptop with no GPU, validated and benchmarked against Megatron-LM on rented
hardware, and what it cost. September 4-5, 2026.*

---

## 1. What the project was

Modern language models don't fit on one GPU. A 1.3B model in bf16 with
optimizer state and activations already strains an 80GB card, and frontier
models are three orders of magnitude larger. Training gets split across many
GPUs, and there are only a handful of ways to split. Each cuts along a
different axis, and each buys memory at the price of communication:

- **Data parallelism** replicates the model and splits the batch. Simple, but
  useless once the model itself doesn't fit.
- **Tensor parallelism** splits the weight matrices themselves — the MLP's
  `Z = GeLU(X·A)·B` splits `A` by columns (GeLU is elementwise, a column split
  never mixes entries) and `B` by rows (a sum across GPUs, one all-reduce).
  Attention splits by heads. Bandwidth-hungry; belongs inside one node.
- **Pipeline parallelism** splits by layer. Communication is tiny but GPUs
  idle waiting on the stage ahead — the bubble, `(p−1)/m` of the step for p
  stages and m microbatches. Interleaving improves it to `(p−1)/(m·v)`.
- **Sequence parallelism** patches TP's memory leak: LayerNorm and dropout
  can't split along the hidden dimension, so those regions shard along the
  sequence axis instead, for free bandwidth-wise.
- **Selective recompute** discards the `s²·b·a`-scaling attention activations
  and rebuilds them in backward for a couple percent FLOP overhead.

The project: implement all of it from scratch, prove every piece numerically
correct, compose everything into one trainer, and benchmark the result against
Megatron-LM — the NVIDIA framework these techniques come from.

**The from-scratch boundary**, stated up front because it's the first thing
anyone asks: every collective call and all parallelism logic is written in
this repo. Allowed primitives: autograd, `torch.matmul`/cuBLAS, raw NCCL/gloo
collectives via `torch.distributed`. Not used: `DistributedDataParallel`,
FSDP, `torch.distributed.pipelining`, DeepSpeed, Megatron, or any parallelism
library. No custom CUDA — nothing here needs a kernel; adding one would expand
surface area without adding insight.

## 2. The method that made it tractable

Three decisions early on did most of the heavy lifting.

**CPU-first development.** `torch.distributed` exposes the same collectives on
gloo (CPU) as on NCCL (GPU), so all correctness work ran as multi-process jobs
on a MacBook with zero GPUs. Every schedule, every operator, every composition
was debugged at fp32 tolerances on CPU before a single GPU-hour was spent.
When the code finally touched NCCL, all five suites passed unchanged on the
first run. (One macOS wrinkle: torchrun's TCP rendezvous hangs on DNS there,
so `scripts/launch_local.sh` spawns ranks over a FileStore, which needs no
sockets. On Linux/GPU it's plain torchrun; the code paths are identical.)

**Decomposition-invariant initialization.** The framework's whole verification
strategy rests on one trick: a model split *any* way holds bit-identical
weights to the unsplit model under the same seed.

- TP-invariance: sharded layers draw the *full* weight matrix from a seeded
  generator and keep their slice, so a tp=4 model holds exact slices of the
  tp=1 model's weights.
- PP-invariance: every component (each block, each embedding, the head) draws
  from its own generator seeded by `(base_seed, component_id)`, so a pipeline
  stage that builds only its layers draws exactly what the full model drew for
  them.

Consequence: every rank can construct the full single-process reference
locally and compare — losses, every gradient shard against the corresponding
reference slice, and multi-step Adam trajectories — with no weight copying, no
checkpoint plumbing, no broadcast. This is what let the suites make claims
like "every gradient matches to 1e-8" cheaply.

**Correctness suites as the product.** Each milestone shipped with a suite
that would fail loudly if the math was wrong, and they did fail, usefully
(section 6). Five suites: TP (with SP/recompute/vocab-parallel flags), PP
(both schedules, composed with TP/SP), DP (composed with everything),
recompute (memory measured, not asserted), precision (the bf16 failure mode
demonstrated live).

## 3. What was built, in order

Eleven working sessions' worth of milestones, each committed and pushed as it
passed its suite.

### 3.1 Tensor parallelism (`dst/ops.py`, `dst/layers.py`)

The conjugate operator pair is the entire correctness story:

| | forward | backward |
|---|---|---|
| `f` (copy_to_tp_region) | identity | all-reduce |
| `f̄` (reduce_from_tp_region) | all-reduce | identity |

`ColumnParallelLinear` applies `f` to its input and holds a row-shard of the
weight (a column split of `A` in math convention); `RowParallelLinear` holds a
column-shard and applies `f̄` to its output, bias added after the reduce. Wrap
attention and MLP in the pair and every gradient comes out right with no other
bookkeeping — including replicated parameters (LayerNorms, embeddings), whose
gradients emerge identical on every rank because `f`'s backward all-reduce
restores the full gradient at each TP boundary. The suite asserts this
explicitly rather than assuming it.

Attention splits by heads. Initially q/k/v were three separate column-parallel
layers (obviously correct, easy to verify); the perf pass later fused them
(3.8). The LM head is column-parallel over a vocab padded to a multiple of 128
regardless of tp — fixed padding keeps init draws tp-invariant.

### 3.2 Sequence parallelism (`dst/ops.py`)

The insight from the paper: an all-reduce *is* a reduce-scatter followed by an
all-gather, so swapping `f`/`f̄` for their sequence-sharded conjugates moves
the same bytes while making all per-layer activation memory divide by t:

| | forward | backward |
|---|---|---|
| `g` (gather_along_seq) | all-gather over seq | reduce-scatter over seq |
| `ḡ` (reduce_scatter_along_seq) | reduce-scatter over seq | all-gather over seq |

Column-parallel layers enter via `g`, row-parallel exit via `ḡ`, the embedding
output scatters along seq, and inside the TP region the sequence stays full
(attention needs every key). Forward hooks in the suite verify every LayerNorm
really sees `[B, T/t, C]`.

**The bug the suite caught:** first run, gradients wrong by 7e-2 on
`attn.proj.bias`. Replicated params that consume sequence-sharded activations
(LayerNorm weights/biases, row-parallel biases added after the reduce-scatter)
only see their rank's sequence chunk in backward — their gradients come out
*partial*. This is the one piece of bookkeeping the conjugate pair doesn't
give for free; Megatron all-reduces exactly these grads across the TP group,
and now so does `finalize_grads()`. Embedding grads are unaffected (the
scatter's backward all-gathers, restoring full gradients) — also asserted.

gloo has no native reduce-scatter, so the CPU path emulates it with
all-reduce + slice; NCCL uses the real `reduce_scatter_tensor`, exercised for
the first time in the GPU phase.

### 3.3 1F1B pipeline parallelism (`dst/pipeline.py`, `dst/parallel.py`)

Topology: `rank = pp·(tp·dp) + dp·tp + tp_rank` — TP innermost so its
bandwidth-hungry traffic stays on consecutive (node-local) ranks; pipeline
neighbors stride by `tp·dp` and carry tiny p2p traffic, matching where the
slow links are on real clusters.

Autograd doesn't span processes, so each stage stashes (input, output) pairs
for in-flight microbatches; backward receives `∂loss/∂output` from downstream,
runs `torch.autograd.backward` over the local segment, and ships
`∂loss/∂input` upstream. Per-microbatch losses scale by 1/m so accumulated
gradients equal the full-batch gradient — verified to ~1e-8.

The schedule is 1F1B: warmup of `p−1−stage` forwards, then strict
one-forward-one-backward, capping in-flight activations at `p−stage`
regardless of m. Each step's sends and receives post as a single
`batch_isend_irecv`, which is what makes the steady state deadlock-free — a
blocking send can never wait on a recv that hasn't been posted. Activation
shapes are static per config, so there's no shape handshake.

### 3.4 Interleaved schedule (`dst/pipeline.py`)

Each rank holds v non-contiguous chunks; virtual stage `k = c·p + r` lives on
rank `k mod p`, so a microbatch traverses the ranks v times and p2p becomes a
ring, with rank p−1 wrapping forward activations back to rank 0 at each chunk
boundary. Step order follows Megatron: `m·v` steps per direction, warmup
`2(p−1−r) + (v−1)p`, chunk `(step mod pv) ÷ p`, m divisible by p.

Two design choices worth recording:

- **Channels.** The ring wrap puts multiple identically-shaped message streams
  on one rank pair (forward-wrap activations *and* backward grads), so
  untagged FIFO matching could silently pair a recv with the wrong stream.
  NCCL has no p2p tags; the portable equivalent chosen was a separate process
  group per (direction, chunk).
- **Send discipline.** Fire-and-forget isends with blocking recvs, so causal
  consistency of the step order suffices for deadlock-freedom — no per-step
  flag machinery. The suite asserts peak in-flight pairs stay within warmup
  depth, the property separating real 1F1B from naive
  all-forward-then-all-backward.

The second choice turned out to be the source of the project's one open
limitation (section 6.4) — correct on gloo, deadlock-prone on NCCL when
composed with tp>1.

### 3.5 Selective recompute (`dst/recompute.py`)

Core attention was rewritten by hand (QKᵀ, causal mask, softmax, ·V),
deliberately materializing the `s×s` attention matrix the way
pre-FlashAttention kernels do — those `s²·b·a` tensors are what selective
recompute exists to discard, so the framework has to actually allocate them
(SDPA never would, and there'd be nothing to measure). `recompute()` is a
~30-line custom autograd Function rather than `torch.utils.checkpoint`:
forward under no_grad saving only q/k/v (alive in the surrounding graph
anyway), backward re-runs the region and backpropagates through the fresh
subgraph.

The test measures reality rather than asserting theory: with
`saved_tensors_hooks` counting every byte autograd stashes, enabling recompute
dropped 4,210,688 bytes against an analytically predicted 4,194,304
(`2·s²·b·a` per layer), with bitwise-identical gradients and the region
entered exactly once per layer per direction.

### 3.6 Data parallelism (`dst/dp.py`) and the full composition

DP itself is the simple axis: each replica takes its batch slice; after
backward, gradients all-reduce and average across the DP group. The reference
implementation is a per-parameter blocking all-reduce — correct, and later the
measured motivation for bucketing (3.8).

The flagship test at this point: **dp2 × pp2 × tp2, interleaved, with
sequence parallelism, on 8 processes** — DP-averaged losses, every gradient,
and a 5-step Adam trajectory matching the full-batch single-process reference
to fp32 noise.

### 3.7 bf16 + masters, data, the trainer

`MasterWeightOptimizer` (`dst/precision.py`): bf16 params/activations/grads,
fp32 master weights and Adam state; grads upcast into the masters, Adam steps
in fp32, masters rounded down once per step. Why: a weight update of `lr·grad`
~1e-4 of the weight underflows bf16's 8 mantissa bits and training stalls.
The precision suite demonstrates the failure mode live instead of citing it —
the same model with plain-bf16 Adam lands measurably behind (1.71 vs 1.57
final loss on the overfit test) while bf16+masters tracks the fp32 trajectory
to a 0.006 gap, with params bitwise equal to `master.to(bf16)`.

Data (`dst/data.py`): memmapped uint16 token shards, nanoGPT convention.
Batches draw at offsets seeded by `(seed, step, dp_rank)`, which does two jobs
with zero communication: every rank inside one DP replica computes the
identical batch (first stage needs tokens, last stage needs targets — no
broadcast), and different DP replicas draw independent data.

`train.py` composes every axis behind flags. First end-to-end proof on the
laptop: an 8-process dp2×pp2×tp2 interleaved+SP+recompute+bf16 run on a
byte-level shard built from the repo's own source code, loss 5.57 → 3.51.

### 3.8 The perf pass (built after first GPU measurements)

The first GPU session (section 5.2) produced two motivating numbers: dp=2
scaled at 1.64× (the unbucketed per-param all-reduce), and tp=2 was *slower
than one GPU* on PCIe. Three optimizations, all CPU-verified before returning
to the GPUs:

- **GradReducer** (`dst/dp.py`) — DDP's core rebuilt on raw collectives:
  ~25MB gradient buckets in reverse parameter order (the order backward
  produces them), each flattened and all-reduced asynchronously the moment its
  last gradient lands via `post_accumulate_grad_hook`, overlapping
  communication with the rest of backward. Pipeline schedules accumulate over
  microbatches so they use a non-overlapped variant (buckets still pipeline
  against each other). DP-then-TP-finalize ordering is safe because the two
  reductions are sums over disjoint groups and commute.
- **Fused qkv** (`dst/model.py`) — one column-parallel matmul of width 3h with
  head-major row layout `[head0: q|k|v, head1: …]`, so the TP column split
  still hands each rank whole heads. One entry collective (`f` or `g`) per
  attention instead of three.
- **Vocab-parallel cross-entropy** (`dst/loss.py`) — with a column-parallel
  head, gathering logits materializes `[b, s, 50304]` per rank, the largest
  activation in the model. Instead: local max + all-reduce(MAX), local Σexp +
  all-reduce(SUM), target logit (each id lives on exactly one rank) +
  all-reduce(SUM); loss = logΣexp − target logit. Three `[N]`-scalar
  all-reduces replace the gather; backward is local softmax-minus-onehot on
  this rank's columns; padded vocab columns mask to −inf and get zero grad.

## 4. Infrastructure

Development machine: a MacBook (Apple Silicon, no CUDA). GPU work: RunPod,
driven end-to-end by `runpodctl` from the same session — pod creation, SSH
key registration, SSH-exec of every benchmark, teardown. Three boxes over the
project:

| box | cloud | rate | used for |
|---|---|---|---|
| 2× RTX 4090 (EU-RO-1, secure, PCIe) | RunPod | $1.48/hr | NCCL validation, FineWeb training, perf A/B, Megatron dp2/tp2 |
| 8× "MIG 1g.24gb" slices | RunPod | $4.72/hr | ~8 minutes; NCCL doesn't work across MIG slices — terminated |
| 4× RTX 3090 (community, public IP) | RunPod | $0.88/hr | pp4 bubble sweep, interleaved-vs-1F1B, composition, Megatron pp4 |

Operational details that cost time and are worth writing down:

- torchrun's rendezvous hangs on macOS DNS; the FileStore launcher fixed it.
- RunPod's external SSH port changes on every pod stop/start.
- Community pods without `--public-ip` may never publish an SSH port mapping.
- 8× MIG slices advertise as 8 GPUs but NCCL cannot communicate across them.
- `pkill -f <pattern>` over SSH kills the SSH session's own shell when the
  pattern appears anywhere in the remote command string — which it does, twice,
  when the command later launches the very process you're trying to kill. Cost
  two mystery failures before diagnosis.

## 5. Results

All throughput numbers are steady-state (measured over steps 15-29, warmup
excluded), bf16 + fp32 masters, selective recompute on.

### 5.1 NCCL validation

All five correctness suites passed under torchrun + NCCL **unchanged from the
gloo versions, on the first run** — including the native
`reduce_scatter_tensor` path gloo could only emulate, bf16 collectives, and
GPU p2p for both pipeline schedules. Gradient errors at the same ~1e-8 level
as CPU.

### 5.2 End-to-end training

GPT-2 124M on a 30M-token GPT-2-BPE FineWeb shard (tokenized on the pod, ~7
minutes), dp=2 on the 2×4090 box: **loss 10.99 → 6.24 over 600 steps**,
sustained 69.7k tok/s pre-perf-pass (`docs/gpu-run-2x4090-fineweb.log`).

### 5.3 The perf pass, measured

2× RTX 4090, GPT-2 124M, micro-batch 8:

| config | before | after | what did it |
|---|---|---|---|
| 1 GPU | 41.6k | 41.6k | (control — fused qkv is comms-bound, not FLOPs) |
| dp=2 | 68.1k (1.64×) | **75.4k (1.81×)** | bucketing + overlap |
| tp=2 | 23.7k | **41.3k** (+75%) | fused qkv + vocab-parallel CE, with SP |

The tp=2 story is the papers' NVLink claim, measured: on PCIe, TP's per-layer
all-reduces made it *lose* to a single GPU; eliminating two-thirds of the
entry collectives and the entire logit gather recovered parity. NVLink is what
would turn parity into speedup.

### 5.4 Benchmarks vs Megatron-LM

Method: Megatron-LM cloned onto the same boxes, run with
`--transformer-impl local`, `--no-masked-softmax-fusion`,
`--no-persist-layer-norm`, `--no-gradient-accumulation-fusion` — its
TransformerEngine and fused CUDA kernels disabled — so both frameworks sit on
identical stock PyTorch kernels and the comparison isolates the parallelism
plumbing, which is the part this project rebuilt. Same model, same
micro-batch, same recompute setting (`--recompute-activations` is Megatron's
selective recompute), mock/synthetic data on both sides.

| config | box | dstraining | Megatron-LM | delta |
|---|---|---|---|---|
| dp=2 | 2×4090 | **75.4k** | 64.9k (with `--overlap-grad-reduce`) | **+16%** |
| tp=2 | 2×4090 | 41.3k (with SP+vp) | 43.6k (its torch path can't do SP) | −5% |
| pp=4, m=16 | 4×3090 | **51.6k** | 37.7k | **+37%** |

### 5.5 The pipeline bubble, measured

4× RTX 3090, 16-layer GPT-2ish model, batch 32:

| m | pp=4 1F1B | pp=4 interleaved v=2 |
|---|---|---|
| 4 | 42.8k | **46.0k** |
| 8 | 49.1k | 49.1k |
| 16 | **51.6k** | 49.3k |
| 32 | 50.7k | 45.6k |

Exactly the textbook shape. 1F1B climbs as `(p−1)/m` dies off and plateaus.
Interleaving wins at small m — +7.5% at m=4, where halving the bubble matters
most — and loses at large m, where the bubble is already small and v× more
p2p messages dominate. Single-GPU baseline for the same model: 21.9k tok/s
(measured at batch 8; batch 32 OOMs one 24GB card, which is itself the
argument for pipelining). Composed tp2 × pp2 + SP + vocab-parallel CE:
24.8k (m=4) → 26.5k (m=16).

## 6. Bugs found and lessons paid for

### 6.1 SP's partial gradients (caught by the suite, day 1)

Described in 3.2. The general lesson: conjugate-operator correctness covers
everything *except* replicated parameters that consume sharded activations.
Every framework has this all-reduce; the suite makes its absence a 7e-2 error
instead of silent training degradation.

### 6.2 The NCCL lazy-init hang (found on GPUs, fixed)

First composed interleaved run on NCCL hung. Diagnosis: NCCL creates a process
group's communicator *collectively on first use*; the interleaved channels are
world-spanning groups whose first op is pair-wise p2p, so the pair waits
forever for ranks that never join the lazy init. gloo has no such step, which
is why every CPU run passed. Fix: warm each channel with a no-op all-reduce at
construction, when every rank is provably present.

### 6.3 The multi-communicator deadlock (found on GPUs, documented)

The warm-up fix was necessary but not sufficient: **interleaved × tp still
deadlocks under NCCL** (isolated cleanly — 1F1B+tp works, interleaved+tp1
works, interleaved+tp2 hangs with or without SP/vp, and env-level mitigations
`CUDA_DEVICE_MAX_CONNECTIONS=1` / `NCCL_P2P_DISABLE=1` don't help). Diagnosis:
NCCL requires ops across multiple communicators to be issued in a globally
consistent order; with tp>1 the tp-collective and channel-p2p orders diverge
across ranks, and unmatched p2p spin-kernels can starve the collectives. The
proper fix is Megatron's approach — per-step combined batched p2p with the
full warmup/steady flag machinery instead of fire-and-forget sends. The combo
is fully correctness-verified on gloo; on NCCL, use 1F1B when composing
pipeline with TP. Documented in the README as a known limitation rather than
papered over.

### 6.4 Smaller ones

- Slicing a Parameter and reading `.grad` off the slice returns None — the
  gradient tensor must be sliced, not the parameter (first draft of the
  gradient checks).
- Attention under SP must take its sequence length from the projection output,
  not the (sequence-sharded) input.
- Vocab padding must be a fixed multiple (128), not a function of tp, or init
  draws stop being tp-invariant.
- Megatron-LM without TransformerEngine requires four separate "no fusion"
  flags, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `python3-dev` for its dataset
  helpers, and a manual rename of the built extension (its Makefile drops the
  Python extension suffix when `python3-config` appeared mid-build).

## 7. Cost accounting

| item | amount |
|---|---|
| 2×4090 sessions (validation, training, perf, Megatron dp/tp) | ~$1.00 |
| 8× MIG experiment (terminated after NCCL probe failed) | ~$0.55 |
| 4×3090 session (bubble sweep, composition, Megatron pp4) | ~$0.55 |
| **Total GPU spend, entire project** | **$2.12** |

The CPU-first method is why: by the time hardware was rented, the only
failures left were the ones only hardware could reveal — and there were
exactly two (6.2, 6.3), both NCCL-semantics issues rather than math issues.

## 8. What's deliberately not done

- **Interleaved × tp on NCCL** — diagnosed, documented, fix known (6.3).
- **TP over NVLink** — needs an A100/H100 pair for about an hour; the PCIe
  numbers tell the qualitative story, NVLink would put a speedup number on it.
- **p=8 pipelines** — no 8×-NCCL-capable box in stock on the day.
- **Dropout > 0** — needs the shared-vs-per-rank RNG state discipline in
  replicated vs parallel regions; under SP the dropout regions are
  sequence-sharded (disjoint), which makes per-rank RNG correct there, but the
  plain-TP path's shared-state requirement is unimplemented.
- **Weight tying** (head holds a vocab shard, embedding is replicated) and
  convergence-grade training runs (loss falling proves correctness;
  convergence was never the point).

## 9. Timeline

Roughly two working days end to end:

| phase | outcome |
|---|---|
| Day 1 | TP + suite · SP (partial-grad bug found and fixed) · 1F1B · interleaved · selective recompute · DP + 8-process full composition · bf16 masters · data + trainer — all on the laptop |
| Day 2 AM | CUDA plumbing · RunPod setup · NCCL validation (5/5 first try) · FineWeb training run · baseline measurements |
| Day 2 PM | perf pass (bucketing/fused qkv/vp CE) built and CPU-verified · A/B measurements · Megatron dp2/tp2 · MIG detour · 4×3090 bubble sweep · NCCL deadlock isolation + channel-warmup fix · Megatron pp4 · teardown, README, this document |
