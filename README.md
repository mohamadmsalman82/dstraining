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
- [ ] Sequence parallelism
- [ ] 1F1B pipeline parallelism
- [ ] Interleaved pipeline schedule
- [ ] Selective activation recompute
- [ ] Data parallelism on top; benchmark against Megatron-LM

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

## Layout

```
dst/parallel.py   process groups and TP topology (TPContext)
dst/ops.py        f, f̄, and gather — the only collectives in milestone 1
dst/layers.py     ColumnParallelLinear, RowParallelLinear
dst/model.py      GPT-2 from scratch, TP-aware
tests/            numerical correctness suite
scripts/          launchers
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

TP degree = number of processes. The suite checks, in fp32:
conjugacy of `f`/`f̄` on a toy split, shard-vs-slice init identity, forward
logits vs the reference and across ranks, every gradient shard vs the sliced
reference gradient, and a 10-step Adam trajectory that must track the
reference and fall.

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
- Dropout defaults to 0. Parallel-region dropout needs per-rank RNG state
  (replicated regions need shared state); that lands with sequence
  parallelism, which is where the RNG split becomes load-bearing anyway.
