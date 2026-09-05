#!/usr/bin/env bash
# Local CPU launcher: spawns N ranks with a FileStore rendezvous.
# Exists because torchrun's TCP rendezvous can hang on macOS DNS lookups;
# on Linux/GPU use torchrun directly, the code paths are identical.
#
#   scripts/launch_local.sh 2 tests/test_tp_correctness.py [args...]
set -u
N=${1:?usage: launch_local.sh NPROC script.py [args...]}
shift
STORE=$(mktemp -d)/store

pids=()
for r in $(seq 0 $((N - 1))); do
  RANK=$r LOCAL_RANK=$r WORLD_SIZE=$N DST_INIT_METHOD="file://$STORE" \
    python3 "$@" &
  pids+=($!)
done

fail=0
for p in "${pids[@]}"; do
  wait "$p" || fail=1
done
exit $fail
