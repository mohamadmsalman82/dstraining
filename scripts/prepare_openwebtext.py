"""Tokenize OpenWebText into flat uint16 shards (nanoGPT convention).

    pip install datasets tiktoken
    python3 scripts/prepare_openwebtext.py [--out data] [--num-proc 8]

Writes data/train.bin and data/val.bin: GPT-2 BPE token ids as raw
uint16, ready for dst.data.TokenShard. Expect ~17GB for train.
"""

import argparse
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data")
    parser.add_argument("--num-proc", type=int, default=8)
    args = parser.parse_args()

    import tiktoken
    from datasets import load_dataset

    enc = tiktoken.get_encoding("gpt2")
    os.makedirs(args.out, exist_ok=True)

    dataset = load_dataset("openwebtext", num_proc=args.num_proc)
    split = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)

    def tokenize(example):
        ids = enc.encode_ordinary(example["text"])
        ids.append(enc.eot_token)
        return {"ids": ids, "len": len(ids)}

    tokenized = split.map(
        tokenize, remove_columns=["text"], num_proc=args.num_proc, desc="tokenizing"
    )

    for name, dset in [("train", tokenized["train"]), ("val", tokenized["test"])]:
        total = int(np.sum(dset["len"], dtype=np.uint64))
        path = os.path.join(args.out, f"{name}.bin")
        arr = np.memmap(path, dtype=np.uint16, mode="w+", shape=(total,))
        pos = 0
        for batch in dset.iter(batch_size=1024):
            for ids in batch["ids"]:
                arr[pos : pos + len(ids)] = ids
                pos += len(ids)
        arr.flush()
        print(f"{path}: {total:,} tokens")


if __name__ == "__main__":
    main()
