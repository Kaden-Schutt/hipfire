#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Extract per-layer down_proj column norms (`‖W_down[:,j]‖`) for H-Neurons CETT.

Tooling (not inference hot path): reads the source fp16 `down_proj.weight`
tensors from a HuggingFace safetensors checkpoint and writes a compact binary
the daemon's `cett_load_colnorms` op consumes:

    [u32 n_layers][u32 intermediate][f32 x n_layers*intermediate]  (little-endian)

down_proj.weight is [hidden, intermediate] (torch Linear: out x in); the paper's
`weight_norms = torch.norm(weight, dim=0)` reduces over `hidden` -> one norm per
input neuron j, i.e. `‖W_down[:,j]‖`. Matches extract_activations.py.
"""
import json
import os
import struct
import sys

import numpy as np
import torch
from safetensors import safe_open


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <model_dir> <out.bin>", file=sys.stderr)
        return 2
    model_dir, out_path = sys.argv[1], sys.argv[2]
    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
    wmap = idx["weight_map"]
    down = {
        k: v for k, v in wmap.items() if k.endswith("mlp.down_proj.weight")
    }
    # Order by layer index parsed from `model.layers.{i}.mlp.down_proj.weight`.
    def layer_of(name: str) -> int:
        return int(name.split("model.layers.")[1].split(".")[0])

    names = sorted(down, key=layer_of)
    n_layers = len(names)
    if n_layers == 0:
        print("no down_proj tensors found", file=sys.stderr)
        return 1

    # Group tensor loads by shard to open each file once.
    by_shard: dict[str, list[str]] = {}
    for n in names:
        by_shard.setdefault(down[n], []).append(n)

    # framework="pt" handles bf16 checkpoints numpy can't parse; norm in fp32.
    rows: dict[str, np.ndarray] = {}
    for shard, tensor_names in by_shard.items():
        with safe_open(os.path.join(model_dir, shard), framework="pt") as f:
            for n in tensor_names:
                w = f.get_tensor(n).to(torch.float32)  # [hidden, intermediate]
                rows[n] = torch.linalg.norm(w, dim=0).numpy()  # [intermediate]

    intermediate = int(rows[names[0]].shape[0])
    for n in names:
        assert rows[n].shape[0] == intermediate, f"{n}: ragged intermediate"

    with open(out_path, "wb") as out:
        out.write(struct.pack("<II", n_layers, intermediate))
        for n in names:
            out.write(rows[n].astype("<f4").tobytes())

    print(
        f"wrote {out_path}: n_layers={n_layers} intermediate={intermediate} "
        f"({8 + n_layers * intermediate * 4} bytes)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
