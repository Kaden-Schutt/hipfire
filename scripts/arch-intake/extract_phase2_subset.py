#!/usr/bin/env python3
"""Extract a tiny subset of ZAYA1-8B weights needed for Phase 2.A CPU
validation (RMSNorm + res_scale on layer 0 + 1 + final_norm).

Run on hiptrx (model lives there); writes a single safetensors blob
back to ~/zaya-work/zaya1_phase2_subset.safetensors which scp's to
local for the Rust validator to consume.
"""
import json
import os
from collections import defaultdict
from safetensors.torch import load_file, save_file

MODEL_DIR = "/home/kaden/zaya-work/ZAYA1-8B"
OUT = "/home/kaden/zaya-work/zaya1_phase2_subset.safetensors"

idx = json.load(open(f"{MODEL_DIR}/model.safetensors.index.json"))
wmap = idx["weight_map"]

needed = [
    "model.layers.0.input_norm.weight",
    "model.layers.0.res_scale.hidden_states_bias",
    "model.layers.0.res_scale.hidden_states_scale",
    "model.layers.1.input_norm.weight",
    "model.layers.1.res_scale.hidden_states_bias",
    "model.layers.1.res_scale.hidden_states_scale",
    "model.layers.1.res_scale.residual_bias",
    "model.layers.1.res_scale.residual_scale",
    "model.final_norm.weight",
]

by_shard = defaultdict(list)
for n in needed:
    if n in wmap:
        by_shard[wmap[n]].append(n)
    else:
        print(f"MISSING: {n}")

extracted = {}
for shard, names in by_shard.items():
    t = load_file(f"{MODEL_DIR}/{shard}")
    for n in names:
        extracted[n] = t[n]
        print(f"loaded {n} {tuple(t[n].shape)} dtype={t[n].dtype}")

save_file(extracted, OUT)
print()
print(f"Wrote {OUT}")
print(f"Size: {os.path.getsize(OUT)} bytes")
