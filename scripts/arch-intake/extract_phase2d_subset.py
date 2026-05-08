#!/usr/bin/env python3
"""Extract Phase 2.D weight subset for the 4-component validation.

Adds to the Phase 2.A subset:
  - layer 1 router (down_proj, rmsnorm_eda, router_mlp.{0,2,4}, balancing_biases)
  - layer 1 expert 0 (linear_fc1, linear_fc2)
  - layer 0 o_proj
"""
import json
import os
from collections import defaultdict
from safetensors.torch import load_file, save_file

MODEL_DIR = "/home/kaden/zaya-work/ZAYA1-8B"
OUT = "/home/kaden/zaya-work/zaya1_phase2d_subset.safetensors"

idx = json.load(open(f"{MODEL_DIR}/model.safetensors.index.json"))
wmap = idx["weight_map"]

needed = [
    # phase 2.A reuse
    "model.layers.0.input_norm.weight",
    "model.layers.0.res_scale.hidden_states_bias",
    "model.layers.0.res_scale.hidden_states_scale",
    "model.layers.1.input_norm.weight",
    "model.layers.1.res_scale.hidden_states_bias",
    "model.layers.1.res_scale.hidden_states_scale",
    "model.layers.1.res_scale.residual_bias",
    "model.layers.1.res_scale.residual_scale",
    "model.final_norm.weight",
    # phase 2.D additions:
    # layer 0 o_proj (post-attention output projection)
    "model.layers.0.self_attn.o_proj.weight",
    # layer 1 router
    "model.layers.1.zaya_block.router.down_proj.weight",
    "model.layers.1.zaya_block.router.down_proj.bias",
    "model.layers.1.zaya_block.router.rmsnorm_eda.weight",
    "model.layers.1.zaya_block.router.router_mlp.0.weight",
    "model.layers.1.zaya_block.router.router_mlp.0.bias",
    "model.layers.1.zaya_block.router.router_mlp.2.weight",
    "model.layers.1.zaya_block.router.router_mlp.2.bias",
    "model.layers.1.zaya_block.router.router_mlp.4.weight",
    "model.layers.1.zaya_block.router.balancing_biases",
    # layer 1 expert 0 (validate one expert; we know the routing assignments
    # from router.out1 in the dump and apply the right expert per token).
    "model.layers.1.zaya_block.experts.local_experts.0.linear_fc1.weight",
    "model.layers.1.zaya_block.experts.local_experts.0.linear_fc2.weight",
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
