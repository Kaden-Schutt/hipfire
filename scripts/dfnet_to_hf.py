#!/usr/bin/env python3
# Remap a trained DFlash .dfnet (trainer Net) into an HF DFlash-draft safetensors dir
# that dflash_convert can turn into a .hf4. Drops in_proj_v/out_proj_v (~identity;
# the inference engine omits them). Sources hidden_norm.weight from the original z-lab
# safetensors (frozen during finetune, not saved in .dfnet).
import sys, struct, os, json, shutil
import torch
from safetensors.torch import save_file, safe_open

DFNET = sys.argv[1]          # /workspace/ws-g-ce.dfnet
OUTDIR = sys.argv[2]          # /workspace/ws-g-ce-hf
ZLAB = "/workspace/zlab-dflash"

def read_dfnet(path):
    f = open(path, "rb")
    assert f.read(8)[:5] == b"DFNET", "bad magic"
    ne = struct.unpack("<I", f.read(4))[0]
    cfg = struct.unpack("<9I", f.read(36))  # d,nl,nh,nkv,hd,convk,inter,dtgt,vocab
    t = {}
    for _ in range(ne):
        nl = struct.unpack("<I", f.read(4))[0]
        name = f.read(nl).decode()
        dl = struct.unpack("<I", f.read(4))[0]
        data = torch.frombuffer(bytearray(f.read(dl * 4)), dtype=torch.float32).clone()
        t[name] = data
    return cfg, t

cfg, t = read_dfnet(DFNET)
d, nl, nh, nkv, hd, convk, inter, dtgt, vocab = cfg
qd, kvd = nh * hd, nkv * hd
print(f"dfnet cfg: d={d} layers={nl} nh={nh} nkv={nkv} hd={hd} inter={inter} dtgt={dtgt} vocab={vocab}")

# shape table (HF [out,in])
shp = lambda name: {
    "fc": (d, 5 * dtgt), "norm": (d,),
    "input_layernorm": (d,), "post_attention_layernorm": (d,),
    "q_proj": (qd, d), "k_proj": (kvd, d), "v_proj": (kvd, d), "o_proj": (d, qd),
    "q_norm": (hd,), "k_norm": (hd,),
    "gate_proj": (inter, d), "up_proj": (inter, d), "down_proj": (d, inter),
}[name]

out = {}
def put(hf_name, flat, shape):
    want = 1
    for s in shape:
        want *= s
    assert flat.numel() == want, f"{hf_name}: numel {flat.numel()} != {shape} ({want})"
    out[hf_name] = flat.reshape(shape).to(torch.bfloat16).contiguous()

put("fc.weight", t["fc"], shp("fc"))
put("norm.weight", t["final_norm"], shp("norm"))
for i in range(nl):
    p = f"layers.{i}"
    put(f"{p}.input_layernorm.weight", t[f"layers.{i}.op_norm"], shp("input_layernorm"))
    put(f"{p}.post_attention_layernorm.weight", t[f"layers.{i}.ffn_norm"], shp("post_attention_layernorm"))
    put(f"{p}.self_attn.q_proj.weight", t[f"layers.{i}.wq"], shp("q_proj"))
    put(f"{p}.self_attn.k_proj.weight", t[f"layers.{i}.wk"], shp("k_proj"))
    put(f"{p}.self_attn.v_proj.weight", t[f"layers.{i}.wv"], shp("v_proj"))
    put(f"{p}.self_attn.o_proj.weight", t[f"layers.{i}.wo"], shp("o_proj"))
    put(f"{p}.self_attn.q_norm.weight", t[f"layers.{i}.q_norm"], shp("q_norm"))
    put(f"{p}.self_attn.k_norm.weight", t[f"layers.{i}.k_norm"], shp("k_norm"))
    put(f"{p}.mlp.gate_proj.weight", t[f"layers.{i}.w1"], shp("gate_proj"))
    put(f"{p}.mlp.up_proj.weight", t[f"layers.{i}.w3"], shp("up_proj"))
    put(f"{p}.mlp.down_proj.weight", t[f"layers.{i}.w2"], shp("down_proj"))

# hidden_norm.weight: from original z-lab safetensors (frozen, not in .dfnet)
with safe_open(f"{ZLAB}/model.safetensors", framework="pt") as f:
    hn = f.get_tensor("hidden_norm.weight")
out["hidden_norm.weight"] = hn.to(torch.bfloat16).contiguous()

os.makedirs(OUTDIR, exist_ok=True)
save_file(out, f"{OUTDIR}/model.safetensors")
shutil.copy(f"{ZLAB}/config.json", f"{OUTDIR}/config.json")
if os.path.exists(f"{ZLAB}/dflash.py"):
    shutil.copy(f"{ZLAB}/dflash.py", f"{OUTDIR}/dflash.py")
print(f"wrote {len(out)} tensors -> {OUTDIR}/model.safetensors (dropped in_proj_v/out_proj_v, hidden_norm from z-lab)")
