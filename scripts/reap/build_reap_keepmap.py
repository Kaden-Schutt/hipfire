#!/usr/bin/env python3
# Build the REAP keep-map sidecar for the keep-map-aware ds4 loader.
# Output:
#   $OUT/keep_by_layer.json   - {kept_per_layer, num_layers, num_hash_layers, original_experts, keep:[[...]]}
#   $OUT/tid2eid_l{0,1,2}.i32 - sero's remapped hash tables, raw LE int32, shape [129280,6]
#   $OUT/MANIFEST.txt         - provenance
import json, struct, os, sys

HUB = "/home/nick/.cache/huggingface/hub"
OUT = "/data/hipfire-models/reap_keepmap_162B_k144"
def snap(m):
    base = os.path.join(HUB, m, "snapshots")
    return os.path.join(base, sorted(os.listdir(base))[0])
SERO = snap("models--0xSero--DeepSeek-V4-Flash-162B")

# ---- dependency-free safetensors reader ----
_hc = {}
def header_for(root, shard):
    key = (root, shard)
    if key not in _hc:
        with open(os.path.join(root, shard), "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        _hc[key] = (hdr, 8 + n)
    return _hc[key]
def read_tensor(root, wm, name):
    if name not in wm:
        return None
    shard = wm[name]
    hdr, base = header_for(root, shard)
    meta = hdr[name]
    s, e = meta["data_offsets"]
    with open(os.path.join(root, shard), "rb") as f:
        f.seek(base + s); data = f.read(e - s)
    return meta["dtype"], meta["shape"], data

wm = json.load(open(os.path.join(SERO, "model.safetensors.index.json")))["weight_map"]
plan = json.load(open(os.path.join(SERO, "reap_plan.json")))
keep = plan["keep_maps"]["keep_by_layer"]
kept = plan["kept_experts_per_layer"]          # 144
orig = plan["original_experts_per_layer"]      # 256
hashL = plan["hash_routed_layers"]             # [0,1,2]
nlay = len(keep)                               # 43

# ---- validate keep map ----
keep_list = []
for L in range(nlay):
    k = keep[str(L)]
    assert len(k) == kept, f"layer {L}: {len(k)} != {kept}"
    assert len(set(k)) == kept, f"layer {L}: duplicates"
    assert min(k) >= 0 and max(k) < orig, f"layer {L}: idx out of range"
    keep_list.append(k)
print(f"keep map OK: {nlay} layers x {kept} kept (of {orig}); hash layers {hashL}")

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "keep_by_layer.json"), "w") as f:
    json.dump({"kept_per_layer": kept, "num_layers": nlay,
               "num_hash_layers": len(hashL), "original_experts": orig,
               "keep": keep_list}, f)

# ---- tid2eid for hash layers, from sero (already remapped, 0..kept-1 space) ----
DT = {"I64": (8, "<q"), "I32": (4, "<i"), "I16": (2, "<h"), "U32": (4, "<I")}
for L in hashL:
    t = read_tensor(SERO, wm, f"layers.{L}.ffn.gate.tid2eid")
    assert t is not None, f"layer {L}: sero missing tid2eid"
    dt, shape, data = t
    assert dt in DT, f"layer {L}: unexpected tid2eid dtype {dt}"
    esz, fmt = DT[dt]
    n = 1
    for s in shape: n *= s
    assert len(data) == n * esz, f"layer {L}: size mismatch"
    vals = struct.unpack("<" + fmt[1] * n, data) if False else [struct.unpack_from(fmt, data, i*esz)[0] for i in range(n)]
    vmin, vmax = min(vals), max(vals)
    assert vmin >= 0 and vmax < kept, f"layer {L}: tid2eid range [{vmin},{vmax}] not in [0,{kept-1}] (NOT slot space!)"
    out = b"".join(struct.pack("<i", v) for v in vals)
    with open(os.path.join(OUT, f"tid2eid_l{L}.i32"), "wb") as f:
        f.write(out)
    print(f"  layer {L}: tid2eid {dt}{shape} range [{vmin},{vmax}] -> {len(out)}B i32  OK (slot space)")

with open(os.path.join(OUT, "MANIFEST.txt"), "w") as f:
    f.write(f"source: {SERO}\nbase: {plan['base_model_id']}\n"
            f"target: {plan['target_label']}\nkept_per_layer: {kept}\n"
            f"num_layers: {nlay}\nhash_layers: {hashL}\n")
print(f"\nsidecar written to {OUT}")
for fn in sorted(os.listdir(OUT)):
    print(f"  {fn}  ({os.path.getsize(os.path.join(OUT, fn))}B)")
