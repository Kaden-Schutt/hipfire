#!/usr/bin/env python3
# Keep-ALL identity plan: keep every expert (no pruning). If the generic REAP
# machinery is correct, loading any MoE arch with HIPFIRE_REAP_PLAN=<out> must
# reproduce that arch's no-plan baseline NLL to ~10 decimals — isolating
# machinery bugs from REAP pruning cost. Arch-agnostic.
#
# Usage:
#   build_keepall_sidecar.py --num-layers N --num-experts E --out DIR [--arch NAME]
#   # ds4 convenience (also emits hash-layer tid2eid sidecars + legacy
#   # keep_by_layer.json so the HIPFIRE_DEEPSEEK4_REAP_KEEPMAP alias path is
#   # exercised too):
#   build_keepall_sidecar.py --ds4 [--out DIR]
#
# Emits <out>/reap_plan.json (new generic schema). In --ds4 mode also emits
# tid2eid_l{L}.i32 (identity, original) and the legacy keep_by_layer.json.
import argparse, json, os, struct

HUB = "/home/nick/.cache/huggingface/hub"


def snap(m):
    base = os.path.join(HUB, m, "snapshots")
    return os.path.join(base, sorted(os.listdir(base))[0])


_hc = {}


def header_for(root, shard):
    k = (root, shard)
    if k not in _hc:
        with open(os.path.join(root, shard), "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            _hc[k] = (json.loads(f.read(n)), 8 + n)
    return _hc[k]


def read_tensor(root, wm, name):
    shard = wm[name]
    hdr, base = header_for(root, shard)
    meta = hdr[name]
    s, e = meta["data_offsets"]
    with open(os.path.join(root, shard), "rb") as f:
        f.seek(base + s)
        return meta["dtype"], meta["shape"], f.read(e - s)


def write_plan(out, arch, num_layers, num_experts):
    """Write the generic keep-all reap_plan.json (ReapPlan::load schema)."""
    os.makedirs(out, exist_ok=True)
    keep = [list(range(num_experts)) for _ in range(num_layers)]
    plan = {
        "version": 1,
        "original_experts": num_experts,
        "num_layers": num_layers,
        "keep": {"per_layer": keep},
    }
    if arch:
        plan["model_arch"] = arch
    with open(os.path.join(out, "reap_plan.json"), "w") as f:
        json.dump(plan, f)
    print(f"keep-all reap_plan.json ({num_layers}L x {num_experts}E"
          f"{', arch=' + arch if arch else ''}) -> {out}")


def write_ds4_extras(out, orig_snapshot, num_layers, num_experts, hash_layers):
    """ds4-only: identity tid2eid sidecars + legacy keep_by_layer.json."""
    wm = json.load(open(os.path.join(orig_snapshot, "model.safetensors.index.json")))["weight_map"]
    # Legacy alias schema (load_legacy_keepmap / HIPFIRE_DEEPSEEK4_REAP_KEEPMAP).
    keep = [list(range(num_experts)) for _ in range(num_layers)]
    json.dump(
        {"kept_per_layer": num_experts, "num_layers": num_layers,
         "num_hash_layers": len(hash_layers), "original_experts": num_experts, "keep": keep},
        open(os.path.join(out, "keep_by_layer.json"), "w"),
    )
    DT = {"I64": (8, "<q"), "I32": (4, "<i"), "U32": (4, "<I")}
    for L in hash_layers:
        dt, shape, data = read_tensor(orig_snapshot, wm, f"layers.{L}.ffn.gate.tid2eid")
        esz, fmt = DT[dt]
        n = 1
        for s in shape:
            n *= s
        vals = [struct.unpack_from(fmt, data, i * esz)[0] for i in range(n)]
        assert 0 <= min(vals) and max(vals) < num_experts, f"L{L} range {min(vals)}..{max(vals)}"
        with open(os.path.join(out, f"tid2eid_l{L}.i32"), "wb") as f:
            f.write(b"".join(struct.pack("<i", v) for v in vals))
        print(f"  L{L}: orig tid2eid {dt}{shape} range [{min(vals)},{max(vals)}] -> i32 OK")
    print(f"ds4 legacy keep_by_layer.json + tid2eid sidecars -> {out}")


def main():
    ap = argparse.ArgumentParser(description="Build a keep-all identity REAP plan.")
    ap.add_argument("--num-layers", type=int, help="number of MoE layers")
    ap.add_argument("--num-experts", type=int, help="routed experts per layer (original count)")
    ap.add_argument("--arch", default=None, help="optional model_arch tag for the plan")
    ap.add_argument("--out", default=None, help="output dir for the plan")
    ap.add_argument("--ds4", action="store_true",
                    help="DeepSeek-V4-Flash convenience: 43L x 256E + tid2eid + legacy sidecar")
    ap.add_argument("--ds4-model", default="models--deepseek-ai--DeepSeek-V4-Flash",
                    help="HF hub dir name for the ds4 original (for tid2eid)")
    ap.add_argument("--hash-layers", default="0,1,2", help="ds4 hash layer indices (comma-sep)")
    args = ap.parse_args()

    if args.ds4:
        num_layers = args.num_layers or 43
        num_experts = args.num_experts or 256
        out = args.out or "/data/hipfire-models/reap_keepall_256"
        hash_layers = [int(x) for x in args.hash_layers.split(",") if x != ""]
        write_plan(out, "deepseek4", num_layers, num_experts)
        write_ds4_extras(out, snap(args.ds4_model), num_layers, num_experts, hash_layers)
    else:
        if args.num_layers is None or args.num_experts is None or args.out is None:
            ap.error("--num-layers, --num-experts and --out are required (or use --ds4)")
        write_plan(args.out, args.arch, args.num_layers, args.num_experts)


if __name__ == "__main__":
    main()
