#!/usr/bin/env python3
"""Compile hipfire profiling JSON + llama.cpp bench output into markdown report."""
import json, glob, sys, os, re
from collections import defaultdict

results_dir = sys.argv[1] if len(sys.argv) > 1 else "bench/results"
timestamp = sys.argv[2] if len(sys.argv) > 2 else ""

# ─── Load hipfire JSON results ───
hipfire_data = defaultdict(list)  # key: (model, n_tok) -> list of run dicts
for f in sorted(glob.glob(f"{results_dir}/hipfire_*_{timestamp}.json")):
    with open(f) as fh:
        d = json.load(fh)
    bn = os.path.basename(f)
    parts = bn.split("_")  # hipfire_0.6b_20tok_run1_...
    model = parts[1]
    n_tok = int(parts[2].replace("tok", ""))
    hipfire_data[(model, n_tok)].append(d)

# ─── Load llama.cpp results ───
llamacpp_data = {}  # key: (model, n_tok) -> {pp_tok_s, tg_tok_s}
for f in sorted(glob.glob(f"{results_dir}/llamacpp_*_{timestamp}.txt")):
    bn = os.path.basename(f)
    parts = bn.split("_")
    model = parts[1]
    n_tok = int(parts[2].replace("tok", ""))
    with open(f) as fh:
        text = fh.read()
    pp_match = re.search(r'pp\d+\s*\|\s*([\d.]+)\s*±', text)
    tg_match = re.search(r'tg\d+\s*\|\s*([\d.]+)\s*±', text)
    llamacpp_data[(model, n_tok)] = {
        "pp_tok_s": float(pp_match.group(1)) if pp_match else 0,
        "tg_tok_s": float(tg_match.group(1)) if tg_match else 0,
    }

# ─── Aggregate hipfire per-op timings ───
def avg_timings(runs):
    """Average token timings across runs, return per-op averages."""
    all_tokens = []
    for run in runs:
        all_tokens.extend(run["token_timings"])
    if not all_tokens:
        return {}

    n = len(all_tokens)
    n_layers = len(all_tokens[0]["layers"])

    # Average per-token overhead
    avg = {}
    for key in ["embedding_us", "output_norm_us", "output_proj_us", "sampling_us", "total_us"]:
        avg[key] = sum(t[key] for t in all_tokens) / n

    # Average per-layer
    avg_layers = []
    for li in range(n_layers):
        layer_avg = {}
        for key in all_tokens[0]["layers"][0].keys():
            layer_avg[key] = sum(t["layers"][li][key] for t in all_tokens) / n
        avg_layers.append(layer_avg)
    avg["layers"] = avg_layers

    # Category totals (summed across layers, averaged across tokens)
    cats = defaultdict(float)
    for la in avg_layers:
        for k, v in la.items():
            if k != "total_us":
                cats[k] += v
    avg["category_totals"] = dict(cats)
    avg["layers_total_us"] = sum(la["total_us"] for la in avg_layers)

    # System snapshots
    all_snaps = []
    for run in runs:
        all_snaps.extend(run.get("system_snapshots", []))
    if all_snaps:
        avg["system"] = {
            "gpu_temp_c": sum(s["gpu_temp_c"] for s in all_snaps) / len(all_snaps),
            "gpu_power_w": sum(s["gpu_power_w"] for s in all_snaps) / len(all_snaps),
            "gpu_sclk_mhz": max(s["gpu_sclk_mhz"] for s in all_snaps),
            "gpu_mclk_mhz": max(s["gpu_mclk_mhz"] for s in all_snaps),
            "gpu_util_pct": sum(s["gpu_util_pct"] for s in all_snaps) / len(all_snaps),
            "gpu_vram_used_mb": sum(s["gpu_vram_used_mb"] for s in all_snaps) / len(all_snaps),
            "gpu_vram_total_mb": all_snaps[0]["gpu_vram_total_mb"],
            "cpu_temp_c": sum(s["cpu_temp_c"] for s in all_snaps) / len(all_snaps),
            "ram_used_mb": sum(s["ram_used_mb"] for s in all_snaps) / len(all_snaps),
            "ram_total_mb": all_snaps[0]["ram_total_mb"],
        }
    return avg

# ─── Generate markdown ───
lines = []
lines.append("# hipfire Profiling Report")
lines.append("")
lines.append(f"Generated: {timestamp}")
lines.append(f"GPU: AMD RX 5700 XT (gfx1010, RDNA1, 8GB GDDR6)")
lines.append("")

for (model, n_tok), runs in sorted(hipfire_data.items()):
    avg = avg_timings(runs)
    if not avg:
        continue

    lines.append(f"## Qwen3-{model} — {n_tok} generation tokens")
    lines.append("")

    # System stats
    if "system" in avg:
        s = avg["system"]
        lines.append(f"**System during profiling:** GPU {s['gpu_temp_c']:.0f}°C, {s['gpu_power_w']:.0f}W, "
                     f"SCLK {s['gpu_sclk_mhz']}MHz, MCLK {s['gpu_mclk_mhz']}MHz, "
                     f"GPU util {s['gpu_util_pct']:.0f}%, VRAM {s['gpu_vram_used_mb']:.0f}/{s['gpu_vram_total_mb']:.0f}MB, "
                     f"CPU {s['cpu_temp_c']:.0f}°C, RAM {s['ram_used_mb']:.0f}/{s['ram_total_mb']:.0f}MB")
        lines.append("")

    # Top-level timing
    tok_s = 1_000_000.0 / avg["total_us"] if avg["total_us"] > 0 else 0
    lines.append(f"**Per-token average:** {avg['total_us']:.0f}µs ({tok_s:.1f} tok/s)")
    lines.append("")

    # llama.cpp comparison
    lc = llamacpp_data.get((model, n_tok))
    if lc:
        lines.append(f"**llama.cpp ROCm:** {lc['tg_tok_s']:.1f} tok/s generation, {lc['pp_tok_s']:.1f} tok/s prefill")
        lines.append(f"**hipfire vs llama.cpp:** {tok_s / lc['tg_tok_s']:.2f}x generation" if lc['tg_tok_s'] > 0 else "")
        lines.append("")

    # Overhead breakdown
    lines.append("### Non-layer overhead")
    lines.append("")
    lines.append("| Component | Time (µs) | % of total |")
    lines.append("|-----------|-----------|-----------|")
    for key, label in [("embedding_us", "Embedding lookup"), ("output_norm_us", "Output RMSNorm"),
                       ("output_proj_us", "Output projection"), ("sampling_us", "Sampling")]:
        pct = avg[key] / avg["total_us"] * 100 if avg["total_us"] > 0 else 0
        lines.append(f"| {label} | {avg[key]:.1f} | {pct:.1f}% |")
    layer_pct = avg["layers_total_us"] / avg["total_us"] * 100 if avg["total_us"] > 0 else 0
    lines.append(f"| **All layers** | **{avg['layers_total_us']:.0f}** | **{layer_pct:.1f}%** |")
    lines.append(f"| **Total** | **{avg['total_us']:.0f}** | **100%** |")
    lines.append("")

    # Per-category breakdown (summed across all layers)
    cats = avg["category_totals"]
    lines.append("### Per-operation breakdown (summed across all layers)")
    lines.append("")
    lines.append("| Operation | Total µs | % of layer time | % of token |")
    lines.append("|-----------|----------|-----------------|-----------|")
    sorted_cats = sorted(cats.items(), key=lambda x: -x[1])
    for k, v in sorted_cats:
        lpct = v / avg["layers_total_us"] * 100 if avg["layers_total_us"] > 0 else 0
        tpct = v / avg["total_us"] * 100 if avg["total_us"] > 0 else 0
        label = k.replace("_us", "").replace("_", " ")
        lines.append(f"| {label} | {v:.0f} | {lpct:.1f}% | {tpct:.1f}% |")
    lines.append("")

    # Sample layer detail (layer 0 and last layer)
    n_layers = len(avg["layers"])
    for li in [0, n_layers - 1]:
        la = avg["layers"][li]
        lines.append(f"### Layer {li} detail")
        lines.append("")
        lines.append("| Op | µs |")
        lines.append("|----|---:|")
        for k in ["attn_norm_us", "q_proj_us", "k_proj_us", "v_proj_us", "qk_norm_us",
                   "rope_us", "kv_cache_us", "attention_us", "o_proj_us", "attn_residual_us",
                   "ffn_norm_us", "gate_proj_us", "up_proj_us", "silu_mul_us", "down_proj_us",
                   "ffn_residual_us", "total_us"]:
            label = k.replace("_us", "").replace("_", " ")
            lines.append(f"| {label} | {la[k]:.1f} |")
        lines.append("")

lines.append("---")
lines.append("*Generated by bench/compile_results.py*")

md = "\n".join(lines)
out_path = f"{results_dir}/profile_report_{timestamp}.md"
with open(out_path, "w") as f:
    f.write(md)
print(f"Report written to: {out_path}")
print(md)
