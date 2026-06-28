#!/usr/bin/env python3
"""Aggregate quant_kv_matrix results.csv into a markdown report."""
import csv, sys, collections

csv_path = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
rows = list(csv.DictReader(open(csv_path)))

def num(x):
    try:
        return float(x)
    except Exception:
        return None

# Order models by the canonical sweep order.
model_order = ["LFM2.5-350M.bf16.hfq","qwen3.5-0.8b.bf16.hfq","llama-3.2-1b-instruct.bf16.hfq",
               "qwen3.5-2b.bf16.hfq","qwen3.6-35b-a3b.bf16.hfq","qwen3.5-4b.bf16.hfq",
               "qwen3.5-9b.bf16.hfq","qwen3.6-27b.bf16.hfq"]
fmt_order = ["q8f16","hfq4","hfq6","mq3","mq4","mq6","oq4","oq8","oq4+","oq4++","oq8+","oq8++"]
def mkey(m): return model_order.index(m) if m in model_order else 99
def fkey(f): return fmt_order.index(f) if f in fmt_order else 99

out = []
out.append("# Quant x KV speed matrix\n")
n_pass = sum(1 for r in rows if r["status"]=="pass")
out.append(f"Total rows: {len(rows)} | pass: {n_pass} | "
           f"non-pass: {len(rows)-n_pass}\n")
out.append("Metric shown: warm-decode `tok_s` (decode_tok_s for qwen) / prefill_tok_s. "
           "Pruned-after-bench; numbers are from `hipfire eval --battery speed`.\n")

# Main table: rows = model+format, columns = kv modes -> tok_s
by_mf = collections.defaultdict(dict)
kvs_seen = []
for r in rows:
    key = (r["model"], r["format"])
    kv = r["kv"]
    if kv not in kvs_seen: kvs_seen.append(kv)
    dec = num(r["decode_tok_s"]) or num(r["tok_s"])
    by_mf[key][kv] = (dec, r["status"])
kvs = [k for k in ["q8","asym4","asym3","asym2","fp32"] if k in kvs_seen]

out.append("\n## Decode tok/s by model x format x KV\n")
out.append("| model | format | " + " | ".join(kvs) + " |")
out.append("|---|---|" + "|".join("---" for _ in kvs) + "|")
for (m,f) in sorted(by_mf, key=lambda x:(mkey(x[0]), fkey(x[1]))):
    cells=[]
    for kv in kvs:
        v = by_mf[(m,f)].get(kv)
        if v is None: cells.append("·")
        elif v[1]!="pass": cells.append(f"_{v[1]}_")
        else: cells.append(f"{v[0]:.1f}" if v[0] is not None else "?")
    out.append(f"| {m.replace('.bf16.hfq','')} | {f} | " + " | ".join(cells) + " |")

# Prefill table (qwen has prefill_tok_s)
out.append("\n## Prefill tok/s by model x format x KV (where reported)\n")
by_pf = collections.defaultdict(dict)
for r in rows:
    p = num(r["prefill_tok_s"])
    if p is not None: by_pf[(r["model"],r["format"])][r["kv"]] = p
out.append("| model | format | " + " | ".join(kvs) + " |")
out.append("|---|---|" + "|".join("---" for _ in kvs) + "|")
for (m,f) in sorted(by_pf, key=lambda x:(mkey(x[0]), fkey(x[1]))):
    cells=[f"{by_pf[(m,f)].get(kv):.0f}" if by_pf[(m,f)].get(kv) is not None else "·" for kv in kvs]
    out.append(f"| {m.replace('.bf16.hfq','')} | {f} | " + " | ".join(cells) + " |")

# Failures
out.append("\n## Non-pass cells\n")
fails = collections.defaultdict(list)
for r in rows:
    if r["status"]!="pass":
        fails[(r["model"],r["format"],r["status"])].append(r["kv"])
if not fails:
    out.append("None.\n")
else:
    out.append("| model | format | status | kv | reason |")
    out.append("|---|---|---|---|---|")
    seen=set()
    for r in rows:
        if r["status"]=="pass": continue
        k=(r["model"],r["format"],r["status"],r["reason"][:80])
        if k in seen: continue
        seen.add(k)
        out.append(f"| {r['model'].replace('.bf16.hfq','')} | {r['format']} | {r['status']} | {r['kv']} | {r['reason'][:80]} |")

print("\n".join(out))
