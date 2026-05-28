#!/usr/bin/env python3
"""Build an HTML/JSON KV-quant identicality dashboard from quantization-gate output."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
from collections import defaultdict
from pathlib import Path


def read_json(path: Path):
    with path.open() as f:
        return json.load(f)


def finite_float(value, default=math.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else math.nan


def mode_bytes_per_head(mode: str, head_dim: int) -> int | None:
    mode = mode.lower()
    q8_half = (head_dim // 32) * 34
    asym4_k = 4 + head_dim // 2
    if mode in ("fp32", "f32"):
        return head_dim * 2 * 4
    if mode == "q8":
        return q8_half * 2
    if mode in ("asym4", "turbo4", "fwht4"):
        return asym4_k + q8_half
    if mode in ("asym3", "turbo3", "turbo", "fwht3"):
        return (4 + (head_dim * 3) // 8) + q8_half
    if mode in ("asym2", "turbo2", "fwht2"):
        return (4 + head_dim // 4) + q8_half
    if mode in ("asym4_tqv4", "tqv4"):
        return asym4_k + (4 + head_dim // 2)
    if mode in ("asym4_tqv3", "tqv3"):
        return asym4_k + (4 + (head_dim * 3) // 8)
    if mode in ("asym4_tqv2", "tqv2"):
        return asym4_k + (4 + head_dim // 4)
    if mode in ("asym4_tqv1", "tqv1", "tq1"):
        return asym4_k + (4 + head_dim // 4)
    return None


def identicality(row: dict) -> float:
    steps = int(row.get("steps_compared") or 0)
    top1 = finite_float(row.get("top1_agreement"), 0.0)
    top5 = finite_float(row.get("mean_top5_overlap"), 0.0) / 5.0
    first_div = row.get("first_divergence")
    if first_div is None:
        survival = 1.0
    else:
        try:
            survival = max(0.0, min(1.0, int(first_div) / steps)) if steps > 0 else 0.0
        except Exception:
            survival = 0.0
    return max(0.0, min(1.0, 0.55 * top1 + 0.30 * top5 + 0.15 * survival))


def parse_perf(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            mode = row.get("mode", "")
            if not mode:
                continue
            for key in ("prefill_tok_s", "decode_tok_s"):
                value = finite_float(row.get(key))
                if math.isfinite(value):
                    grouped[mode][key].append(value)
    return {
        mode: {
            "prefill_tok_s": mean(vals.get("prefill_tok_s", [])),
            "decode_tok_s": mean(vals.get("decode_tok_s", [])),
        }
        for mode, vals in grouped.items()
    }


def load_prompt_text(path: Path, max_chars: int) -> str:
    if not path.exists():
        return ""
    text = path.read_text(errors="replace").strip()
    if len(text) > max_chars:
        return text[:max_chars].rstrip() + "..."
    return text


def parse_hfq_metadata(path: Path) -> dict:
    try:
        with path.open("rb") as f:
            header = f.read(32)
            if len(header) < 32 or header[:4] != b"HFQM":
                return {"format": "unknown"}
            arch_id = int.from_bytes(header[8:12], "little")
            n_tensors = int.from_bytes(header[12:16], "little")
            metadata_offset = int.from_bytes(header[16:24], "little")
            data_offset = int.from_bytes(header[24:32], "little")
            f.seek(metadata_offset)
            meta_bytes = f.read(data_offset - metadata_offset)
    except Exception as exc:
        return {"error": str(exc)}
    depth = 0
    in_string = False
    escape = False
    json_end = 0
    for i, b in enumerate(meta_bytes):
        if escape:
            escape = False
            continue
        if b == 0x5C and in_string:
            escape = True
            continue
        if b == 0x22:
            in_string = not in_string
            continue
        if not in_string:
            if b == 0x7B:
                depth += 1
            elif b == 0x7D:
                depth -= 1
                if depth == 0:
                    json_end = i + 1
                    break
    out = {
        "format": "HFQ",
        "arch_id": arch_id,
        "tensor_count": n_tensors,
    }
    if not json_end:
        return out
    try:
        meta = json.loads(meta_bytes[:json_end].decode("utf-8", "replace"))
    except Exception as exc:
        out["metadata_error"] = str(exc)
        return out
    config = meta.get("config") or {}
    tc = config.get("text_config") or config
    layer_types = tc.get("layer_types") or []
    if isinstance(layer_types, list):
        layer_counts = {name: layer_types.count(name) for name in sorted(set(layer_types))}
    else:
        layer_counts = {}
    fields = [
        "model_type",
        "architectures",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
        "intermediate_size",
        "num_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "rope_theta",
        "rms_norm_eps",
    ]
    out["config"] = {k: tc[k] for k in fields if k in tc}
    if layer_counts:
        out["config"]["layer_counts"] = layer_counts
    if "quantization" in meta:
        out["quantization"] = meta["quantization"]
    if "source" in meta:
        out["source"] = meta["source"]
    return out


def model_details(model: Path | None, model_name: str | None) -> dict:
    if not model:
        return {"name": model_name or ""}
    out = {
        "name": model_name or model.name,
        "path": str(model),
    }
    try:
        st = model.stat()
        out["size_bytes"] = st.st_size
        out["size_gb"] = st.st_size / 1_000_000_000
        h = hashlib.md5()
        with model.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        out["md5"] = h.hexdigest()
    except Exception as exc:
        out["stat_error"] = str(exc)
    out.update(parse_hfq_metadata(model))
    return out


def collect(gate_dir: Path, baseline: str, head_dim: int, excerpt_chars: int, model: Path | None, model_name: str | None) -> dict:
    perf = parse_perf(gate_dir / "perf" / "perf.csv")
    prompt_rows: list[dict] = []
    by_mode: dict[str, list[dict]] = defaultdict(list)

    logits_dir = gate_dir / "logits"
    for prompt_dir in sorted(p for p in logits_dir.glob("*") if p.is_dir()):
        compare = prompt_dir / "compare.json"
        if not compare.exists():
            continue
        for row in read_json(compare):
            item = dict(row)
            item["prompt"] = prompt_dir.name
            item["identicality"] = identicality(item)
            prompt_rows.append(item)
            by_mode[item["mode"]].append(item)

    all_modes = {baseline, *by_mode.keys(), *perf.keys()}
    prompt_texts: dict[str, dict[str, str]] = defaultdict(dict)
    prompt_inputs: dict[str, dict[str, str]] = defaultdict(dict)
    prompt_dir = gate_dir / "prompts"
    for path in sorted(prompt_dir.glob("*.prompt.txt")):
        prompt = path.name[: -len(".prompt.txt")]
        prompt_inputs[prompt]["prompt"] = load_prompt_text(path, max(excerpt_chars * 3, 2400))
    for path in sorted(prompt_dir.glob("*.system.txt")):
        prompt = path.name[: -len(".system.txt")]
        prompt_inputs[prompt]["system"] = load_prompt_text(path, max(excerpt_chars * 3, 2400))
    for path in sorted(prompt_dir.glob("*.txt")):
        stem = path.name[:-4]
        if "." not in stem:
            continue
        if stem.endswith(".prompt") or stem.endswith(".system"):
            continue
        prompt, mode = stem.rsplit(".", 1)
        prompt_texts[prompt][mode] = load_prompt_text(path, excerpt_chars)
        all_modes.add(mode)

    summary = []
    fp32_bytes = mode_bytes_per_head("fp32", head_dim) or 0
    for mode in sorted(all_modes):
        rows = by_mode.get(mode, [])
        bytes_head = mode_bytes_per_head(mode, head_dim)
        perf_row = perf.get(mode, {})
        if mode == baseline:
            identical = top1 = top5 = survival = 1.0
        else:
            identical = mean([finite_float(r["identicality"]) for r in rows])
            top1 = mean([finite_float(r.get("top1_agreement")) for r in rows])
            top5 = mean([finite_float(r.get("mean_top5_overlap")) / 5.0 for r in rows])
            survival = mean(
                [
                    1.0
                    if r.get("first_divergence") is None
                    else max(
                        0.0,
                        min(
                            1.0,
                            int(r.get("first_divergence") or 0)
                            / max(1, int(r.get("steps_compared") or 1)),
                        ),
                    )
                    for r in rows
                ]
            )
        summary.append(
            {
                "mode": mode,
                "baseline": mode == baseline,
                "identicality": identical,
                "top1_agreement": top1,
                "top5_ratio": top5,
                "prefix_survival": survival,
                "bytes_per_head": bytes_head,
                "memory_ratio_vs_fp32": (bytes_head / fp32_bytes) if bytes_head else math.nan,
                "prefill_tok_s": perf_row.get("prefill_tok_s", math.nan),
                "decode_tok_s": perf_row.get("decode_tok_s", math.nan),
                "prompt_count": len(rows) if mode != baseline else len({r["prompt"] for r in prompt_rows}),
            }
        )

    return {
        "gate_dir": str(gate_dir),
        "baseline": baseline,
        "head_dim": head_dim,
        "model": model_details(model, model_name),
        "summary": summary,
        "prompts": prompt_rows,
        "prompt_inputs": prompt_inputs,
        "prompt_text": prompt_texts,
    }


def pct(value: float) -> str:
    return "n/a" if not math.isfinite(value) else f"{value * 100:.1f}%"


def num(value: float, digits: int = 1) -> str:
    return "n/a" if not math.isfinite(value) else f"{value:.{digits}f}"


def bar(value: float, width: int = 120) -> str:
    v = 0.0 if not math.isfinite(value) else max(0.0, min(1.0, value))
    filled = int(round(v * width))
    return (
        f'<svg class="bar" width="{width}" height="10" viewBox="0 0 {width} 10" role="img">'
        f'<rect width="{width}" height="10" rx="2"></rect>'
        f'<rect class="fill" width="{filled}" height="10" rx="2"></rect>'
        "</svg>"
    )


def render_html(data: dict) -> str:
    summary = sorted(data["summary"], key=lambda r: (not r["baseline"], r["mode"]))
    prompts = sorted(data["prompts"], key=lambda r: (r["prompt"], r["mode"]))
    prompt_names = sorted({r["prompt"] for r in prompts})
    modes = [r["mode"] for r in summary]
    prompt_by_key = {(r["prompt"], r["mode"]): r for r in prompts}
    model = data.get("model", {})
    cfg = model.get("config", {}) if isinstance(model.get("config"), dict) else {}
    model_bits = [
        ("Name", model.get("name")),
        ("Path", model.get("path")),
        ("Size", f"{model.get('size_gb'):.2f} GB" if isinstance(model.get("size_gb"), (int, float)) else None),
        ("MD5", model.get("md5")),
        ("Format", model.get("format")),
        ("arch_id", model.get("arch_id")),
        ("Tensors", model.get("tensor_count")),
        ("model_type", cfg.get("model_type")),
        ("hidden_size", cfg.get("hidden_size")),
        ("layers", cfg.get("num_hidden_layers")),
        ("heads", cfg.get("num_attention_heads")),
        ("kv_heads", cfg.get("num_key_value_heads")),
        ("head_dim", cfg.get("head_dim")),
        ("vocab", cfg.get("vocab_size")),
        ("experts", cfg.get("num_experts")),
        ("layer_counts", json.dumps(cfg.get("layer_counts"), sort_keys=True) if cfg.get("layer_counts") else None),
    ]
    model_html = "".join(
        f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>"
        for k, v in model_bits
        if v not in (None, "")
    )

    rows_html = []
    for row in summary:
        cls = ' class="baseline"' if row["baseline"] else ""
        rows_html.append(
            f"<tr{cls}><td>{html.escape(row['mode'])}</td>"
            f"<td data-sort=\"{row['identicality'] if math.isfinite(row['identicality']) else ''}\">{pct(row['identicality'])} {bar(row['identicality'])}</td>"
            f"<td data-sort=\"{row['top1_agreement'] if math.isfinite(row['top1_agreement']) else ''}\">{pct(row['top1_agreement'])}</td>"
            f"<td data-sort=\"{row['top5_ratio'] if math.isfinite(row['top5_ratio']) else ''}\">{pct(row['top5_ratio'])}</td>"
            f"<td data-sort=\"{row['prefix_survival'] if math.isfinite(row['prefix_survival']) else ''}\">{pct(row['prefix_survival'])}</td>"
            f"<td data-sort=\"{row['bytes_per_head'] or ''}\">{row['bytes_per_head'] or 'n/a'}</td>"
            f"<td data-sort=\"{row['memory_ratio_vs_fp32'] if math.isfinite(row['memory_ratio_vs_fp32']) else ''}\">{pct(row['memory_ratio_vs_fp32'])}</td>"
            f"<td data-sort=\"{row['decode_tok_s'] if math.isfinite(row['decode_tok_s']) else ''}\">{num(row['decode_tok_s'])}</td>"
            f"<td data-sort=\"{row['prefill_tok_s'] if math.isfinite(row['prefill_tok_s']) else ''}\">{num(row['prefill_tok_s'])}</td></tr>"
        )

    heat_rows = []
    for prompt in prompt_names:
        cells = [f"<th>{html.escape(prompt)}</th>"]
        for mode in modes:
            if mode == data["baseline"]:
                cells.append('<td class="baseline">100.0%</td>')
                continue
            row = prompt_by_key.get((prompt, mode))
            q = finite_float(row.get("identicality")) if row else math.nan
            shade = 255 - int(max(0.0, min(1.0, q if math.isfinite(q) else 0.0)) * 110)
            cells.append(
                f'<td style="background: rgb({shade},255,{shade});">{pct(q)}</td>'
            )
        heat_rows.append("<tr>" + "".join(cells) + "</tr>")

    excerpts = []
    prompt_text = data.get("prompt_text", {})
    prompt_inputs = data.get("prompt_inputs", {})
    for prompt in sorted(prompt_text):
        excerpts.append(f"<h3>{html.escape(prompt)}</h3><div class=\"excerpt-grid\">")
        inputs = prompt_inputs.get(prompt, {})
        prompt_body = inputs.get("prompt", "")
        system_body = inputs.get("system", "")
        if prompt_body or system_body:
            body = ""
            if system_body:
                body += f"<h4>System Prompt</h4><pre>{html.escape(system_body)}</pre>"
            if prompt_body:
                body += f"<h4>User Prompt</h4><pre>{html.escape(prompt_body)}</pre>"
            excerpts.append(f'<section class="prompt-input">{body}</section>')
        for mode in modes:
            text = prompt_text[prompt].get(mode)
            if not text:
                continue
            excerpts.append(
                f"<section><h4>{html.escape(mode)}</h4><pre>{html.escape(text)}</pre></section>"
            )
        excerpts.append("</div>")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Hipfire KV Quantization Identicality</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 24px; color: #18212f; background: #f7f8fa; }}
h1, h2 {{ margin: 0 0 12px; }}
h2 {{ margin-top: 28px; }}
table {{ border-collapse: collapse; width: 100%; background: white; }}
th, td {{ border: 1px solid #d6dae1; padding: 8px 10px; text-align: left; vertical-align: middle; }}
th {{ background: #eef1f5; }}
th.sortable {{ cursor: pointer; user-select: none; }}
th.sortable::after {{ content: " ↕"; color: #7a8594; font-weight: 400; }}
th.sortable.asc::after {{ content: " ↑"; color: #18212f; }}
th.sortable.desc::after {{ content: " ↓"; color: #18212f; }}
tr.baseline, td.baseline {{ background: #e8f4ff; font-weight: 600; }}
.bar rect:first-child {{ fill: #dde3eb; }}
.bar .fill {{ fill: #2166ac; }}
.meta {{ color: #526070; margin-bottom: 20px; }}
.excerpt-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; }}
section {{ background: white; border: 1px solid #d6dae1; border-radius: 6px; padding: 10px; }}
section.prompt-input {{ grid-column: 1 / -1; border-left: 4px solid #2166ac; }}
h4 {{ margin: 0 0 8px; }}
pre {{ white-space: pre-wrap; overflow-wrap: anywhere; margin: 0; font-size: 12px; line-height: 1.35; }}
</style>
</head>
<body>
<h1>Hipfire KV Quantization Identicality</h1>
<div class="meta">Baseline: <strong>{html.escape(data['baseline'])}</strong> |
head_dim={data['head_dim']} | source: {html.escape(data['gate_dir'])}</div>
<h2>Model</h2>
<table class="model-table">
{model_html}
</table>
<h2>Mode Summary</h2>
<table id="mode-summary">
<tr><th class="sortable">Mode</th><th class="sortable">Identicality vs baseline</th><th class="sortable">Top-1</th><th class="sortable">Top-5</th><th class="sortable">Prefix survival</th><th class="sortable">B/head</th><th class="sortable">Memory vs fp32</th><th class="sortable">Decode tok/s</th><th class="sortable">Prefill tok/s</th></tr>
{''.join(rows_html)}
</table>
<h2>Prompt Heatmap</h2>
<table>
<tr><th>Prompt</th>{''.join(f'<th>{html.escape(m)}</th>' for m in modes)}</tr>
{''.join(heat_rows)}
</table>
<h2>Output Excerpts</h2>
{''.join(excerpts)}
<script>
document.querySelectorAll("#mode-summary th.sortable").forEach((th, idx) => {{
  th.addEventListener("click", () => {{
    const table = th.closest("table");
    const tbody = table.tBodies[0] || table;
    const rows = Array.from(table.querySelectorAll("tr")).slice(1);
    const desc = !th.classList.contains("desc");
    table.querySelectorAll("th").forEach(h => h.classList.remove("asc", "desc"));
    th.classList.add(desc ? "desc" : "asc");
    rows.sort((a, b) => {{
      const av = a.children[idx]?.dataset.sort || a.children[idx]?.textContent || "";
      const bv = b.children[idx]?.dataset.sort || b.children[idx]?.textContent || "";
      const an = Number(av), bn = Number(bv);
      const cmp = Number.isFinite(an) && Number.isFinite(bn)
        ? an - bn
        : String(av).localeCompare(String(bv));
      return desc ? -cmp : cmp;
    }});
    rows.forEach(r => tbody.appendChild(r));
  }});
}});
</script>
</body>
</html>
"""


def json_clean(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: json_clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_clean(v) for v in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("gate_dir", type=Path)
    parser.add_argument("--baseline", default="fp32")
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--model-name")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--excerpt-chars", type=int, default=900)
    args = parser.parse_args()

    data = collect(args.gate_dir, args.baseline, args.head_dim, args.excerpt_chars, args.model, args.model_name)
    html_out = args.out or (args.gate_dir / "dashboard.html")
    json_out = args.json_out or (args.gate_dir / "identicality.json")
    html_out.write_text(render_html(data))
    json_out.write_text(json.dumps(json_clean(data), indent=2, sort_keys=True, allow_nan=False))
    print(f"dashboard: {html_out}")
    print(f"identicality_json: {json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
