#!/usr/bin/env python3
"""Capture dots.ocr layout output via a running vLLM instance.

Complements scripts/capture_dots_ocr_reference.py — the CPU+bf16 HF
path produces coherent prefill logits but a degenerate greedy decode
(collapses into a repeated-token attractor after ~5 generated tokens).
This script asks a vLLM server (GPU + flash_attn + bf16, the model's
intended deployment configuration) to do the layout extraction and
saves the real JSON output for the phase-4 OCR coherence gate.

Run from repo root:
    .venv/bin/python scripts/capture_dots_ocr_vllm.py http://localhost:8000

Override the image / output path with --image / --out if needed.

What this captures:
- The full text response from vLLM (the layout JSON, generated to EOS
  or to --max-tokens).
- Best-effort JSON parse of the response.
- Request metadata: model name (as vLLM advertised it), endpoint,
  prompt template key, image md5, timestamps.

This artifact does NOT capture logits — vLLM's standard chat-completions
endpoint doesn't return them. For per-position logit reference,
use scripts/capture_dots_ocr_reference.py (the HF/CPU path).

Output location: benchmarks/references/dots_ocr_smoke_001_vllm.json
(distinct from the HF-path artifact at dots_ocr_smoke_001.json).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import urllib.request

REPO = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE = REPO / "benchmarks" / "images" / "dots_ocr_smoke_001.jpg"
DEFAULT_OUT = REPO / "benchmarks" / "references" / "dots_ocr_smoke_001_vllm.json"

# Same prompt the HF-path capture uses — keep these byte-identical so
# the two captures are comparable. From dots_ocr/utils/prompts.py:
# prompt_layout_all_en.
PROMPT_LAYOUT_ALL_EN = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
"""


def md5(path: Path) -> str:
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("endpoint", help="vLLM base URL, e.g. http://localhost:8000")
    p.add_argument("--model", default=None,
                   help="vLLM model name. If omitted, queries /v1/models and uses the first one.")
    p.add_argument("--image", default=str(DEFAULT_IMAGE),
                   help=f"image path (default: {DEFAULT_IMAGE.relative_to(REPO)})")
    p.add_argument("--out", default=str(DEFAULT_OUT),
                   help=f"output JSON path (default: {DEFAULT_OUT.relative_to(REPO)})")
    p.add_argument("--max-tokens", type=int, default=16384,
                   help="max completion tokens (default: 16384 — full layout)")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="sampling temperature (default: 0.0 = greedy)")
    args = p.parse_args()

    image_path = Path(args.image).resolve()
    out_path = Path(args.out).resolve()
    if not image_path.exists():
        print(f"error: image not found: {image_path}", file=sys.stderr)
        return 1
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve model name if not provided.
    model = args.model
    if model is None:
        try:
            with urllib.request.urlopen(urljoin(args.endpoint, "/v1/models")) as r:
                payload = json.load(r)
            ids = [m["id"] for m in payload.get("data", [])]
            if not ids:
                print(f"error: /v1/models returned no models; pass --model", file=sys.stderr)
                return 1
            model = ids[0]
            print(f"detected vLLM model: {model}  (others: {ids[1:]})")
        except Exception as e:
            print(f"error: could not query /v1/models at {args.endpoint}: {e}", file=sys.stderr)
            print(f"hint: pass --model <name> explicitly", file=sys.stderr)
            return 1

    image_bytes = image_path.read_bytes()
    image_md5 = hashlib.md5(image_bytes).hexdigest()
    b64 = base64.b64encode(image_bytes).decode("ascii")
    mime = "image/jpeg" if image_path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    data_url = f"data:{mime};base64,{b64}"
    print(f"image: {image_path}  ({len(image_bytes)} bytes, md5={image_md5})")

    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": PROMPT_LAYOUT_ALL_EN},
                ],
            }
        ],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": False,
    }

    print(f"POST {args.endpoint}/v1/chat/completions  (max_tokens={args.max_tokens}, temp={args.temperature})")
    t0 = time.time()
    req = urllib.request.Request(
        urljoin(args.endpoint, "/v1/chat/completions"),
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            resp = json.load(r)
    except Exception as e:
        print(f"error: request failed: {e}", file=sys.stderr)
        return 1
    dt = time.time() - t0
    print(f"  response in {dt:.1f}s")

    # Extract the assistant text.
    try:
        text = resp["choices"][0]["message"]["content"]
    except Exception:
        print(f"error: unexpected response shape; got keys={list(resp.keys())}", file=sys.stderr)
        out_path.write_text(json.dumps(resp, indent=2) + "\n")
        print(f"  raw response saved to {out_path}", file=sys.stderr)
        return 1
    print(f"  completion: {len(text)} chars")

    # Best-effort JSON parse.
    parsed_json = None
    parse_status = "ok"
    try:
        parsed_json = json.loads(text)
    except Exception as e:
        parse_status = f"unparseable: {type(e).__name__}: {e}"

    artifact = {
        "source": "vllm",
        "vllm_endpoint": args.endpoint,
        "vllm_model": model,
        "request_wall_seconds": dt,
        "image_path": str(image_path.relative_to(REPO)),
        "image_md5": image_md5,
        "image_byte_count": len(image_bytes),
        "prompt_template_key": "prompt_layout_all_en",
        "prompt_template_text": PROMPT_LAYOUT_ALL_EN,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "raw_response": resp,
        "completion_text": text,
        "parsed_json": parsed_json,
        "parse_status": parse_status,
    }
    out_path.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"wrote: {out_path} ({out_path.stat().st_size} bytes)")
    print(f"  parse_status: {parse_status}")
    if parsed_json is not None:
        try:
            print(f"  parsed: {len(parsed_json)} layout elements")
        except TypeError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
