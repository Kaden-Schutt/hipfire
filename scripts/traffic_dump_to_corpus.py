#!/usr/bin/env python3
"""Convert an agent traffic dump into MTP-corpus JSONL for build_mtp_vocab_sidecar.py.

Handles the `pi`-agent traffic-dump shape (one JSON object per line):
  - response record: {"direction":"response","model":...,"message":{"role":"assistant",
                      "content":[{"type":"thinking",...},{"type":"text",...},
                                 {"type":"toolCall","name":...,"arguments":{...}}]}}
  - request record:  {"direction":"request","payload":{"model":...,"messages":[...]}}
  - response_headers / other directions are ignored.

Output side only: from each ASSISTANT generation we reconstruct
`<think>…</think>` + text + `<tool_call>…</tool_call>` and emit {"output": <text>}
— the exact format the sidecar builder's `--corpus-jsonl` consumes. Response
bodies are the richest source (full final turn + reasoning); request `messages`
histories are harvested too (recovers assistant turns captured before response
bodies were logged). Everything is deduped by output hash.

Usage:
  python scripts/traffic_dump_to_corpus.py /tmp/2026-05-25.jsonl \
      --output corpus/agent_2026-05-25.jsonl [--model-filter qwen3.6] [--responses-only]
"""
import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path


def _from_blocks(content) -> str:
    """Anthropic-style content-blocks list -> reconstructed assistant text."""
    if isinstance(content, str):
        return content
    parts = []
    for b in (content or []):
        if not isinstance(b, dict):
            continue
        t = b.get("type")
        if t == "thinking" and b.get("thinking"):
            parts.append(f"<think>\n{b['thinking']}\n</think>")
        elif t == "text" and isinstance(b.get("text"), str) and b["text"].strip():
            parts.append(b["text"])
        elif t in ("toolCall", "tool_call"):
            args = b.get("arguments")
            args = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
            parts.append(f'<tool_call>\n{{"name": "{b.get("name")}", "arguments": {args}}}\n</tool_call>')
    return "\n".join(p for p in parts if p)


def _from_openai_msg(msg: dict) -> str:
    """OpenAI-style assistant message (content str/blocks + tool_calls) -> text."""
    parts = []
    rc = msg.get("reasoning_content") or msg.get("reasoning")
    if isinstance(rc, str) and rc.strip():
        parts.append(f"<think>\n{rc}\n</think>")
    c = msg.get("content")
    if isinstance(c, str) and c.strip():
        parts.append(c)
    elif isinstance(c, list):
        blk = _from_blocks(c)
        if blk:
            parts.append(blk)
    for tc in (msg.get("tool_calls") or []):
        fn = tc.get("function", tc)
        args = fn.get("arguments")
        args = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
        parts.append(f'<tool_call>\n{{"name": "{fn.get("name")}", "arguments": {args}}}\n</tool_call>')
    return "\n".join(p for p in parts if p)


def _model_of(rec: dict) -> str:
    m = rec.get("model")
    if m:
        return m
    p = rec.get("payload")
    if isinstance(p, str):
        try:
            p = json.loads(p)
        except json.JSONDecodeError:
            return ""
    if isinstance(p, dict):
        return p.get("model", "")
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="traffic-dump JSONL file(s) or globs")
    ap.add_argument("--output", required=True, help="output corpus JSONL")
    ap.add_argument("--model-filter", default=None,
                    help="case-insensitive substring; drop records whose model doesn't match "
                         "(use to keep only Qwen3.6 traffic)")
    ap.add_argument("--responses-only", action="store_true",
                    help="only harvest response bodies; skip request message-history turns")
    ap.add_argument("--append", action="store_true",
                    help="append to existing output file instead of overwriting; "
                         "pre-loads existing hashes to prevent duplicates")
    args = ap.parse_args()

    paths = []
    for pat in args.inputs:
        paths.extend(sorted(glob.glob(pat)))
    if not paths:
        sys.stderr.write("ERROR: no input files matched\n")
        return 1

    mf = args.model_filter.lower() if args.model_filter else None
    seen: set[str] = set()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {"lines": 0, "resp": 0, "req_turns": 0, "dropped_model": 0, "dup": 0, "written": 0}

    if args.append and out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                text = rec.get("output")
                if isinstance(text, str) and text.strip():
                    seen.add(hashlib.sha1(text.encode("utf-8")).hexdigest())
            except json.JSONDecodeError:
                pass
        mode = "a"
    else:
        mode = "w"

    with open(out_path, mode, encoding="utf-8") as out:
        for fp in paths:
            for line in open(fp, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                stats["lines"] += 1
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if mf and mf not in _model_of(rec).lower():
                    stats["dropped_model"] += 1
                    continue
                outputs = []
                direction = rec.get("direction")
                if direction == "response" and isinstance(rec.get("message"), dict):
                    if rec["message"].get("role") == "assistant":
                        outputs.append((_from_blocks(rec["message"].get("content")), "resp"))
                elif direction == "response" and rec.get("payload"):
                    p = rec["payload"]
                    if isinstance(p, str):
                        try:
                            p = json.loads(p)
                        except json.JSONDecodeError:
                            p = {}
                    for ch in (p.get("choices") or []):
                        m = ch.get("message") or ch.get("delta") or {}
                        outputs.append((_from_openai_msg(m), "resp"))
                elif direction == "request" and not args.responses_only:
                    p = rec.get("payload")
                    if isinstance(p, str):
                        try:
                            p = json.loads(p)
                        except json.JSONDecodeError:
                            p = {}
                    for m in (p.get("messages") or []):
                        if isinstance(m, dict) and m.get("role") == "assistant":
                            outputs.append((_from_openai_msg(m), "req"))
                for text, src in outputs:
                    if not text or not text.strip():
                        continue
                    h = hashlib.sha1(text.encode("utf-8")).hexdigest()
                    if h in seen:
                        stats["dup"] += 1
                        continue
                    seen.add(h)
                    out.write(json.dumps({"output": text, "src": src,
                                          "model": _model_of(rec)}, ensure_ascii=False) + "\n")
                    stats["written"] += 1
                    stats["resp" if src == "resp" else "req_turns"] += 1

    print(f"in={stats['lines']} lines | wrote {stats['written']} unique assistant outputs "
          f"({stats['resp']} from responses, {stats['req_turns']} from request histories) | "
          f"{stats['dup']} dup, {stats['dropped_model']} dropped-by-model -> {out_path}",
          file=sys.stderr)
    return 0 if stats["written"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
