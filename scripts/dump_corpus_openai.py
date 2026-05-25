#!/usr/bin/env python3
"""Generate an MTP-corpus dump by calling an OpenAI-compatible chat endpoint
(vLLM or OpenRouter) and saving each assistant generation as JSONL.

The output JSONL is consumed directly by build_mtp_vocab_sidecar.py
(`--corpus-jsonl`). We capture the OUTPUT side only — the tokens the draft head
must learn to predict: reasoning + content + serialized tool_calls, joined into
an `output` field. The sidecar builder force-includes all special/added tokens
(<think>, <tool_call>, <|im_end|>, chatml frame) regardless, so exact frame
reconstruction here is not required — what matters is covering the CONTENT
tokens (JSON keys, identifiers, words, punctuation) of real outputs.

IMPORTANT — distribution match: point this at a genuine **Qwen3.6-27B** endpoint
(same tokenizer/weights as the trunk). Full-precision vLLM is ideal; OpenRouter
fp8/AWQ is fine for v1 frequency counting (negligible argmax shift). If you proxy
multiple models, only Qwen3.6 outputs belong in the corpus.

No third-party deps (stdlib urllib only).

Usage:
  # vLLM (local, usually no key):
  python scripts/dump_corpus_openai.py \
      --base-url http://localhost:8000/v1 --model Qwen/Qwen3.6-27B \
      --seeds seeds.jsonl --output corpus/qwen36_dump.jsonl --concurrency 8

  # OpenRouter:
  OPENROUTER_API_KEY=sk-... python scripts/dump_corpus_openai.py \
      --base-url https://openrouter.ai/api/v1 --model qwen/qwen3.6-27b \
      --seeds seeds.jsonl --output corpus/qwen36_dump.jsonl

Seeds file: one per line, each either
  - plain text                         -> {"role":"user","content":<line>}
  - {"prompt": "...", "system": "...", "tools": [...]}
  - {"messages": [...], "tools": [...]}   (full control)
Include `tools` (OpenAI tool schema) on seeds where you want tool_call output.

Notes:
  - temperature defaults to 0.7 for output DIVERSITY (wider vocab coverage). Use
    --temperature 0 to match a greedy-inference argmax distribution more tightly
    (less diverse, closer to the v2 trunk-argmax signal).
  - --resume skips seeds already present in --output (keyed by a stable seed id),
    so long runs are interruptible. Output is opened in append mode.
"""
import argparse
import hashlib
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def load_seeds(path: str) -> list[dict]:
    seeds: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line[0] == "{":
            obj = json.loads(line)
            if "messages" in obj:
                msgs = obj["messages"]
            else:
                msgs = []
                if obj.get("system"):
                    msgs.append({"role": "system", "content": obj["system"]})
                msgs.append({"role": "user", "content": obj.get("prompt", "")})
            seeds.append({"messages": msgs, "tools": obj.get("tools")})
        else:
            seeds.append({"messages": [{"role": "user", "content": line}], "tools": None})
    return seeds


def seed_id(seed: dict) -> str:
    blob = json.dumps(seed["messages"], sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def reconstruct_output(msg: dict) -> str:
    """Approximate the on-the-wire assistant generation (reasoning + content +
    tool_calls). Frame/special tokens are force-included by the sidecar builder,
    so this only needs to surface the content tokens."""
    parts: list[str] = []
    rc = msg.get("reasoning_content") or msg.get("reasoning")
    if isinstance(rc, str) and rc.strip():
        parts.append(f"<think>\n{rc}\n</think>")
    c = msg.get("content")
    if isinstance(c, str) and c.strip():
        parts.append(c)
    for tc in (msg.get("tool_calls") or []):
        fn = tc.get("function", tc)
        name = fn.get("name")
        args = fn.get("arguments")
        arg_str = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
        parts.append(f'<tool_call>\n{{"name": "{name}", "arguments": {arg_str}}}\n</tool_call>')
    return "\n".join(parts)


def call(base_url, model, key, seed, max_tokens, temperature, extra_headers, retries=3):
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": seed["messages"],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if seed.get("tools"):
        body["tools"] = seed["tools"]
    data = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    headers.update(extra_headers)
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.loads(r.read())
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last = e
            time.sleep(2 * (attempt + 1))
    raise last


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", required=True, help="e.g. http://localhost:8000/v1 or https://openrouter.ai/api/v1")
    p.add_argument("--model", required=True, help="model slug (must be Qwen3.6-27B)")
    p.add_argument("--seeds", required=True, help="seed prompts file (see header)")
    p.add_argument("--output", required=True, help="output JSONL (append mode)")
    p.add_argument("--api-key-env", default=None,
                   help="env var holding the API key (default: try OPENROUTER_API_KEY then OPENAI_API_KEY)")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--resume", action="store_true", help="skip seeds already in --output")
    args = p.parse_args()

    if args.api_key_env:
        key = os.environ.get(args.api_key_env)
    else:
        key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    extra_headers = {}
    if "openrouter" in args.base_url:
        extra_headers["HTTP-Referer"] = "https://github.com/hipfire"
        extra_headers["X-Title"] = "hipfire-mtp-corpus"

    seeds = load_seeds(args.seeds)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    if args.resume and out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            try:
                sid = json.loads(line).get("seed_id")
                if sid:
                    done.add(sid)
            except json.JSONDecodeError:
                pass
    todo = [s for s in seeds if seed_id(s) not in done]
    print(f"seeds: {len(seeds)} total, {len(done)} already done, {len(todo)} to generate",
          file=sys.stderr)

    lock = threading.Lock()
    fh = open(out_path, "a", encoding="utf-8")
    stats = {"ok": 0, "err": 0, "toks": 0}

    def work(seed: dict) -> None:
        sid = seed_id(seed)
        try:
            resp = call(args.base_url, args.model, key, seed,
                        args.max_tokens, args.temperature, extra_headers)
            msg = resp["choices"][0]["message"]
            rec = {
                "seed_id": sid,
                "output": reconstruct_output(msg),
                "content": msg.get("content"),
                "reasoning_content": msg.get("reasoning_content") or msg.get("reasoning"),
                "tool_calls": msg.get("tool_calls"),
                "completion_tokens": resp.get("usage", {}).get("completion_tokens"),
            }
            with lock:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                stats["ok"] += 1
                stats["toks"] += rec["completion_tokens"] or 0
                if stats["ok"] % 20 == 0:
                    print(f"  {stats['ok']} ok / {stats['err']} err / ~{stats['toks']} completion toks",
                          file=sys.stderr)
        except Exception as e:  # noqa: BLE001 — log and continue, don't kill the run
            with lock:
                stats["err"] += 1
            print(f"  ERR seed {sid}: {e}", file=sys.stderr)

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        list(ex.map(work, todo))
    fh.close()
    print(f"done: {stats['ok']} ok, {stats['err']} err, ~{stats['toks']} completion tokens "
          f"-> {out_path}", file=sys.stderr)
    return 0 if stats["ok"] > 0 or not todo else 1


if __name__ == "__main__":
    sys.exit(main())
