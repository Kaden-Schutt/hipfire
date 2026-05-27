#!/usr/bin/env python3
"""Multi-turn corpus dumper for MTP vocab sidecar building.

Drives any OpenAI-compatible chat endpoint (vLLM / OpenRouter) through
MULTI-TURN conversations. The assistant's output from each turn is fed back
as context for the next turn in the same seed, so the model sees the full
conversation history - exactly as it would in a real multi-turn session.

This is the key lever for realistic MTP vocab sidecars: single-turn seeds
only capture the distribution of *first* assistant turns. Multi-turn seeds
cover follow-up reasoning, clarification, tool-call loops, code review, etc.

Output JSONL is consumed by build_mtp_vocab_sidecar.py (`--corpus-jsonl`).
Each turn emits one JSONL record with a `turn` field so you can inspect
turn-level coverage in the sidecar builder.

IMPORTANT - distribution match: point this at a genuine **Qwen3.6-27B**
endpoint (same tokenizer/weights as the trunk). Full-precision vLLM is
ideal; OpenRouter fp8/AWQ is fine for v1 frequency counting.

No third-party deps (stdlib urllib only).

Usage:
  # vLLM (local, usually no key):
  python scripts/dump_corpus_openai_multiturn.py \
      --base-url http://localhost:8000/v1 --model Qwen/Qwen3.6-27B \
      --seeds seeds.jsonl --output corpus/qwen36_mt_dump.jsonl \
      --concurrency 4 --max-tokens 512

  # With plain-text turn files:
  python scripts/dump_corpus_openai_multiturn.py \
      --base-url http://localhost:8000/v1 --model Qwen/Qwen3.6-27B \
      --turns-dir turns/ --output corpus/qwen36_mt_dump.jsonl

  # OpenRouter:
  OPENROUTER_API_KEY=sk-... python scripts/dump_corpus_openai_multiturn.py \
      --base-url https://openrouter.ai/api/v1 --model qwen/qwen3.6-27b \
      --turns-dir turns/ --output corpus/qwen36_mt_dump.jsonl

Input formats (provide one or both):

  --seeds SEEDS.JSONL
    Same as dump_corpus_openai.py: one JSON seed per line.

  --turns-dir TURNS_DIR/
    Plain-text turn files, one per conversation. Each file is a series of
    user prompts separated by a line containing only `----` (four dashes):

      Propose three current cheap SSDs
      ----
      Which of those has the best write endurance for a NAS?
      ----
      Can you compare their MTBF numbers?

    Each paragraph between `----` boundaries becomes a user turn. The first
    turn starts the conversation; each subsequent turn is sent with the full
    conversation history.

Behavior:
  - Sends turn[0] to the API, gets assistant response.
  - Appends assistant response to conversation history.
  - Sends turn[1] (with full history), gets next response.
  - Repeats for all turns.
  - Each turn emits one JSONL record with {"seed_id", "turn", "output", ...}
  - --resume tracks (seed_id, turn) pairs so you can interrupt and continue.

Notes:
  - --max-tokens is per-TURN (each API call). Default 4096: reasoning models
    spend most of a small budget inside <think> and never emit the answer or
    tool_call, so 512 truncates every turn to reasoning-only. Raise to 8192
    for long agentic tool loops.
  - temperature defaults to 0.7 for output DIVERSITY. Use --temperature 0
    for greedy argmax distribution matching.
  - --resume is keyed by (seed_id, turn_number) so a partial run at turn 3
    of seed A will skip turns 0-2 on resume but continue from turn 3.
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


def load_seeds_jsonl(path: str) -> list[dict]:
    """Load multi-turn seeds from JSONL. Each line is an object with 'turns'."""
    seeds: list[dict] = []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            sys.stderr.write(f"WARNING: line {lineno}: not JSON, skipping: {line[:80]}\n")
            continue
        if "turns" not in obj:
            sys.stderr.write(f"WARNING: line {lineno}: no 'turns' key, skipping (not multi-turn)\n")
            continue
        turns = obj["turns"]
        system = obj.get("system")
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        seeds.append({
            "messages_base": messages,
            "turns": turns,
        })
    return seeds


def load_seeds_turns_dir(dir_path: str) -> list[dict]:
    """Load multi-turn seeds from plain-text files in a directory.

    Each file is a series of user prompts separated by a line of exactly
    `----` (four dashes, trimmed). Each segment becomes a user turn.
    """
    seeds: list[dict] = []
    d = Path(dir_path)
    if not d.is_dir():
        sys.stderr.write(f"ERROR: {dir_path} is not a directory\n")
        return seeds

    txt_files = sorted(list(d.glob("*.txt")) + list(d.glob("*.text")) + list(d.glob("*.md")))
    for fp in txt_files:
        text = fp.read_text(encoding="utf-8")
        turns = []
        current = []
        for line in text.splitlines():
            if line.strip() == "----":
                if current:
                    turn_text = "\n".join(current).strip()
                    if turn_text:
                        turns.append(turn_text)
                current = []
            else:
                current.append(line)
        if current:
            turn_text = "\n".join(current).strip()
            if turn_text:
                turns.append(turn_text)

        if not turns:
            continue

        seeds.append({
            "messages_base": [],
            "turns": turns,
        })
    return seeds


def seed_id(seed: dict) -> str:
    """Stable ID from the seed's turn contents."""
    blob_parts = [json.dumps(seed["messages_base"], sort_keys=True, ensure_ascii=False)]
    for turn in seed["turns"]:
        if isinstance(turn, str):
            blob_parts.append(turn)
        elif isinstance(turn, dict):
            blob_parts.append(json.dumps(turn, sort_keys=True, ensure_ascii=False))
        else:
            blob_parts.append(str(turn))
    blob = "\x00".join(blob_parts)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def parse_turn(turn, seed_tools=None):
    """Parse a turn element into (message_dict, tools_to_use, is_inject)."""
    if isinstance(turn, str):
        return {"role": "user", "content": turn}, None, False
    elif isinstance(turn, dict):
        role = turn.get("role", "user")
        if role == "assistant":
            msg = {"role": "assistant", "content": turn.get("content", "")}
            return msg, None, True
        msg = {"role": role, "content": turn.get("content", "")}
        tools = turn.get("tools") if "tools" in turn else seed_tools
        return msg, tools, False
    else:
        return {"role": "user", "content": str(turn)}, None, False


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


def call(base_url, model, key, messages, tools, max_tokens, temperature,
         extra_headers, retries=3):
    """Send a chat completion request and return the parsed response."""
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if tools:
        body["tools"] = tools
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
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError,
                json.JSONDecodeError) as e:
            last = e
            time.sleep(2 * (attempt + 1))
    raise last


def run_multiturn_seed(seed, base_url, model, key, max_tokens, temperature,
                       extra_headers, done_turns, out_path):
    """Run all turns for one multi-turn seed. Returns list of (sid, turn, record)."""
    sid = seed_id(seed)
    messages = list(seed["messages_base"])
    results = []

    for turn_idx, turn_raw in enumerate(seed["turns"]):
        resume_key = f"{sid}:{turn_idx}"
        if resume_key in done_turns:
            continue

        msg, tools, is_inject = parse_turn(turn_raw)

        if is_inject:
            messages.append(msg)
            rec = {
                "seed_id": sid,
                "turn": turn_idx,
                "output": msg.get("content", ""),
                "content": msg.get("content"),
                "reasoning_content": None,
                "tool_calls": None,
                "completion_tokens": None,
                "inject": True,
            }
            results.append((sid, turn_idx, resume_key, rec))
            continue

        messages.append(msg)
        try:
            resp = call(base_url, model, key, messages, tools,
                        max_tokens, temperature, extra_headers)
            assistant_msg = resp["choices"][0]["message"]

            rec = {
                "seed_id": sid,
                "turn": turn_idx,
                "output": reconstruct_output(assistant_msg),
                "content": assistant_msg.get("content"),
                "reasoning_content": (assistant_msg.get("reasoning_content") or
                                      assistant_msg.get("reasoning")),
                "tool_calls": assistant_msg.get("tool_calls"),
                "completion_tokens": resp.get("usage", {}).get("completion_tokens"),
            }
            results.append((sid, turn_idx, resume_key, rec))

            feedback_msg = {
                "role": "assistant",
                "content": assistant_msg.get("content"),
            }
            rc = assistant_msg.get("reasoning_content") or assistant_msg.get("reasoning")
            if rc:
                feedback_msg["reasoning_content"] = rc
            if assistant_msg.get("tool_calls"):
                feedback_msg["tool_calls"] = assistant_msg["tool_calls"]
            messages.append(feedback_msg)
        except Exception as e:
            results.append((sid, turn_idx, resume_key, None, str(e)))

    return results


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", required=True,
                   help="e.g. http://localhost:8000/v1 or https://openrouter.ai/api/v1")
    p.add_argument("--model", required=True,
                   help="model slug (must be Qwen3.6-27B)")
    p.add_argument("--seeds", default=None,
                   help="seeds JSONL file (optional, use --turns-dir for plain text)")
    p.add_argument("--turns-dir", default=None,
                   help="directory of plain-text turn files separated by '----'")
    p.add_argument("--output", required=True,
                   help="output JSONL (append mode)")
    p.add_argument("--api-key-env", default=None,
                   help="env var holding the API key (default: try OPENROUTER_API_KEY then OPENAI_API_KEY)")
    p.add_argument("--max-tokens", type=int, default=4096,
                   help="max tokens PER TURN (default 4096; reasoning models spend "
                        "most of a small budget inside <think> and never reach the "
                        "answer/tool_call, so 512 truncates every turn to reasoning-"
                        "only — raise to 8192 for long agentic tool loops)")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="temperature (default 0.7 for diversity)")
    p.add_argument("--concurrency", type=int, default=4,
                   help="parallel seeds (default 4)")
    p.add_argument("--resume", action="store_true",
                   help="skip (seed_id, turn) pairs already in --output")
    args = p.parse_args()

    if not args.seeds and not args.turns_dir:
        sys.stderr.write("ERROR: must provide --seeds or --turns-dir\n")
        return 1

    if args.api_key_env:
        key = os.environ.get(args.api_key_env)
    else:
        key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    extra_headers = {}
    if "openrouter" in args.base_url:
        extra_headers["HTTP-Referer"] = "https://github.com/hipfire"
        extra_headers["X-Title"] = "hipfire-mtp-corpus-multiturn"

    seeds: list[dict] = []
    if args.seeds:
        seeds.extend(load_seeds_jsonl(args.seeds))
    if args.turns_dir:
        seeds.extend(load_seeds_turns_dir(args.turns_dir))

    if not seeds:
        sys.stderr.write("ERROR: no valid seeds found\n")
        return 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_turns: set[str] = set()
    if args.resume and out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                sid = rec.get("seed_id")
                turn = rec.get("turn")
                if sid is not None and turn is not None:
                    done_turns.add(f"{sid}:{turn}")
            except json.JSONDecodeError:
                pass

    total_turns = sum(len(s["turns"]) for s in seeds)
    done_count = sum(
        sum(1 for ti in range(len(s["turns"])) if f"{seed_id(s)}:{ti}" in done_turns)
        for s in seeds
    )
    todo_turns = total_turns - done_count
    sys.stderr.write(f"seeds: {len(seeds)} total, {total_turns} turns, "
                     f"{done_count} already done, {todo_turns} to generate\n")

    lock = threading.Lock()
    fh = open(out_path, "a", encoding="utf-8")
    stats = {"ok": 0, "err": 0, "toks": 0}

    def work(seed: dict) -> None:
        results = run_multiturn_seed(seed, args.base_url, args.model, key,
                                     args.max_tokens, args.temperature,
                                     extra_headers, done_turns, out_path)
        for item in results:
            sid, turn_idx, resume_key, rec, *err = item
            if rec is None:
                with lock:
                    stats["err"] += 1
                sys.stderr.write(f"  ERR seed {sid} turn {turn_idx}: {err[0] if err else 'unknown'}\n")
                continue
            with lock:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                stats["ok"] += 1
                stats["toks"] += rec.get("completion_tokens") or 0
                if stats["ok"] % 20 == 0:
                    sys.stderr.write(f"  {stats['ok']} ok / {stats['err']} err / "
                                     f"~{stats['toks']} completion toks\n")

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        list(ex.map(work, seeds))
    fh.close()
    sys.stderr.write(f"done: {stats['ok']} ok, {stats['err']} err, "
                     f"~{stats['toks']} completion tokens -> {out_path}\n")
    return 0 if stats["ok"] > 0 or not todo_turns else 1


if __name__ == "__main__":
    sys.exit(main())
