#!/usr/bin/env python3
"""qwen3.5-9b DFlash prefix-cache perf matrix (committed reproducer).

Run from the repo root:
    python3 benchmarks/results/qwen35-9b-dflash-pfix.py

Axes: prefix-cache {on,off} x dflash {on,off} x sampling {greedy,sampled},
each as a 3-turn multiturn conversation. 3 fresh-process reps/cell, medians.
Result schema + provenance: benchmarks/results/qwen35-9b-dflash-pfix.json.
Requires ~/.hipfire/models/qwen3.5-9b.mq4 + qwen35-9b-dflash-mq4.hf4 and a
release daemon at target/release/examples/daemon. GPU: gfx1201 / ROCm 7.2.0."""

import json, os, subprocess, sys, time, statistics, urllib.request
from pathlib import Path
sys.path.insert(0, "scripts")
import lfm_serve_harness as H

MODEL = "/home/kaden/.hipfire/models/qwen3.5-9b.mq4"
DRAFT = "/home/kaden/.hipfire/models/qwen35-9b-dflash-mq4.hf4"
REPO = Path("scripts").resolve().parent
MAXTOK = 512
REPS = 3
TURNS = [
    "Write a Python function that checks whether a number is prime.",
    "Now modify it to return a list of all primes up to n.",
    "Add a docstring and PEP 484 type hints to that function.",
]

def spawn(dflash, cache):
    H._proc = None; H._daemon_link = None
    home, log, port = H.allocate_runtime_defaults(None, None, 0)
    daemon = Path("target/release/examples/daemon").resolve()
    daemon_exec = H._prepare_daemon(daemon, Path(home).parent)
    hh = Path(home) / ".hipfire"; hh.mkdir(parents=True, exist_ok=True)
    (hh / "config.json").write_text(json.dumps({
        "max_seq": 8192, "mtp_mode": "off", "ngram_mode": "off", "max_tokens": MAXTOK,
        "kv_mode": "q8", "dflash_mode": "auto" if dflash else "off",
    }))
    env = dict(os.environ, HOME=home, HIP_VISIBLE_DEVICES="0",
               HIPFIRE_DAEMON_BIN=str(daemon_exec), HIPFIRE_DAEMON_NAME=daemon_exec.name,
               HIPFIRE_KV_MODE="q8", HIPFIRE_CASK_OFF="1", HIPFIRE_MODEL=MODEL,
               HIPFIRE_DFLASH_DRAFT=(DRAFT if dflash else ""))
    if not cache:
        env["HIPFIRE_QWEN_PROMPT_CACHE"] = "0"
    lp = Path(log); lp.write_text("")
    lh = lp.open("a")
    H._proc = subprocess.Popen(["/home/kaden/.bun/bin/bun", "cli/index.ts", "serve", "127.0.0.1", str(port)],
                               cwd=REPO, env=env, stdout=lh, stderr=subprocess.STDOUT, start_new_session=True)
    lh.close()
    deadline = time.monotonic() + 300
    warmed = False
    while time.monotonic() < deadline:
        text = lp.read_text(errors="replace")
        warmed = warmed or "warm-up complete" in text
        if H._proc.poll() is not None:
            raise RuntimeError(f"serve exited {H._proc.returncode}\n{H._log_tail(lp)}")
        if H.FATAL_WARMUP.search(text):
            raise RuntimeError(f"serve failed warm-up\n{H._log_tail(lp)}")
        if warmed and H._health_ready(port):
            return port, lp
        time.sleep(0.5)
    raise TimeoutError("serve did not warm")

def conv(port, sampled):
    """One fresh 3-turn conversation. Returns list of per-turn dicts."""
    rows, messages = [], []
    for u in TURNS:
        messages.append({"role": "user", "content": u})
        body = {"model": MODEL, "messages": messages, "max_tokens": MAXTOK, "stream": True,
                "stream_options": {"include_usage": True}, "reasoning_effort": "none"}
        if sampled:
            body.update(temperature=0.7, top_p=0.8, seed=4242)
        else:
            body.update(temperature=0.0, seed=4242)
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        content, usage, timings = [], {}, {}
        for raw in urllib.request.urlopen(req, timeout=300):
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            pp = line[5:].strip()
            if pp == "[DONE]":
                break
            try:
                ch = json.loads(pp)
            except json.JSONDecodeError:
                continue
            if ch.get("usage"):
                usage = ch["usage"]
            if ch.get("timings"):
                timings = ch["timings"]
            content.append(((ch.get("choices") or [{}])[0].get("delta") or {}).get("content") or "")
        messages.append({"role": "assistant", "content": "".join(content)})
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        rows.append({
            "ptok": usage.get("prompt_tokens"), "cached": cached,
            "gen": timings.get("tokens"), "pf_tok": timings.get("prefill_tokens"),
            "ttft_ms": timings.get("ttft_ms"), "dec_toks": timings.get("decode_tok_s"),
            "tau": timings.get("tau"),
        })
    return rows

def med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 1) if xs else None

CONFIGS = [(d, c, s) for d in (False, True) for c in (False, True) for s in (False, True)]
agg = {}
for (dflash, cache, sampled) in CONFIGS:
    label = f"dflash={'on' if dflash else 'off':<3} cache={'on' if cache else 'off':<3} samp={'on' if sampled else 'off':<3}"
    reps = []
    for r in range(REPS):
        port, lp = spawn(dflash, cache)
        try:
            reps.append(conv(port, sampled))
        finally:
            H._kill_server()
    # median across reps per turn
    turns = []
    for ti in range(len(TURNS)):
        turns.append({
            "dec_toks": med([reps[r][ti]["dec_toks"] for r in range(REPS)]),
            "ttft_ms":  med([reps[r][ti]["ttft_ms"]  for r in range(REPS)]),
            "tau":      med([reps[r][ti]["tau"]      for r in range(REPS)]),
            "cached":   reps[-1][ti]["cached"],
            "pf_tok":   reps[-1][ti]["pf_tok"],
            "gen":      reps[-1][ti]["gen"],
            "dec_all":  [reps[r][ti]["dec_toks"] for r in range(REPS)],
        })
    agg[label] = turns
    print(f"{label}: " + " | ".join(
        f"t{ti+1} dec={t['dec_toks']} (raw {t['dec_all']}) ttft={t['ttft_ms']} cached={t['cached']} tau={t['tau']}"
        for ti, t in enumerate(turns)), flush=True)

def cell(t):
    return f"{t['dec_toks']!s:>6}/{t['ttft_ms']!s:>6}/{t['cached']!s:>5}/{t['tau']!s:>5}"

print("\n==== PERF MATRIX (median of 3) — cell = decode tok/s / ttft ms / cached tok / tau ====")
hdr = f"{'config':30} | {'turn1':26} | {'turn2':26} | {'turn3':26}"
print(hdr); print("-"*len(hdr))
for label, turns in agg.items():
    print(f"{label:30} | {cell(turns[0]):26} | {cell(turns[1]):26} | {cell(turns[2]):26}")
Path("/tmp/lfm_bench/perf_matrix_v2_result.json").write_text(json.dumps(agg, indent=2))
print("\nraw -> /tmp/lfm_bench/perf_matrix_v2_result.json")
