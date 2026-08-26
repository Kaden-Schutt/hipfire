#!/usr/bin/env python3
"""Minimal hipfire daemon JSONL driver for DS4 G0 coherence (pre/post)."""
import json, os, sys, time, subprocess, select, hashlib

DAEMON = os.environ["HIPFIRE_DAEMON_BIN"]
MODEL = os.environ["HIPFIRE_MODEL"]
PROMPT = os.environ.get(
    "HIPFIRE_PROMPT", "Explain what a GPU kernel is in two sentences."
)
MAX_TOKENS = int(os.environ.get("HIPFIRE_MAX_TOKENS", "64"))
TEMP = float(os.environ.get("HIPFIRE_TEMP", "0"))
LABEL = os.environ.get("HIPFIRE_LABEL", "run")
EVENT_LOG = os.environ.get("HIPFIRE_EVENT_LOG", "")
RESULT_JSON = os.environ.get("HIPFIRE_RESULT_JSON", "")
TEXT_OUT = os.environ.get("HIPFIRE_TEXT_OUT", "")
DAEMON_STDERR = os.environ.get("HIPFIRE_DAEMON_STDERR", "")

env = os.environ.copy()
print(f"[driver] label={LABEL}", file=sys.stderr, flush=True)
print(f"[driver] daemon={DAEMON}", file=sys.stderr, flush=True)
print(f"[driver] model={MODEL}", file=sys.stderr, flush=True)
pb = PROMPT.encode("utf-8")
print(
    f"[driver] prompt_bytes={len(pb)} sha256={hashlib.sha256(pb).hexdigest()}",
    file=sys.stderr,
    flush=True,
)
print(f"[driver] max_tokens={MAX_TOKENS} temp={TEMP}", file=sys.stderr, flush=True)

stderr_f = open(DAEMON_STDERR, "w", buffering=1) if DAEMON_STDERR else None

proc = subprocess.Popen(
    [DAEMON],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    env=env,
    text=True,
    bufsize=1,
)


def send(obj):
    line = json.dumps(obj, separators=(",", ":"))
    print(f"[driver] >> {line[:800]}", file=sys.stderr, flush=True)
    proc.stdin.write(line + "\n")
    proc.stdin.flush()


def recv(timeout=3600.0):
    deadline = time.time() + timeout
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            raise TimeoutError("recv timeout")
        if proc.poll() is not None:
            while True:
                chunk = proc.stderr.readline() if proc.stderr else ""
                if not chunk:
                    break
                print(chunk, file=sys.stderr, end="", flush=True)
                if stderr_f:
                    stderr_f.write(chunk)
            while True:
                chunk = proc.stdout.readline() if proc.stdout else ""
                if not chunk:
                    break
                line = chunk.strip()
                if not line:
                    continue
                if EVENT_LOG:
                    with open(EVENT_LOG, "a") as f:
                        f.write(line + "\n")
                print(f"[driver] << {line[:800]}", file=sys.stderr, flush=True)
                try:
                    return json.loads(line)
                except Exception:
                    print(f"[driver] non-json stdout: {line}", file=sys.stderr, flush=True)
            raise RuntimeError(f"daemon exited rc={proc.returncode}")
        ready, _, _ = select.select(
            [proc.stdout, proc.stderr], [], [], min(remaining, 0.5)
        )
        for fd in ready:
            chunk = fd.readline()
            if not chunk:
                continue
            if fd is proc.stderr:
                print(chunk, file=sys.stderr, end="", flush=True)
                if stderr_f:
                    stderr_f.write(chunk)
                continue
            line = chunk.strip()
            if not line:
                continue
            if EVENT_LOG:
                with open(EVENT_LOG, "a") as f:
                    f.write(line + "\n")
            print(f"[driver] << {line[:800]}", file=sys.stderr, flush=True)
            try:
                return json.loads(line)
            except Exception as e:
                print(f"[driver] bad json: {e}: {line[:200]}", file=sys.stderr, flush=True)


try:
    send({"type": "ping"})
    msg = recv(60)
    assert msg.get("type") == "pong", msg

    params = {
        "max_seq": 4096,
        "kv_mode": "q8",
        "kv_backend": "contiguous",
        "speculation": "off",
        "dflash_mode": "off",
        "mtp_mode": "off",
        "ngram_draft": False,
        "dspark_mode": "off",
    }
    send({"type": "load", "model": MODEL, "params": params})
    loaded = None
    t0 = time.time()
    while True:
        msg = recv(3600)
        t = msg.get("type")
        if t == "loaded":
            loaded = msg
            break
        if t == "error":
            print(json.dumps({"error": msg}, indent=2, ensure_ascii=False))
            sys.exit(2)
        print(f"[driver] load-event {t}", file=sys.stderr, flush=True)
    load_s = time.time() - t0
    print(f"[driver] LOADED in {load_s:.1f}s", file=sys.stderr, flush=True)

    req_id = f"g0-{LABEL}"
    attempt_id = 1
    send(
        {
            "type": "generate",
            "id": req_id,
            "attempt_id": attempt_id,
            "prompt": PROMPT,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMP,
        }
    )

    content = []
    events = []
    done = None
    terminal = None
    t1 = time.time()
    while True:
        msg = recv(3600)
        events.append(msg)
        t = msg.get("type")
        if t == "token":
            content.append(msg.get("text") or "")
        elif t == "committed":
            txt = msg.get("text") or ""
            if txt:
                content.append(txt)
        elif t == "commit_ready":
            # Commit immediately — terminal-control timeout is short.
            send({"type": "commit", "id": req_id, "attempt_id": attempt_id})
            done = msg
        elif t == "done":
            done = msg
            terminal = msg
            break
        elif t in ("aborted", "error"):
            terminal = msg
            print(f"[driver] terminal {t}: {msg}", file=sys.stderr, flush=True)
            break
        elif t == "gen_start":
            print(f"[driver] gen_start {msg}", file=sys.stderr, flush=True)
        else:
            print(f"[driver] event {t}", file=sys.stderr, flush=True)

    gen_s = time.time() - t1
    text = "".join(content)
    if isinstance(done, dict):
        for k in ("text", "content", "output"):
            if isinstance(done.get(k), str) and done.get(k):
                if len(done[k]) >= len(text):
                    text = done[k]

    result = {
        "label": LABEL,
        "text": text,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "prompt": PROMPT,
        "prompt_sha256": hashlib.sha256(PROMPT.encode("utf-8")).hexdigest(),
        "max_tokens": MAX_TOKENS,
        "temperature": TEMP,
        "done": done,
        "terminal": terminal,
        "loaded": dict(loaded) if loaded else None,
        "elapsed_gen_s": gen_s,
        "elapsed_load_s": load_s,
        "n_token_events": sum(1 for e in events if e.get("type") == "token"),
        "daemon_bin": DAEMON,
        "model": MODEL,
    }
    out = json.dumps(result, indent=2, ensure_ascii=False)
    print(out)
    if RESULT_JSON:
        with open(RESULT_JSON, "w") as f:
            f.write(out + "\n")
    if TEXT_OUT:
        # Verbatim decoded text, no added trailing newline beyond model output.
        with open(TEXT_OUT, "w") as f:
            f.write(text)

    try:
        send({"type": "unload"})
        recv(120)
    except Exception as e:
        print(f"[driver] unload err {e}", file=sys.stderr, flush=True)
    try:
        proc.stdin.close()
    except Exception:
        pass
    try:
        proc.wait(timeout=60)
    except Exception:
        proc.kill()
    if stderr_f:
        stderr_f.close()
    if terminal and terminal.get("type") == "error":
        sys.exit(3)
    if not text and (not done):
        sys.exit(4)
    sys.exit(0)
except Exception as e:
    print(f"[driver] FATAL {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    try:
        proc.kill()
    except Exception:
        pass
    if stderr_f:
        stderr_f.close()
    sys.exit(99)
