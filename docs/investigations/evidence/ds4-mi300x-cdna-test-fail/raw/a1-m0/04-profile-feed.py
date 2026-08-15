#!/usr/bin/env python3
"""Feed daemon JSONL under rocprof; one load + one generate; print all events."""
import json, os, sys, time, select, hashlib, subprocess

DAEMON = os.environ["HIPFIRE_DAEMON_BIN"]
MODEL = os.environ["HIPFIRE_MODEL"]
PROMPT = os.environ.get("HIPFIRE_PROMPT", "Explain what a GPU kernel is in two sentences.")
MAX_TOKENS = int(os.environ.get("HIPFIRE_MAX_TOKENS", "64"))
TEMP = float(os.environ.get("HIPFIRE_TEMP", "0"))
LABEL = os.environ.get("HIPFIRE_LABEL", "m0-profile")
EVENT_LOG = os.environ.get("HIPFIRE_EVENT_LOG", "")
RESULT_JSON = os.environ.get("HIPFIRE_RESULT_JSON", "")
TEXT_OUT = os.environ.get("HIPFIRE_TEXT_OUT", "")
DAEMON_STDERR = os.environ.get("HIPFIRE_DAEMON_STDERR", "")
# When ROCPROF wraps us, we ARE the app - just run daemon as self if HIPFIRE_PROFILE_SELF=1
# Actually: this script IS launched by rocprof and spawns daemon - that failed before.
# Alternative mode: if argv contains --daemon-only, exec is not used; we expect to BE replaced.
# Better: this script becomes the stdin feeder AND we launch daemon as the ONLY hip process
# by being a thin parent. For rocprof attach to child, use LD_PRELOAD inheritance - it should work
# if we increase ping timeout dramatically and ensure line buffering.

# Root cause last time: ping timeout 60s while rocprof init on child. Increase to 600s.
PING_TIMEOUT = float(os.environ.get("HIPFIRE_PING_TIMEOUT", "600"))
LOAD_TIMEOUT = float(os.environ.get("HIPFIRE_LOAD_TIMEOUT", "900"))
GEN_TIMEOUT = float(os.environ.get("HIPFIRE_GEN_TIMEOUT", "1800"))

stderr_f = open(DAEMON_STDERR, "w", buffering=1) if DAEMON_STDERR else None
env = os.environ.copy()

print(f"[feed] label={LABEL}", file=sys.stderr, flush=True)
print(f"[feed] daemon={DAEMON}", file=sys.stderr, flush=True)
print(f"[feed] model={MODEL}", file=sys.stderr, flush=True)

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
    print(f"[feed] >> {line[:500]}", file=sys.stderr, flush=True)
    proc.stdin.write(line + "\n")
    proc.stdin.flush()

def recv(timeout=600.0):
    deadline = time.time() + timeout
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            raise TimeoutError(f"recv timeout after {timeout}s")
        if proc.poll() is not None:
            # drain
            if proc.stderr:
                for chunk in proc.stderr:
                    print(chunk, file=sys.stderr, end="", flush=True)
                    if stderr_f: stderr_f.write(chunk)
            raise RuntimeError(f"daemon exited rc={proc.returncode}")
        ready, _, _ = select.select([proc.stdout, proc.stderr], [], [], min(remaining, 1.0))
        for fd in ready:
            chunk = fd.readline()
            if not chunk:
                continue
            if fd is proc.stderr:
                print(chunk, file=sys.stderr, end="", flush=True)
                if stderr_f: stderr_f.write(chunk)
                continue
            line = chunk.strip()
            if not line:
                continue
            if EVENT_LOG:
                with open(EVENT_LOG, "a") as f:
                    f.write(line + "\n")
            print(f"[feed] << {line[:500]}", file=sys.stderr, flush=True)
            try:
                return json.loads(line)
            except Exception as e:
                print(f"[feed] bad json: {e}: {line[:200]}", file=sys.stderr, flush=True)

try:
    t0 = time.time()
    send({"type": "ping"})
    msg = recv(PING_TIMEOUT)
    assert msg.get("type") == "pong", msg
    print(f"[feed] pong in {time.time()-t0:.2f}s", file=sys.stderr, flush=True)

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
    t_load = time.time()
    send({"type": "load", "model": MODEL, "params": params})
    loaded = None
    while True:
        msg = recv(LOAD_TIMEOUT)
        t = msg.get("type")
        if t == "loaded":
            loaded = msg
            break
        if t == "error":
            raise RuntimeError(f"load error: {msg}")
        print(f"[feed] load event {t}", file=sys.stderr, flush=True)
    load_s = time.time() - t_load
    print(f"[feed] loaded in {load_s:.2f}s", file=sys.stderr, flush=True)

    # Marker for decode phase wall clock
    print("[feed] DECODE_PHASE_BEGIN", file=sys.stderr, flush=True)
    t1 = time.time()
    req_id = f"g0-{LABEL}"
    attempt_id = 1
    send({
        "type": "generate",
        "id": req_id,
        "attempt_id": attempt_id,
        "prompt": PROMPT,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMP,
    })
    content = []
    events = []
    done = None
    terminal = None
    while True:
        msg = recv(GEN_TIMEOUT)
        events.append(msg)
        t = msg.get("type")
        if t == "token":
            txt = msg.get("text") or msg.get("token") or ""
            if txt:
                content.append(txt)
        elif t == "commit_ready":
            send({"type": "commit", "id": req_id, "attempt_id": attempt_id})
            done = msg
        elif t == "done":
            done = msg
            terminal = msg
            break
        elif t in ("aborted", "error"):
            terminal = msg
            print(f"[feed] terminal {t}: {msg}", file=sys.stderr, flush=True)
            break
        elif t == "gen_start":
            print(f"[feed] gen_start {msg}", file=sys.stderr, flush=True)
        else:
            print(f"[feed] event {t}", file=sys.stderr, flush=True)
    gen_s = time.time() - t1
    print("[feed] DECODE_PHASE_END", file=sys.stderr, flush=True)
    text = "".join(content)
    if isinstance(done, dict):
        for k in ("text", "content", "output"):
            if isinstance(done.get(k), str) and done.get(k):
                if len(done[k]) >= len(text):
                    text = done[k]
    result = {
        "label": LABEL,
        "text": text,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "prompt": PROMPT,
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
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
        open(RESULT_JSON, "w").write(out + "\n")
    if TEXT_OUT:
        open(TEXT_OUT, "w").write(text)
    try:
        send({"type": "unload"})
        recv(120)
    except Exception as e:
        print(f"[feed] unload err {e}", file=sys.stderr, flush=True)
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
    sys.exit(0 if text else 4)
except Exception as e:
    print(f"[feed] FATAL {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    try: proc.kill()
    except Exception: pass
    if stderr_f: stderr_f.close()
    sys.exit(99)
