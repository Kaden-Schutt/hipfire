#!/usr/bin/env python3
"""Minimal hipfire daemon JSONL driver for DS4 coherence smoke.
Reads config from env; writes full event log to stderr, decoded text to stdout.
"""
import json, os, sys, time, subprocess, select, signal

DAEMON = os.environ["HIPFIRE_DAEMON_BIN"]
MODEL = os.environ["HIPFIRE_MODEL"]
PROMPT = os.environ.get("HIPFIRE_PROMPT", "Explain what a GPU kernel is in two sentences.")
MAX_TOKENS = int(os.environ.get("HIPFIRE_MAX_TOKENS", "32"))
TEMP = float(os.environ.get("HIPFIRE_TEMP", "0"))
LABEL = os.environ.get("HIPFIRE_LABEL", "run")
EVENT_LOG = os.environ.get("HIPFIRE_EVENT_LOG", "")

env = os.environ.copy()
# pass through HIP_* and HIPFIRE_* already in env

cmd = [DAEMON]
print(f"[driver] spawning {DAEMON}", file=sys.stderr, flush=True)
proc = subprocess.Popen(
    cmd,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    env=env,
    text=True,
    bufsize=1,
)

def send(obj):
    line = json.dumps(obj, separators=(",", ":"))
    print(f"[driver] >> {line}", file=sys.stderr, flush=True)
    proc.stdin.write(line + "\n")
    proc.stdin.flush()

def recv(timeout=3600.0):
    # multiplex stdout+stderr
    deadline = time.time() + timeout
    buf_out = ""
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            raise TimeoutError("recv timeout")
        rlist = [proc.stdout, proc.stderr]
        ready, _, _ = select.select(rlist, [], [], min(remaining, 1.0))
        if proc.poll() is not None and not ready:
            # drain
            err = proc.stderr.read() if proc.stderr else ""
            out = proc.stdout.read() if proc.stdout else ""
            if err:
                print(err, file=sys.stderr, end="", flush=True)
            if out:
                for line in out.splitlines():
                    line=line.strip()
                    if not line: continue
                    try:
                        return json.loads(line)
                    except Exception:
                        print(f"[driver] non-json stdout: {line}", file=sys.stderr, flush=True)
            raise RuntimeError(f"daemon exited rc={proc.returncode}")
        for fd in ready:
            chunk = fd.readline()
            if not chunk:
                continue
            if fd is proc.stderr:
                print(chunk, file=sys.stderr, end="", flush=True)
                continue
            # stdout
            line = chunk.strip()
            if not line:
                continue
            if EVENT_LOG:
                with open(EVENT_LOG, "a") as f:
                    f.write(line + "\n")
            print(f"[driver] << {line[:500]}", file=sys.stderr, flush=True)
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
            print(json.dumps({"error": msg}, indent=2))
            sys.exit(2)
        # progress events ok
        print(f"[driver] load-event {t}", file=sys.stderr, flush=True)
    print(f"[driver] LOADED in {time.time()-t0:.1f}s: {json.dumps(loaded)[:500]}", file=sys.stderr, flush=True)

    attempt_id = 1
    send({
        "type": "generate",
        "id": "g0smoke",
        "attempt_id": attempt_id,
        "prompt": PROMPT,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMP,
    })
    content = []
    done = None
    t1 = time.time()
    while True:
        msg = recv(3600)
        t = msg.get("type")
        if t == "token":
            text = msg.get("text") or ""
            content.append(text)
            # also accept "committed" style
        elif t == "committed":
            text = msg.get("text") or ""
            if text:
                content.append(text)
        elif t == "done" or t == "commit_ready":
            done = msg
            # if commit_ready, may need commit
            if t == "commit_ready":
                send({"type": "commit", "id": "g0smoke", "attempt_id": attempt_id})
                # wait for done
                continue
            break
        elif t == "error":
            print(json.dumps({"error": msg, "partial": "".join(content)}, indent=2))
            sys.exit(3)
        elif t == "gen_start":
            print(f"[driver] gen_start {msg}", file=sys.stderr, flush=True)
        else:
            print(f"[driver] event {t}", file=sys.stderr, flush=True)
    text = "".join(content)
    result = {
        "label": LABEL,
        "text": text,
        "done": done,
        "loaded": {k: loaded.get(k) for k in ("arch","dim","layers","vocab","retry_reset_eligible","kv_mode") if k in loaded} if loaded else None,
        "elapsed_gen_s": time.time() - t1,
        "elapsed_load_s": t1 - t0,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    try:
        send({"type": "unload"})
        recv(60)
    except Exception as e:
        print(f"[driver] unload err {e}", file=sys.stderr, flush=True)
    proc.stdin.close()
    try:
        proc.wait(timeout=30)
    except Exception:
        proc.kill()
    sys.exit(0)
except Exception as e:
    print(f"[driver] FATAL {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    try:
        proc.kill()
    except Exception:
        pass
    sys.exit(99)
