#!/bin/bash
set -euo pipefail
E=/mnt/scratch/hipfire-evidence/ds4-mi300x-agentmaxx/a1-m0
W=/mnt/scratch/hipfire-work/ds4-mi300x-agentmaxx
source /root/.cargo/env
export HIP_VISIBLE_DEVICES=0
export HIPFIRE_LOCAL=1
export HIPFIRE_SPECULATION=off
export HIPFIRE_DAEMON_BIN=$W/target/release/examples/daemon
export HIPFIRE_MODEL=/mnt/scratch/models/deepseek-v4-flash-mq2r/deepseek-v4-flash.mq2r
export HIPFIRE_PROMPT="Explain what a GPU kernel is in two sentences."
export HIPFIRE_MAX_TOKENS=64
export HIPFIRE_TEMP=0
export HIPFIRE_LABEL=m0-profile
export HIPFIRE_KERNEL_CACHE=$W/.hipfire_kernels
export HIPFIRE_PING_TIMEOUT=600
export HIPFIRE_LOAD_TIMEOUT=900
export HIPFIRE_GEN_TIMEOUT=1800
cd "$W"

mkdir -p "$E/04-rocprof"
export HIPFIRE_EVENT_LOG=$E/04-profile-events.jsonl
export HIPFIRE_RESULT_JSON=$E/04-profile-result.json
export HIPFIRE_TEXT_OUT=$E/04-profile-decoded.txt
export HIPFIRE_DAEMON_STDERR=$E/04-profile-daemon.stderr
rm -f "$HIPFIRE_EVENT_LOG" "$HIPFIRE_RESULT_JSON" "$HIPFIRE_TEXT_OUT" "$HIPFIRE_DAEMON_STDERR"

# Strategy: rocprofv3 wraps ONLY the daemon. A sibling python feeds it via a fifo.
FIFO=$E/04-daemon.fifo
rm -f "$FIFO"
mkfifo "$FIFO"
OUTF=$E/04-daemon.stdout

# Start feeder that writes commands after seeing pong etc - actually simpler:
# Use a python that opens daemon as subprocess WITHOUT rocprof, but set
# ROCPROF to attach... 
# Direct approach: python feeder runs daemon under `rocprofv3 -- daemon` by
# being the thing that execs via subprocess of rocprofv3.

# Cleanest working approach for rocprofv3 child:
# Run: rocprofv3 ... -- $DAEMON 
# and feed stdin from a python that generates the conversation with enough
# waits, reading daemon stdout from a tee.

# Background: reader/controller
python3 - << "PY" &
import json, os, sys, time, select, hashlib

E = "/mnt/scratch/hipfire-evidence/ds4-mi300x-agentmaxx/a1-m0"
fifo_path = E + "/04-daemon.fifo"
stdout_path = E + "/04-daemon.stdout"
event_log = E + "/04-profile-events.jsonl"
result_json = E + "/04-profile-result.json"
text_out = E + "/04-profile-decoded.txt"
model = os.environ["HIPFIRE_MODEL"]
prompt = os.environ["HIPFIRE_PROMPT"]
max_tokens = int(os.environ["HIPFIRE_MAX_TOKENS"])
temp = float(os.environ["HIPFIRE_TEMP"])
label = os.environ["HIPFIRE_LABEL"]

# Wait for fifo writer side (daemon stdin) - we open WRONLY which blocks until reader
# Actually daemon will open fifo as stdin (read). We open write.
# Also need daemon stdout file - rocprof script will tee stdout to file.

# Open fifo for write (blocks until daemon opens read end)
print("[ctrl] opening fifo for write", flush=True)
fifo = open(fifo_path, "w", buffering=1)

# Wait for stdout file to appear and have content
print("[ctrl] waiting for stdout file", flush=True)
for _ in range(600):
    if os.path.exists(stdout_path) and os.path.getsize(stdout_path) >= 0:
        break
    time.sleep(0.1)

def send(obj):
    line = json.dumps(obj, separators=(",", ":"))
    print(f"[ctrl] >> {line[:400]}", flush=True)
    fifo.write(line + "\n")
    fifo.flush()

def recv(timeout=600.0):
    deadline = time.time() + timeout
    # read new lines from stdout_path
    f = open(stdout_path, "r")
    # seek to end first time? No - start from beginning, track pos
    # Use global position
    if not hasattr(recv, "pos"):
        recv.pos = 0
    while True:
        if time.time() > deadline:
            raise TimeoutError("recv timeout")
        f.seek(recv.pos)
        line = f.readline()
        if not line:
            time.sleep(0.05)
            # check size growth
            continue
        recv.pos = f.tell()
        line = line.strip()
        if not line:
            continue
        # skip non-json (rocprof warnings)
        if not line.startswith("{"):
            print(f"[ctrl] skip nonjson: {line[:200]}", flush=True)
            continue
        with open(event_log, "a") as ef:
            ef.write(line + "\n")
        print(f"[ctrl] << {line[:400]}", flush=True)
        try:
            return json.loads(line)
        except Exception as e:
            print(f"[ctrl] bad json {e}", flush=True)

try:
    # give daemon a moment under rocprof
    time.sleep(2)
    send({"type":"ping"})
    msg = recv(300)
    assert msg.get("type")=="pong", msg
    print("[ctrl] got pong", flush=True)
    params = {"max_seq":4096,"kv_mode":"q8","kv_backend":"contiguous","speculation":"off",
              "dflash_mode":"off","mtp_mode":"off","ngram_draft":False,"dspark_mode":"off"}
    t_load=time.time()
    send({"type":"load","model":model,"params":params})
    loaded=None
    while True:
        msg=recv(900)
        if msg.get("type")=="loaded":
            loaded=msg
            break
        if msg.get("type")=="error":
            raise RuntimeError(msg)
    load_s=time.time()-t_load
    print(f"[ctrl] loaded {load_s:.2f}s", flush=True)
    print("[ctrl] DECODE_PHASE_BEGIN", flush=True)
    t1=time.time()
    req_id=f"g0-{label}"
    send({"type":"generate","id":req_id,"attempt_id":1,"prompt":prompt,"max_tokens":max_tokens,"temperature":temp})
    content=[]
    done=None
    terminal=None
    ntok=0
    while True:
        msg=recv(1800)
        t=msg.get("type")
        if t=="token":
            txt=msg.get("text") or msg.get("token") or ""
            if txt: content.append(txt)
            ntok += 1
        elif t=="commit_ready":
            send({"type":"commit","id":req_id,"attempt_id":1})
            done=msg
        elif t=="done":
            done=msg; terminal=msg; break
        elif t in ("aborted","error"):
            terminal=msg; break
    gen_s=time.time()-t1
    print("[ctrl] DECODE_PHASE_END", flush=True)
    text="".join(content)
    if isinstance(done,dict):
        for k in ("text","content","output"):
            if isinstance(done.get(k),str) and done.get(k) and len(done[k])>=len(text):
                text=done[k]
    result={
        "label":label,"text":text,
        "text_sha256":hashlib.sha256(text.encode()).hexdigest(),
        "prompt":prompt,"prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest(),
        "max_tokens":max_tokens,"temperature":temp,
        "done":done,"terminal":terminal,"loaded":loaded,
        "elapsed_gen_s":gen_s,"elapsed_load_s":load_s,"n_token_events":ntok,
        "daemon_bin":os.environ["HIPFIRE_DAEMON_BIN"],"model":model,
    }
    open(result_json,"w").write(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    open(text_out,"w").write(text)
    print(json.dumps({"ok":True,"gen_s":gen_s,"load_s":load_s,"ntok":ntok,"text_len":len(text)},indent=2), flush=True)
    try:
        send({"type":"unload"})
        recv(120)
    except Exception as e:
        print(f"[ctrl] unload {e}", flush=True)
    fifo.close()
except Exception as e:
    print(f"[ctrl] FATAL {type(e).__name__}: {e}", flush=True)
    try: fifo.close()
    except Exception: pass
    sys.exit(99)
PY
CTRL_PID=$!
echo "ctrl_pid=$CTRL_PID"

# Small delay so ctrl reaches fifo open (blocks until daemon opens)
sleep 0.5

echo "=== launching rocprofv3 around daemon ==="
date -u
# Daemon stdin from fifo; stdout tee to file and also keep for rocprof
# Use bash to connect fifo
rm -f "$OUTF"
touch "$OUTF"

# rocprofv3 runs daemon with stdin=fifo
# Note: hip-trace + kernel-trace + stats
set +e
/usr/bin/time -v rocprofv3 \
  --kernel-trace \
  --stats \
  --summary \
  --summary-units usec \
  -d "$E/04-rocprof" \
  -o m0_decode \
  -f csv json \
  -- "$HIPFIRE_DAEMON_BIN" \
  < "$FIFO" > "$OUTF" 2> "$E/04-profile-daemon.stderr"
PROF_RC=$?
set -e
echo PROFILE_DAEMON_EXIT:$PROF_RC
date -u

# wait for controller
wait $CTRL_PID || echo CTRL_EXIT:$?
echo CTRL_DONE
ls -la "$E/04-rocprof" || true
find "$E/04-rocprof" -type f | head -50
if [ -f "$E/04-profile-result.json" ]; then
  python3 -c "import json;r=json.load(open(/mnt/scratch/hipfire-evidence/ds4-mi300x-agentmaxx/a1-m0/04-profile-result.json));print(gen,r.get(elapsed_gen_s),tok,r.get(n_token_events),sha,r.get(text_sha256));print(r.get(text,)[:200])"
fi
