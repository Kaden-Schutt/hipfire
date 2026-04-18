#!/usr/bin/env bash
# hermes_validate_run.sh — Stage B (GPU work, runs after current chain drains):
# quantize Carnice-9b, calibrate agentic sidecars on hermes corpus, smoke-test
# hipfire-daemon + OpenAI-compatible HTTP, configure hermes-agent, run a
# small agent task battery.
#
# Prereqs (from Stage A):
#   - bun installed, hipfire daemon binary built, hermes-agent installed
#   - /root/hf_cache/models--kai-os--Carnice-9b/snapshots/... present
#   - /root/hermes_traces_corpus.txt present
#
# Arrivals:
#   - /root/models/carnice-9b.mq4                      Quantized target
#   - /root/models/carnice-9b.mq4.hermes.triattn.bin   Agentic sidecar for Carnice
#   - /root/models/qwen3.6-35b-a3b.mq4.hermes.triattn.bin   Agentic sidecar for A3B
#   - /root/hermes_validate_results/                   Per-task agent-run logs

set -euo pipefail

export PATH=/root/.cargo/bin:/root/.bun/bin:/opt/rocm/bin:/opt/rocm/lib/llvm/bin:$PATH
export HIP_PATH=/opt/rocm
export ROCM_PATH=/opt/rocm
export HIPFIRE_FP16=0

log() { printf '[hermes-run] %s\n' "$*"; }
wait_pid_dead() {
    while pgrep -f "$1" > /dev/null; do sleep 10; done
}

cd /root/hipfire

log "waiting for any running GPU workload..."
wait_pid_dead "triattn_validate|dflash_spec_demo|dflash_train_poc"
log "GPU free"

# ── 1. Quantize Carnice-9b HF bf16 → MQ4 ─────────────────────────────
CARNICE_HF="$(ls -d /root/hf_cache/models--kai-os--Carnice-9b/snapshots/* | head -1)"
CARNICE_MQ4=/root/models/carnice-9b.mq4
if [ ! -f "$CARNICE_MQ4" ]; then
    log "quantizing Carnice-9b: $CARNICE_HF → $CARNICE_MQ4"
    ./target/release/hipfire-quantize \
        --input "$CARNICE_HF" \
        --output "$CARNICE_MQ4" \
        --format q4f16 2>&1 | tail -20
else
    log "Carnice-9b MQ4 already present: $CARNICE_MQ4"
fi
ls -la "$CARNICE_MQ4"

# ── 2. Agentic sidecar cal for Carnice-9b ───────────────────────────
CARNICE_SIDECAR=/root/models/carnice-9b.mq4.hermes.triattn.bin
if [ ! -f "$CARNICE_SIDECAR" ]; then
    log "calibrating Carnice-9b agentic sidecar (1M hermes tokens)"
    ./target/release/examples/triattn_validate "$CARNICE_MQ4" \
        --corpus /root/hermes_traces_corpus.txt \
        --max-tokens 1000000 \
        --sidecar "$CARNICE_SIDECAR" \
        > /root/cal_carnice_hermes.log 2>&1
else
    log "Carnice-9b agentic sidecar already present"
fi
tail -6 /root/cal_carnice_hermes.log 2>&1 || true

# ── 3. Agentic sidecar cal for 3.6-A3B ──────────────────────────────
A3B_MQ4=/root/models/qwen3.6-35b-a3b.mq4
A3B_SIDECAR=/root/models/qwen3.6-35b-a3b.mq4.hermes.triattn.bin
if [ ! -f "$A3B_SIDECAR" ]; then
    log "calibrating 3.6-A3B agentic sidecar (1M hermes tokens)"
    ./target/release/examples/triattn_validate "$A3B_MQ4" \
        --corpus /root/hermes_traces_corpus.txt \
        --max-tokens 1000000 \
        --sidecar "$A3B_SIDECAR" \
        > /root/cal_36a3b_hermes.log 2>&1
else
    log "3.6-A3B agentic sidecar already present"
fi
tail -6 /root/cal_36a3b_hermes.log 2>&1 || true

# ── 4. Smoke-test hipfire serve /v1/chat/completions ─────────────────
log "starting hipfire serve on port 8080..."
cd /root/hipfire/cli
PORT=8080
nohup bun run index.ts serve $PORT > /root/hipfire_serve.log 2>&1 &
SERVE_PID=$!
disown
log "serve PID=$SERVE_PID"

# give it time to load model + start listening
for i in $(seq 1 30); do
    if curl -sf "http://127.0.0.1:$PORT/v1/chat/completions" -X POST \
        -H "Content-Type: application/json" \
        -d '{"model":"carnice-9b","messages":[{"role":"user","content":"Say hi in 5 words"}]}' \
        -o /tmp/serve_smoke.json 2>/dev/null; then
        log "serve is up after ${i}s"
        cat /tmp/serve_smoke.json | head -5
        break
    fi
    sleep 2
done

if ! curl -sf "http://127.0.0.1:$PORT/v1/chat/completions" -X POST \
     -H "Content-Type: application/json" \
     -d '{"model":"carnice-9b","messages":[{"role":"user","content":"ping"}]}' > /dev/null 2>&1; then
    log "WARN: serve not responding after 60s — check /root/hipfire_serve.log"
    tail -30 /root/hipfire_serve.log
fi

# ── 5. Configure hermes-agent to use the hipfire endpoint ───────────
log "configuring hermes-agent..."
# hermes-agent config format: TBD, usually ~/.config/hermes or similar
# Placeholder — the real config may need adjustment once we see hermes CLI:
mkdir -p /root/.config/hermes
cat > /root/.config/hermes/config.json <<EOF
{
  "provider": "custom",
  "endpoint": "http://127.0.0.1:$PORT/v1/chat/completions",
  "model": "carnice-9b",
  "api_key": "dummy"
}
EOF
log "hermes config written to /root/.config/hermes/config.json"
log "(NOTE: format may need adjustment per hermes-agent docs)"

# ── 6. Run agent test battery ────────────────────────────────────────
mkdir -p /root/hermes_validate_results

log "running first agent task — simple 'list files' to confirm wiring"
if command -v hermes >/dev/null 2>&1; then
    timeout 120 hermes run "List the files in /root and show their sizes" \
        > /root/hermes_validate_results/task_list_files.log 2>&1 || \
        log "WARN: first task returned non-zero — see log"
    tail -20 /root/hermes_validate_results/task_list_files.log
else
    log "WARN: hermes binary not in PATH — skipping agent task battery"
fi

log "────────────────────────────────────────────────────"
log "STAGE B COMPLETE"
log "  Carnice-9b MQ4:          $CARNICE_MQ4"
log "  Carnice-9b agentic sc:   $CARNICE_SIDECAR"
log "  3.6-A3B agentic sidecar: $A3B_SIDECAR"
log "  hipfire serve (port):    $PORT (PID $SERVE_PID)"
log "  results:                 /root/hermes_validate_results/"
log ""
log "Next — eyeball results, compare agentic vs wikitext sidecar τ,"
log "       and/or run hermes interactively: 'hermes chat' while serve is up."
log "────────────────────────────────────────────────────"
