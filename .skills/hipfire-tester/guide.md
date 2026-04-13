# hipfire-tester: Agent Guide

You are helping a tester bring up hipfire on an RDNA GPU, run the standard
test matrix, and report results back upstream. Works for any agent framework.

## Supported GPUs (v0.1.5 "redline")

| Card | Arch | VRAM | Status | Default KV |
|---|---|---|---|---|
| RX 5700 XT | gfx1010 | 8GB | smoke-test | asym2 |
| BC-250 APU | gfx1013 | 14GB shared | tested | asym2 |
| V620 Pro | gfx1030 | 32GB | tested ✓ | asym3 |
| RX 6700 XT | gfx1031 | 12GB | expected to work | asym3 |
| RX 6600 XT | gfx1032 | 8GB | expected to work | asym2 |
| RX 7900 XTX | gfx1100 | 24GB | **primary target ✓** | asym3 |
| RX 7900 XT | gfx1101 | 16GB | expected | asym3 |
| RX 7800 XT | gfx1102 | 12GB | expected | asym3 |
| Strix Halo | gfx1151 | 16GB APU | expected | asym2 |
| RX 9070 XT | gfx1200/1201 | 16GB (RDNA4) | expected | asym3 |

## Phase 1: Install + verify

```bash
curl -L https://raw.githubusercontent.com/Kaden-Schutt/hipfire/master/scripts/install.sh | bash
hipfire diag
```

`hipfire diag` should report:
- Your arch (e.g. `GPU arch: gfx1030`)
- VRAM free/total
- Pre-compiled kernels count (>= 50 typically)
- HIP probe OK

If `diag` shows issues, chain to the `hipfire-diag` skill for interpretation.

## Phase 2: Pull the reference model

```bash
hipfire pull qwen3.5:4b        # ~2.6 GB — standard test model
```

Other sizes:

```bash
hipfire pull qwen3.5:0.8b       # 0.55 GB  — tiny card
hipfire pull qwen3.5:9b         # 5.3 GB   — 8GB cards OK
hipfire pull qwen3.5:27b        # 15 GB    — needs 16GB+ VRAM
```

MQ6 variants for higher-quality tests:

```bash
hipfire pull qwen3.5:4b-mq6     # 3.5 GB
hipfire pull qwen3.5:9b-mq6     # 7.3 GB
```

## Phase 3: Single-run sanity

```bash
hipfire run qwen3.5:4b "Explain WMMA in one paragraph."
```

Expected: a coherent paragraph ending with tok/s stats. First run incurs
kernel JIT cost (2-5 min on slow hardware, cached at
`/tmp/hipfire_kernels/`).

## Phase 4: Multi-turn recall test

This is the key v0.1.5 regression gate — ensures asym3 K kernel head_dim=256
fix is intact:

```bash
hipfire stop 2>/dev/null
hipfire serve -d
sleep 30   # wait for warmup
curl -s http://localhost:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5:9b","messages":[
    {"role":"user","content":"My name is Kaden."},
    {"role":"assistant","content":"Hello Kaden!"},
    {"role":"user","content":"What is my name? One word."}
  ],"max_tokens":50,"temperature":0}' \
  | python3 -c "import json,sys; r=json.load(sys.stdin); print(repr(r['choices'][0]['message']['content'].strip()))"
hipfire stop
```

**Expected:** `'Kaden'` (after think-block strip) — exact word or with minor
punctuation. Anything else is a regression.

## Phase 5: Prefill + decode sweep

```bash
for size in 0.8b 4b 9b 27b; do
  m=~/.hipfire/models/qwen3.5-${size}.mq4
  [ -f "$m" ] || { echo "$size: MISSING"; continue; }
  for pp in 32 128 512 2048; do
    r=$(HIPFIRE_KV_MODE=asym3 ~/.hipfire/bin/bench_qwen35_mq4 "$m" \
        --prefill $pp --gen 30 --warmup 10 2>&1 | grep -m1 "^SUMMARY")
    p=$(echo "$r" | sed -nE 's/.*prefill_tok_s=([0-9.]+).*/\1/p')
    d=$(echo "$r" | sed -nE 's/.*gen_tok_s=([0-9.]+).*/\1/p')
    bw=$(echo "$r" | sed -nE 's/.*bw_gib_s=([0-9.]+).*/\1/p')
    printf "%-5s pp%-5s prefill=%7s tok/s   decode=%6s tok/s  (%s GiB/s)\n" \
           "$size" "$pp" "$p" "$d" "$bw"
  done
done
```

Report the resulting table — upstream uses these to update
`docs/BENCHMARKS.md`.

## Phase 6: CLI surface smoke

```bash
hipfire config list          # should show kv_cache, flash_mode, thinking, etc.
hipfire config               # TUI — arrow keys should navigate
hipfire ps                   # running daemons
hipfire list                 # local models
```

If TUI doesn't render, report terminal + $TERM env var.

## Phase 7: Quantize smoke (optional)

```bash
hipfire quantize Qwen/Qwen3.5-0.8B --format mq4 -o /tmp/test.mq4
hipfire run /tmp/test.mq4 "Hi"
```

Should produce coherent output. Fresh quant → no cached kernels → slow first
pass is normal.

## Reporting results

Submit via GitHub issue or PR with:

```markdown
## Tester report — {your GPU, gfx arch, VRAM}

- hipfire version: `hipfire --version` → (paste)
- diag output: (paste `hipfire diag`)
- multi-turn Kaden: {pass/fail + actual answer}
- bench sweep: (paste Phase 5 output)
- CLI surface smoke: {all OK / list failures}
- notes: any unusual behavior, hangs, etc.
```

## Quantization formats (v0.1.5)

| Format | B/weight | Speed | Quality | Use case |
|---|---|---|---|---|
| `mq4` | 0.53 | Fastest | ~Q8 on quality gate | **Default** |
| `mq6` | 0.78 | Fast | Near-FP16 | Quality-critical |
| `hfq4` | 0.53 | Fastest | Good | Legacy — loads but not produced |
| `hfq6` | 0.78 | Fast | Better | Legacy |
| `q8` | 1.06 | Moderate | Reference | Correctness debug |

MQ4 is the production default — FWHT-rotated 4-bit with a byte-exact quality
gate on every change. HF4/HF6 are legacy formats that still load but aren't
freshly produced.

## Known quirks

- **BC-250:** daemon HTTP multi-turn hangs on specific request sequences.
  Workaround: `pkill -9 daemon bun` between tests. See
  `.skills/hipfire-autoheal/known-issues.md`.
- **0.8B + hipGraph:** don't set `HIPFIRE_GRAPH=1` for 0.8B — known panic.
  Other sizes are fine.
- **Qwen 3 on asym:** use `HIPFIRE_KV_MODE=q8` for non-Qwen-3.5 models;
  asym modes are flash-only and Qwen 3 falls back to per-token otherwise.

## If stuck, chain to `hipfire-autoheal`

Runtime issues (hangs, kernel errors, port conflicts) are the autoheal
skill's wheelhouse. This skill is for bring-up + the standard test matrix.
