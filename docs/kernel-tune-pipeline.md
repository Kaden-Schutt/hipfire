# Kernel Tuning Pipeline

Automated loop-engineering pipeline for hipfire GPU kernel optimization.
Built during the gfx1100 batch-tiled B=2 WMMA campaign (PR #611) and
validated on Qwen3.5-4B (+14% prefill, p<0.01) and Qwen3.8-27B (+29%
prefill, p<0.01).

## Overview

Three scripts in `scripts/`, plus a state file and JSONL ledger under
`.codeinsight+research/kernel-tune/`. The pipeline encodes the
hipfire-kernel-tuning skill methodology (profile → root-cause → one
lever → implement → correctness → fresh-process measure → decide/log)
as shell commands an agent drives between kernel edits.

```
kernel-tune-loop.sh      Core 6-phase loop driver (baseline → profile → validate → measure → decide)
deep_ab_bt2.sh           Rigorous statistical A/B validation (100 samples/arm, Welch t-test)
deep_ab_isolation.sh     Per-kernel marginal contribution (4 arms, incremental enablement)
```

## 1. `kernel-tune-loop.sh` — core loop driver

A 6-phase state machine. State persists in `loop_state.json`; every
phase appends to `ledger.jsonl`. The agent runs one command per phase,
edits the kernel between phases 1 and 4, and the script handles all
measurement and logging.

```
baseline  →  profile  →  [edit kernel]  →  validate  →  measure  →  decide
```

### Commands

| Command | What it does |
|---|---|
| `baseline` | Records identity hashes (model md5, binary md5, daemon md5, git commit, prompt md5). Runs `hipfire bench` + `bench_qwen35_mq4` with `HIPFIRE_PROFILE=1`. Runs `hipfire profile` for kernel inventory (occupancy, VGPR, LDS). Runs `test_kernels` for correctness baseline. Saves to `runs/baseline-<timestamp>/`. Writes first ledger entry. |
| `profile` | Starts a new iteration. Runs profiled bench, parses the PROFILE section to rank kernels by total time (calls × per-call µs). Shows low-occupancy/high-VGPR kernels from `hipfire profile`. Runs Atlas `collect-ar` + `suggest` for ISA-level analysis. Saves to `runs/iter-N-<timestamp>/`. Increments iteration counter. |
| `validate` | Runs `test_kernels` (correctness gate — fails hard if any kernel breaks). Runs `serve_harness.py battery` for model-level coherence. Runs a quick `hipfire bench` for eyeball check. Sets `correctness_pass` or `correctness_fail` in state. |
| `measure` | Fresh-process A/B: runs `hipfire bench` with the candidate binary, compares against the saved baseline. Computes decode/prefill deltas. Classifies as win / regression / noise. |
| `decide <disposition> [notes]` | Writes the final ledger entry with disposition (`win` / `reject` / `park` / `regression`), notes, all identity hashes, baseline vs candidate numbers. Increments iteration counter for the next loop. |
| `status` | Shows current phase, iteration number, baseline/candidate numbers, and recent ledger entries. |

### What the agent does between commands

After `profile` shows the hot kernels:

1. Read the kernel source (`.hip` file)
2. Identify one lever (consult `.agents/skills/hipfire-kernel-tuning/levers.md`)
3. Edit the `.hip` file
4. Rebuild: `cargo build --release` + examples
5. Run `validate` → `measure` → `decide`
6. Repeat from `profile` for the next kernel

### Configuration (env vars)

| Env var | Default | Purpose |
|---|---|---|
| `HIPFIRE_MODEL` | `~/.hipfire/models/qwen3.5-4b.mq4` | Model path |
| `HIPFIRE_KV_MODE` | `q8` | KV cache mode |
| `HIPFIRE_ARCH` | auto-detect | Target arch |
| `HIPFIRE_BENCH_RUNS` | `5` | Bench runs |
| `HIPFIRE_BENCH_WARMUPS` | `3` | Bench warmups |
| `HIPFIRE_BENCH_MAX_TOKENS` | `128` | Max tokens per run |
| `HIPFIRE_BENCH_BACKEND` | `noslots` | Bench backend |
| `HIPFIRE_BENCH_WORKLOAD` | `stateless` | Bench workload |
| `HIPFIRE_PROMPT_FILE` | `benchmarks/prompts/bare_factual.txt` | Prompt file |

## 2. `deep_ab_bt2.sh` — rigorous statistical A/B

Used after the loop identifies a winning lever and you need publishable
evidence. Not part of the per-iteration loop — it is the final
validation step before a PR.

### Methodology

- **100 samples per arm** (5 sessions × 20 prefill runs)
- **Alternating A/B/A/B** session order to control for thermal/DPM drift
- **Same binary, env-var toggle**: `HIPFIRE_BT2_DISABLE=1` (baseline) vs
  `0` (bt2) — no recompilation between arms
- **Noise controls**: `HIPFIRE_VERIFY_GRAPH=0` (tighter stdev),
  `HIPFIRE_DPM_WARMUP_SECS=20` (full thermal settlement), 20 warmup runs
  discarded
- **Records all individual samples** to JSON, not just medians
- **Built-in statistical analysis**: Welch's t-test (manual computation
  with Welch-Satterthwaite df), Cohen's d effect size, 95% CI for the
  delta, significance classification (p<0.01 / p<0.05 / p<0.10 / n.s.)

### Usage

```bash
bash scripts/deep_ab_bt2.sh <model_path> <output_json>
```

## 3. `deep_ab_isolation.sh` — per-kernel marginal contribution

Measures how much each kernel variant contributes independently. Uses 4
arms instead of 2:

| Arm | Configuration |
|---|---|
| A | All plain WMMA (`HIPFIRE_BT2_DISABLE=1`) |
| B | gate_up bt2 only (`HIPFIRE_BT2_DISABLE=1 HIPFIRE_GATE_UP_VARIANT=bt2`) |
| C | gate_up + qkvza bt2 (`+ HIPFIRE_QKVZA_BT2_FORCE=1`) |
| D | All bt2 (`HIPFIRE_BT2_DISABLE=0`) |

Alternating A/B/A/C/A/D order, 3 sessions × 20 runs = 60 samples per
arm. Reports the incremental delta at each step with significance.

## Artifacts

```
.codeinsight+research/kernel-tune/
├── loop_state.json              # Current phase, iteration, baseline numbers
├── ledger/
│   └── ledger.jsonl             # One line per phase/decision — the experiment log
├── runs/
│   ├── baseline-<timestamp>/
│   │   ├── identity.json        # Model/binary/daemon md5, git commit, prompt md5
│   │   ├── bench_baseline.json  # hipfire bench --json output
│   │   ├── bench_profiled.log   # Per-kernel profile (HIPFIRE_PROFILE=1)
│   │   ├── kernel_profile.json  # hipfire profile --json (occupancy, VGPR, etc.)
│   │   └── test_kernels.log
│   └── iter-N-<timestamp>/
│       ├── identity.json + identity_candidate.json
│       ├── bench_profiled.log   # Profile after kernel edit
│       ├── atlas-raw.jsonl      # Atlas measurement rows
│       ├── isa.json             # ISA disassembly
│       ├── dispatch.json        # Dispatch provenance
│       ├── bench_candidate.json # Fresh-process measurement
│       └── test_kernels_candidate.log
├── deep_ab_results.json         # 4B deep A/B (100 samples/arm)
├── deep_ab_isolation.json       # 4B per-kernel isolation (60 samples/arm)
└── deep_ab_qwen38_27b.json      # 27B deep A/B (80+60 samples)
```

## How to run a new campaign

### Prerequisites

```bash
cargo build --release
cargo build --release --features deltanet --example test_kernels -p hipfire-runtime
cargo build --release --features deltanet --example bench_qwen35_mq4 -p hipfire-runtime
```

### Per-iteration loop

```bash
# 1. Establish baseline (records identity, profiles hot kernels)
HIPFIRE_MODEL=~/.hipfire/models/qwen3.5-4b.mq4 HIPFIRE_KV_MODE=q8 \
    ./scripts/kernel-tune-loop.sh baseline

# 2. Profile to identify the hottest kernel
./scripts/kernel-tune-loop.sh profile

# 3. Read the profile output, pick ONE lever, edit the kernel .hip file
#    (read .agents/skills/hipfire-kernel-tuning/levers.md for lever ideas)

# 4. Rebuild
cargo build --release --features deltanet --example test_kernels -p hipfire-runtime
cargo build --release --features deltanet --example bench_qwen35_mq4 -p hipfire-runtime

# 5. Validate correctness
./scripts/kernel-tune-loop.sh validate

# 6. Measure (fresh-process A/B vs baseline)
./scripts/kernel-tune-loop.sh measure

# 7. Decide and log
./scripts/kernel-tune-loop.sh decide win "batch-tiled B=2 gate_up, +22% per-kernel"

# 8. Repeat from step 2 for the next kernel
./scripts/kernel-tune-loop.sh profile
```

### Final validation before a PR

```bash
# Deep A/B (100 samples, Welch t-test, Cohen's d, 95% CI)
bash scripts/deep_ab_bt2.sh ~/.hipfire/models/qwen3.5-4b.mq4 \
    .codeinsight+research/kernel-tune/deep_ab_results.json

# Per-kernel isolation (which kernel contributed what)
bash scripts/deep_ab_isolation.sh

# Cross-model validation (does the gain hold on a larger model?)
bash scripts/deep_ab_bt2.sh ~/models/hipfire/Qwen/Qwen3.8-27B-MQ4/qwen3.8-27b.mq4 \
    .codeinsight+research/kernel-tune/deep_ab_qwen38_27b.json
```

## Design notes for future agents

1. **The loop script is model-agnostic** — point `HIPFIRE_MODEL` at any
   `.mq4` model. The 27B validation used the same script with a
   different path.

2. **The deep A/B harness requires a kill-switch env var** —
   `HIPFIRE_BT2_DISABLE` is specific to this campaign's bt2 variants.
   For a different optimization, add a corresponding env var to
   `FeatureFlags` and the dispatch sites, then update the harness to
   toggle it. The pattern: same binary, env-var flip, no recompilation
   between arms.

3. **The ledger is the experiment log** — each entry is a JSON line
   with identity hashes, disposition, and notes. It is the audit trail
   for what was tried, what worked, what was rejected, and why.

4. **One lever per iteration** — the methodology forbids bundling
   changes. Each iteration targets one kernel with one lever. The
   isolation harness then decomposes the combined win into per-kernel
   contributions.

5. **`bench_qwen35_mq4` is the per-kernel profiling tool** — `hipfire
   bench` uses batch_size=1 (single-sequence), while `bench_qwen35_mq4
   --prefill 32` uses batch_size=32 where batched kernels like bt2 are
   active. Use the right tool for the path you are optimizing.

6. **Noise discipline** — always run in a fresh process, record model
   md5 and both binary md5s, use `HIPFIRE_VERIFY_GRAPH=0` for tighter
   stdev, and let `HIPFIRE_DPM_WARMUP_SECS=20` settle thermal state.
   Within-session A/B is noisy on gfx1100 (±10–15% from DPM/thermal).

## Provenance

- **Built**: 2026-08-20 during the gfx1100 bt2 WMMA campaign
- **Campaign result**: +14% prefill on 4B, +29% on 27B, p<0.01, zero
  decode regression
- **PR**: [warpfront/hipfire#611](https://github.com/warpfront/hipfire/pull/611)
- **Base commit**: `80a572c8` (warpfront/master)
- **GPU**: AMD RX 7900 XTX (gfx1100, 96 CUs, 24GB)
