# Unified Benchmark Suite

`cli/bench_sweep.ts` is the canonical daemon-driven bench harness for
hipfire. This document describes what the suite measures, why it
measures it this way, the hashed-JSON output schema (the perf-ledger
data source), and a migration table mapping retired in-process
microbenches to their suite equivalents.

Cross-reference: `docs/methodology/perf-benchmarking.md` covers the
measurement methodology (warmup protocol, noise band, DPM discipline,
prompt-md5 rule, DFlash coherence gates). This document is the
implementation spec; that one is the why-behind-the-numbers reference.
Both documents describe one system.

---

## Why a daemon-driven suite

The root lesson that drove this design: **the same code measured three
different ways on the same hardware in the same session produced three
different numbers — 101, 109, and 126.6 tok/s — for A3B AR-decode.**
Those three readings came from:

1. `bench_qwen35_mq4` (in-process, no hipGraph on the decode loop):
   **101 tok/s**
2. `bench_qwen35_mq4` with `HIPFIRE_GRAPH=1` set in the caller's env
   (the env var is the *daemon's* decode-graph toggle; this binary
   ignores it for its decode phase): **109 tok/s** — a ~8% placebo
   from DPM ramp variance, not from hipGraph
3. Daemon with `HIPFIRE_AR_GRAPH=1` and the full continuous-warmup
   protocol in `bench_sweep.ts`: **126.6 tok/s**

The 20% gap between (1) and (3) is not measurement noise — it is the
**daemon's AR hipGraph path**, which wraps the decode loop in a
compiled HIP graph and replays it without per-token kernel-launch
overhead. The in-process bench binary never enters that path regardless
of what env vars are set, so it measures a structurally different
execution path than production.

Consequences:

- Any perf claim that uses `bench_qwen35_mq4` AR-decode numbers to
  represent production throughput is systematically ~20% low on MoE
  models. The underread is not recoverable by warmup or flag tuning.
- Cross-commit A/B using `probe_commits.sh` measures a consistent
  lower bound (in-process, no hipGraph), which is fine for bisect
  purposes but must not be cited as production throughput numbers.
- The production number is what the daemon's `done` event reports:
  `decode_tok_s` from steady-state decode after hipGraph is live.

`bench_sweep.ts` closes this gap by driving the actual daemon binary
and reading the `done` event's authoritative perf fields.

---

## Suite overview

```
cli/bench_sweep.ts <model_path> [max_seq] [pp_csv] [gen_tokens] [prompt...]
```

Three logical subcommands are handled in one pass:

| Subcommand | What it measures | Output field(s) |
|---|---|---|
| **prefill-sweep** | Prefill tok/s at N configurable pp lengths (median of 3) | `pp: {128: X, 256: X, ...}` |
| **ar-decode** | AR-decode tok/s (daemon `decode_tok_s` from `done` event) | `decode_tok_s` |
| **natural-prefill** | Prefill tok/s from the actual generate prompt (daemon `prefill_tok_s`) | `prefill_natural_tok_s` |

DFlash τ is not measured in `bench_sweep.ts`. Use `dflash_spec_demo`
with `--prompts-file` for τ measurement; the resident-bench mode
(described in `perf-benchmarking.md`) is the correct protocol for
multi-row DFlash sweeps.

---

## Canonical methodology baked in

### 1. Daemon-driven with `HIPFIRE_AR_GRAPH=1`

Callers are expected to set `HIPFIRE_AR_GRAPH=1` in the environment
before invoking `bench_sweep.ts`. The script inherits the full process
env and passes it to the spawned daemon via `{ ...process.env }`. This
means the daemon's decode loop enters the hipGraph capture+replay path,
which is the production execution path on all RDNA3+ targets.

Why it matters: the hipGraph path eliminates per-token kernel-launch
overhead. On A3B this is worth ~20 tok/s (101 → 126.6); the in-process
`bench_qwen35_mq4` misses this entirely regardless of env settings
because it does not go through the daemon's `generate` pipeline.

If `HIPFIRE_AR_GRAPH` is not set by the caller, the daemon will use
whatever its compiled default is. For perf ledger entries, explicitly
set `HIPFIRE_AR_GRAPH=1` so the result is unambiguous.

### 2. Continuous warmup — the two-phase protocol

The script implements a two-phase warmup before the measured pass:

**Phase 1 — load-time DPM ramp** (`HIPFIRE_DPM_WARMUP_SECS=10`):
The daemon runs `gpu.dpm_warmup(10)` after weight upload but *before*
emitting the `loaded` ack. This pins the GPU to high DPM state before
the first forward pass. The default is 10s, matching the established
protocol in `bench_qwen35_mq4` and `dflash_spec_demo`.

Why it matters: without DPM warmup the first prefill run on a cold
GPU is 3-7x slower than steady state. This is not "noise" — it is
measurement error. The warmup-before-ack contract means `loaded`
guarantees DPM is live when the bench begins.

**Phase 2 — throwaway forward pass** (kernel JIT + clock re-ramp):
After receiving `loaded`, the script immediately sends one `bench_prefill`
per pp shape plus one full `generate` (max=192). This throwaway pass
does two things that Phase 1 cannot:
- JITs all prefill and decode kernels (kernel compilation leaves the
  GPU idle long enough for clocks to drop after Phase 1)
- Re-pins DPM after the JIT-induced idle

The measured pass runs *immediately* and *continuously* after Phase 2,
so DPM stays high throughout. This is the most complete warmup of any
harness in the tree.

Why both phases are needed: Phase 1 alone does not warm the kernel JIT
cache. Phase 2 alone (without Phase 1) is insufficient on a cold daemon
because the first JIT compilation stalls long enough to drop DPM below
the measurement window. Both phases together produce stable, reproducible
numbers.

### 3. `--prompt-file` discipline and prompt md5 pinning

Current state: `bench_sweep.ts` accepts an inline prompt string as
trailing argv. This violates the md5-pinning rule (see
`perf-benchmarking.md` §"Prompt structure matters").

**Required augmentation before any cross-session or cross-agent perf
comparison:** pass a committed `.txt` fixture file as the prompt source
and record the prompt md5 in the output JSON. The inline-string mode
is acceptable for local exploratory benches but must not produce
entries in the perf ledger.

Why it matters: one newline character can swing DFlash τ by 17% on 27B.
AR-decode is less sensitive to prompt whitespace than DFlash, but
pp-sweep numbers depend on prompt token count (which varies with
whitespace on some tokenizers). Byte-identical prompts are the only
reliable cross-session comparator.

Planned field in hashed JSON output: `prompt_md5: "<md5hex>"`.

### 4. Model, prompt, and binary hashes in output

The output JSON records `arch`, `kv`, `max_seq`, and `model` (path
string) today. For perf ledger entries the following fields must be
present before a result is treated as a signed record:

| Field | Source | Status |
|---|---|---|
| `model_path` | argv | present |
| `arch` | daemon `loaded.arch` | present |
| `kv` | `HIPFIRE_BENCH_KV` env / default `q8` | present |
| `max_seq` | argv / default 9216 | present |
| `pp` | bench_prefill results | present |
| `decode_tok_s` | `done.decode_tok_s` | present |
| `prefill_natural_tok_s` | `done.prefill_tok_s` | present |
| `prompt_md5` | md5 of prompt bytes | **not yet emitted — required** |
| `model_md5` | md5 of model file | **not yet emitted — required** |
| `binary_md5` | md5 of daemon binary | **not yet emitted — required** |
| `hipfire_ar_graph` | `HIPFIRE_AR_GRAPH` env at launch | **not yet emitted — required** |
| `timestamp_utc` | wall clock at run start | **not yet emitted — recommended** |

The hash basis is **md5** for model, prompt, and binary — the exact same
convention the perf ledger requires (`perf-arch-discipline.md` §"Hash
pinning requirements"). One hash algorithm across producer and consumer;
do not emit sha256 here and md5 there. The ledger derives its `bench_date`
from `timestamp_utc` and folds `hipfire_ar_graph` (plus the KV mode) into
its `hipfire_flags` string.

Until the required fields are present, results from `bench_sweep.ts`
should be treated as local exploratory numbers, not perf ledger entries.
The perf ledger (cross-reference `perf-arch-discipline.md`) requires
signed records with full provenance.

---

## Hashed-JSON output schema

`bench_sweep.ts` writes one JSON line to stdout on completion (or on
any fatal error, where `error` is set and numeric fields are null):

```jsonc
{
  "model": "/path/to/model.hfq",
  "arch": "gfx1100",
  "kv": "q8",
  "max_seq": 9216,
  "pp": {
    "128":  1140.5,
    "256":  1082.3,
    "512":   921.7,
    "1024":  743.2,
    "4096":  312.1,
    "8192":  null    // null when pp + 32 > max_seq
  },
  "decode_tok_s":         126.6,
  "prefill_natural_tok_s": 1138.2,

  // --- fields to be added for perf-ledger entries (md5 basis, matching
  //     perf-arch-discipline.md §"Hash pinning requirements") ---
  "prompt_md5":      "<md5hex of prompt bytes>",
  "hipfire_ar_graph": 1,
  "model_md5":       "<md5hex of model file>",
  "binary_md5":      "<md5hex of daemon binary>",
  "timestamp_utc":   "2026-06-12T14:33:07Z"
}
```

Error case:

```jsonc
{ "model": "/path/to/model.hfq", "error": "daemon EOF" }
```

The `pp` field uses the pp length as a string key (JSON maps require
string keys) and the value is the **median of 3 successive
`bench_prefill` calls**, rounded to 1 decimal place. A null value
means the pp shape was skipped because `pp + 32 > max_seq`.

`decode_tok_s` comes from the daemon's `done` event field
`decode_tok_s` (steady-state decode, post-prefill). If the daemon emits
only the legacy `tok_s` field, that is used as a fallback.

`prefill_natural_tok_s` comes from `done.prefill_tok_s`. This is the
prefill speed for the actual prompt passed to `generate`, measured
inside the daemon's forward-pass timer. It differs from the
`pp`-sweep numbers because the natural prompt is a real string (not
a synthetic token sequence) and may be shorter than the sweep lengths.

---

## Known gaps and planned work

**Gap 1: `HIPFIRE_AR_GRAPH=1` not forced by the script**

The script inherits env from the caller. A caller that forgets
`HIPFIRE_AR_GRAPH=1` will measure the non-hipGraph path and produce
a ~20% lower decode number. Planned fix: emit a warning on stderr
if `HIPFIRE_AR_GRAPH` is not set, or force it unconditionally and
document that `HIPFIRE_AR_GRAPH=0` must be explicit to opt out.

**Gap 2: decode measured from a single generate call**

The measured pass runs one `generate` call and reads its `done` event.
A single call is sufficient given the continuous-warmup protocol (DPM
is pinned, JIT is warm), but it means the decode number has no
confidence interval. Planned fix: run best-of-N (N=3) and report
median + stddev, matching the prefill-sweep protocol.

**Gap 3: no DFlash τ subcommand**

`bench_sweep.ts` does not measure spec-decode acceptance rate. DFlash
τ is prompt-structure-sensitive in ways that AR-decode is not (17% τ
swing from one newline on 27B), so it needs a different prompting
discipline and a different output schema. The planned `dflash-tau`
subcommand would wrap `dflash_spec_demo --prompts-file` with the same
md5-pinning and hashed-JSON discipline. Until then, use `dflash_spec_demo`
directly per `perf-benchmarking.md` §"DFlash speed gate".

---

## Migration table

The following in-process microbenches are retired FROM THE BENCH PATH
by `bench_sweep.ts`. "Retired from the bench path" means: do not use
them to produce numbers for perf ledger entries, commit message perf
claims, or cross-session comparisons. Their other uses (correctness
testing, kernel profiling, rocprof attribution) are explicitly kept.

| Retired microbench | Use instead | Reason for retirement |
|---|---|---|
| `bench_qwen35_mq4` (AR-decode) | `bench_sweep.ts` ar-decode | Misses daemon AR hipGraph; systematically ~20% low on MoE models |
| `bench_qwen35_mq4` (prefill-sweep) | `bench_sweep.ts` prefill-sweep | Acceptable alternative, but daemon pp-sweep is canonical for ledger entries |
| `dflash_spec_demo --prompt <string>` (τ bench) | `dflash_spec_demo --prompts-file <fixture.jsonl>` | Inline prompt string not byte-stable across editors/sessions; violates md5-pinning rule |
| `scripts/probe_commits.sh` (AR tok/s) | `bench_sweep.ts` + git bisect | `probe_commits.sh` uses `bench_qwen35_mq4` with `--gen 30 --warmup 3`; shallow warmup + no daemon hipGraph = consistent lower bound for bisect, NOT production throughput |
| `scripts/speed-gate.sh` bench_run (AR tok/s claim) | `bench_sweep.ts` ar-decode | speed-gate.sh calls bench_qwen35_mq4; the gate's pass/fail threshold is calibrated to the in-process lower bound, which is correct for regression detection but not for production perf claims |

**Explicitly NOT retired:**

- `infer_qwen35` — interactive inference path; coherence testing; other
  uses unrelated to the bench path. Do not delete.
- `bench_qwen35_mq4` for kernel profiling with `HIPFIRE_PROFILE=1` /
  `HIPFIRE_PROFILE_DECODE=1` — rocprof attribution workflow requires
  in-process dispatch; the daemon's process boundary is a blocker for
  per-kernel profiling. Keep; use for what it's good at.
- `dflash_spec_demo` for τ measurement — retire only the
  `--prompt <string>` single-row invocation pattern. The binary itself
  and its `--prompts-file` resident-bench mode are the canonical DFlash
  τ harness.
- `probe_commits.sh` for bisect — the consistent lower bound it
  measures is exactly what bisect needs (monotonic, reproducible across
  commits). Keep; do not cite its output as production throughput.
- `scripts/speed-gate.sh` as a regression gate — the gate's baselines
  are calibrated to the in-process lower bound, which is the right
  reference for a gate that fires on `cargo commit`. The gate's job is
  regression detection, not production perf reporting.

---

## Quick-start

```bash
# Prerequisites:
#   - daemon binary built: cargo build --release -p hipfire-runtime --example daemon
#   - bun installed: https://bun.sh

export HIPFIRE_DAEMON_BIN=./target/release/examples/daemon
export HIPFIRE_AR_GRAPH=1
export HIPFIRE_DPM_WARMUP_SECS=10

# Default pp sweep (128,256,512,1024,4096,8192) + decode:
bun cli/bench_sweep.ts ~/.hipfire/models/qwen3.5-9b.mq4

# Custom pp lengths:
bun cli/bench_sweep.ts ~/.hipfire/models/qwen3.5-27b.mq4 9216 256,1024,4096

# Multiple SKUs (one JSON line per model):
for m in ~/.hipfire/models/qwen3.5-{9b,27b}.mq4; do
  bun cli/bench_sweep.ts "$m"
done

# Redirect output for downstream tools:
bun cli/bench_sweep.ts model.mq4 2>/dev/null | jq '.decode_tok_s'
```

Daemon build command:

```bash
cargo build --release -p hipfire-runtime --example daemon --features deltanet
```

---

## Relationship to perf-arch-discipline.md

`bench_sweep.ts` is the data source for the perf ledger described in
`perf-arch-discipline.md`. The ledger ingests hashed-JSON records; the
suite produces them. The pairing is:

- Suite produces: one JSON line per model × (arch, kv, max_seq, flags)
- Ledger consumes: signed JSON records with prompt_md5 + model_md5 + binary_md5
- Gate reads: `tests/speed-baselines/<arch>.txt` (written by
  `speed-gate.sh --update-baselines`, which still uses the in-process
  bench; a future revision should switch it to `bench_sweep.ts`)

The suite is the bench path. The ledger is the audit trail. The gate
is the regression detector. Each has a different tolerance for unsigned
vs signed records; understand which role you need before choosing which
tool to run.
