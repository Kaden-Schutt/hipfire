# Bench suite

Implementation map for hipfire’s **daemon-driven** throughput harness and
related tools. Measurement rules (warmup, identity, noise, disposition)
live in [`perf-benchmarking.md`](perf-benchmarking.md). Published numbers
live in [`docs/BENCHMARKS.md`](../BENCHMARKS.md). Which route to run lives
in [`docs/VALIDATION.md`](../VALIDATION.md).

## Why daemon-driven

In-process examples and the production daemon are not the same execution
path. A documented class of gap: MoE AR-decode measured in-process without
the daemon AR hipGraph path under-reads production `decode_tok_s` by a
large fixed bias (order ~20% on A3B-class models in the original
investigation). That gap is structural — not fixed by longer warmup.

| Path | Typical use |
|---|---|
| Daemon + suite / serve | Production-path prefill/decode claims |
| In-process `bench_qwen35_mq4` | Speed-gate floor, bisect lower bound, `HIPFIRE_PROFILE` / rocprof attach |
| `probe_commits.sh` | Optional fresh-process commit pair (uses in-process bench) |

Do not cite in-process AR-decode as production throughput. Do not cite
daemon numbers as the speed-gate’s calibrated floor without re-baselining
that gate on purpose.

## Canonical suite: `cli/bench_sweep.ts`

```text
HIPFIRE_DAEMON_BIN=./target/release/examples/daemon \
  bun cli/bench_sweep.ts <model_path> [max_seq] [pp_csv] [gen_tokens] [prompt...]
```

| Defaults (source) | Value |
|---|---|
| `max_seq` | `9216` |
| `pp_csv` | `128,256,512,1024,4096,8192` |
| `gen_tokens` | `128` |
| Prompt | Inline trailing argv, or built-in English default if omitted |
| KV | `HIPFIRE_BENCH_KV` or default **`q8`** |
| DPM | Sets `HIPFIRE_DPM_WARMUP_SECS=10` if unset |
| Kernel cache | Sets `HIPFIRE_KERNEL_CACHE` under `~/.cache/hipfire_kernels` if unset |

### What it measures

| Phase | Mechanism | Output |
|---|---|---|
| Prefill sweep | `bench_prefill` ×3 per length; **median** | `pp: { "<n>": tok/s \| null \| { "error": string } }` |
| Natural prefill | `done.prefill_tok_s` on measured generate | `prefill_natural_tok_s` |
| AR decode | `done.decode_tok_s` (fallback `tok_s`) | `decode_tok_s` |

`pp` entry is `null` when `pp + 32 > max_seq`. A per-shape daemon failure
emits `{ "error": <message> }` for that length rather than a number. One
JSON object on stdout on success; on failure the object may be
`{ model, error }` or may include a top-level `error` alongside any
partial `pp` rows already collected.

### Warmup baked in

1. Daemon load with DPM warmup before `loaded` (when env set).
2. Throwaway: every pp shape once + one `generate` (max 192) for JIT +
   clock re-ramp.
3. Measured pass immediately after (continuous → DPM stays high).

### Production-path flags

For ledger-grade daemon decode, set explicitly:

```bash
export HIPFIRE_AR_GRAPH=1
export HIPFIRE_DPM_WARMUP_SECS=10
export HIPFIRE_DAEMON_BIN=./target/release/examples/daemon
```

The suite does **not** force `HIPFIRE_AR_GRAPH`; omitting it measures
whatever the daemon default is. Record the flag with the result.

Build:

```bash
cargo build --release -p hipfire-runtime --example daemon --features deltanet
```

### Identity fields (partial emit; full identity required for kept records)

One `bench_sweep.ts` invocation loads a **single resident daemon** and
emits **one JSON object**. That object is **one resident-process sample**
— not a fresh-process multi-run set and not by itself promotion-grade
evidence. Retained A/B or promotion work must archive **multiple fresh
daemon invocations** (separate process per sample) and their raw JSON in
a declared order, using ABBA/interleaving when order or thermal bias
matters (see [`perf-benchmarking.md`](perf-benchmarking.md)).

Emitted today from source (partial identity only):

| Field | Meaning |
|---|---|
| `model` | Model path string |
| `arch` | Daemon **model-family** label from `loaded.arch` (e.g. `qwen3_5`, `lfm2moe`) — **not** GPU gfx identity and **not** the numeric model arch id |
| `kv` | KV mode used |
| `max_seq` | Max sequence configured |
| `pp` | Prefill sweep map (see schema above) |
| `decode_tok_s` | AR decode throughput |
| `prefill_natural_tok_s` | Natural-prefill throughput from measured generate |

The script does **not** emit a complete kept-record identity. Before
treating any line as durable evidence, record **every** field required by
[`perf-benchmarking.md`](perf-benchmarking.md) § Identity before timing —
including (and not limited to) fields the script omits today:

| Omitted field | How |
|---|---|
| Source (branch + commit; clean/dirty tree) | `git` at run start |
| `prompt_md5` | md5 of exact prompt bytes sent |
| `model_md5` | md5 of weight file |
| Quant + sidecar digests | artifact metadata + md5s |
| `binary_md5` | md5 of `HIPFIRE_DAEMON_BIN` |
| GPU product, gfx arch, PCI / `HIP_VISIBLE_DEVICES` | host inventory at run |
| ROCm / driver identity | `rocminfo` / package versions |
| Route-affecting config | KV, context, gen length, sampler/seed, graph/spec flags, every `HIPFIRE_*` set or unset |
| `hipfire_ar_graph` / flags string | env at launch |
| Process / run-order policy | fresh vs resident; warmup; ABBA/interleave order; run count |
| `timestamp_utc` | wall clock at run start |

Inline default prompt is fine for local exploration. Cross-session or
promotion work must use a committed fixture and `prompt_md5` (see
[`perf-benchmarking.md`](perf-benchmarking.md)).

### Example

```bash
export HIPFIRE_DAEMON_BIN=./target/release/examples/daemon
export HIPFIRE_AR_GRAPH=1
export HIPFIRE_DPM_WARMUP_SECS=10

bun cli/bench_sweep.ts ~/.hipfire/models/qwen3.5-9b.mq4 \
  9216 128,512,2048 128 \
  "$(cat benchmarks/prompts/bare_factual.txt)"
```

## Related harnesses (not the suite)

| Tool | Role | Not for |
|---|---|---|
| `hipfire bench` / daemon `bench_prefill` | Product CLI and daemon arms | Replacing identity protocol |
| `bench_qwen35_mq4` | In-process pp/decode; profile hooks | Production MoE decode claims |
| `scripts/bench-cold.sh` | N× fresh-process wrapper around in-process bench | Daemon path |
| `scripts/speed-gate.sh` | Regression vs `tests/speed-baselines/<arch>.txt` | Production throughput tables |
| `scripts/probe_commits.sh` | Fresh-process commit A/B (in-process) | Daemon path numbers |
| `dflash_spec_demo --prompts-file` | Resident multi-row DFlash τ / tok/s | AR suite substitute |
| `scripts/rocprof-wrap.sh` + `coverage-audit.py` | Device-time attribution | End-to-end alone |
| `scripts/gates.sh --perf` | Optional manual wrapper calling `probe_commits.sh` | Universal gate |

DFlash τ is **out of scope** for `bench_sweep.ts`. Use
`dflash_spec_demo` with fixture manifests under `benchmarks/prompts/`.

## Migration posture (bench path only)

“Retired from the **bench path**” means: do not use as the source of
production-throughput ledger rows or commit-message production claims.
Correctness, profiling, and bisect uses remain.

| Prefer | Instead of (for production-path numbers) |
|---|---|
| `bench_sweep.ts` decode/pp | `bench_qwen35_mq4` AR-decode as production |
| `dflash_spec_demo --prompts-file` + fixtures | `--prompt '...inline...'` for multi-session τ |
| Protocol in `perf-benchmarking.md` + VALIDATION routes | Any single script as universal acceptance |

Keep: `infer_*` interactive paths; in-process bench for rocprof;
`probe_commits.sh` / `speed-gate.sh` for lower-bound regression and
bisect when their policy applies.

## Suite ↔ discipline split

| Artifact | Role |
|---|---|
| `cli/bench_sweep.ts` | Producer of daemon-path JSON |
| [`perf-arch-discipline.md`](perf-arch-discipline.md) | How perf **variants** are selected and evidenced |
| [`perf-benchmarking.md`](perf-benchmarking.md) | How any throughput claim is sampled and disposed |
| `tests/speed-baselines/<arch>.txt` | Speed-gate floor (in-process calibrated) |
| `docs/BENCHMARKS.md` / `perf-checkpoints/` | Human-facing measured/historical tables |

No suite exit code admits a model route. Admissions stay in
[`docs/admissions.yml`](../admissions.yml) (schema v2; fail closed outside the exact earned row).
