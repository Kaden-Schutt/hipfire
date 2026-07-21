# Perf benchmarking methodology

How to measure hipfire throughput so a delta is reproducible and
auditable. This file is **protocol only** — not an executable gate, not
a number store, and not a route selector.

| Concern | Owner |
|---|---|
| Which validation route to run | [`docs/VALIDATION.md`](../VALIDATION.md) |
| Published / dated tables | [`docs/BENCHMARKS.md`](../BENCHMARKS.md) |
| Dated campaign ledgers | [`docs/perf-checkpoints/`](../perf-checkpoints/) |
| Daemon suite layout | [`bench-suite.md`](bench-suite.md) |
| Perf-variant allowlist discipline | [`perf-arch-discipline.md`](perf-arch-discipline.md) |
| Redline timed-arm proof | [`docs/REDLINE.md`](../REDLINE.md) |
| rocprof blindspot audit | [`rocprof-coverage.md`](rocprof-coverage.md) |

Truth labels for any number you produce: **measured** (named fixture +
identity + date), **historical** (retained snapshot), or **planned**.
A measured number is never a product floor, admission, or default.

## Identity before timing

Record these on every claim you intend to keep. Missing identity → the
number is exploratory only.

| Field | What |
|---|---|
| Source | Branch + commit; clean/dirty tree note |
| Binary | Path + **md5** of the executable that ran (daemon or in-process example) |
| Model | Full path + **md5** of the weight artifact; arch id; quant; sidecars + digests |
| Prompt | Committed file path + **md5 of prompt bytes**, or a deterministic token stream + digest |
| GPU | Product, gfx arch, PCI/visible-device selection, ROCm/driver identity |
| Config | KV mode, context, gen length, sampler/seed, graph/spec flags, every route-affecting `HIPFIRE_*` (set or unset) |
| Policy | Fresh-process vs resident-daemon; warmup policy; run order; run count |

Hash basis is **md5** for model, prompt, and binary — one convention
across harness, ledger, and report. Do not mix sha256 on one side and
md5 on the other.

### Byte-identical prompts

Whitespace changes that keep token count fixed can still swing DFlash τ
by double-digit percent on large models (historical observation:
single-blank vs PEP-8 triple-blank on 27B). Rules:

1. Prefer committed fixtures under `benchmarks/prompts/`.
2. Do not rely on heredocs or editor-reformatted inline strings for
   cross-session or promotion claims.
3. Record `prompt_md5` next to the result. No md5 → comparison is
   unreliable.
4. Engine entry may collapse `\n{3,}` → `\n\n` when
   `prompt_normalize=true`. Paths that bypass engine entry still need
   fixture discipline.

## Warmup (DPM + JIT)

Cold first-forward after build pays hipcc JIT and GPU clock ramp.
Treating that as “noise” is measurement error.

**Recommended for timed work:**

```bash
export HIPFIRE_DPM_WARMUP_SECS=10
```

Honored by the daemon (warmup **before** the `loaded` ack — so clients
do not fold warmup into TTFT) and by `bench_qwen35_mq4`. On
`dflash_spec_demo`, the same env is applied **pre-decode / per-row only**
(after prefill has already been timed): it stabilizes DFlash **decode**,
not the reported `prefill_tok_s`. Retain DFlash prefill only after a
separate throwaway or warmed-prefill protocol; do not treat DPM alone as
prefill warmup for that harness. Default is off so production load
latency is unchanged.

Two phases when using the native serve suite
([`scripts/serve_harness.py`](../../scripts/serve_harness.py)):

1. **Load-time DPM** — `HIPFIRE_DPM_WARMUP_SECS` pins clocks before `loaded`.
2. **Throwaway forward** — run an unrecorded matching request so JIT completes
   and clocks re-ramp after JIT idle.

Measure only after both. Continuous measured passes keep DPM high.

## Sampling policy

| Policy | Use when | Rules |
|---|---|---|
| **Fresh process** | Commit A/B, promotion, regression floors, any number you will cite later | One process per sample; rebuild when code changes; do not mix with resident samples in one A/B |
| **Resident daemon** | Multi-row sweeps on one loaded model (e.g. DFlash `--prompts-file`, one `serve_harness.py` JSON) | One suite JSON = one resident-process sample; state reset between independent rows; still record identity; do not call resident medians “fresh-process” |
| **ABBA / interleave** | Ordering scheme **on top of** matched fresh-process samples when order/thermal bias matters | Order A→B→B→A (or declared interleave); archive **raw** samples; report min / median / max — never best-of-one. ABBA is not a substitute for the fresh-process policy |

Do not promote on a single point estimate. Archive raw values and the
report path with the claim. Promotion and retained A/B evidence use
**matched fresh-process samples** (separate process per sample); apply
declared ABBA or interleave **ordering** when bias matters — never
“fresh-process **or** ABBA” as alternative policies.

Supporting scripts (claim-scoped, not universal):

| Script | Role |
|---|---|
| [`scripts/bench-cold.sh`](../../scripts/bench-cold.sh) | N fresh processes; min/median/max/spread for in-process `bench_qwen35_mq4` |
| [`scripts/probe_commits.sh`](../../scripts/probe_commits.sh) | Fresh-process commit pair compare (optional arm of `gates.sh --perf`) |
| [`scripts/speed-gate.sh`](../../scripts/speed-gate.sh) | Prefill/decode vs committed `tests/speed-baselines/<arch>.txt` when that path’s policy applies |
| [`scripts/serve_harness.py`](../../scripts/serve_harness.py) | Native Rust HTTP service battery/session driver (production path timings) |

Route selection for “must I run X?” stays in
[`VALIDATION.md`](../VALIDATION.md).

## Which number is authoritative

| Source | Use for | Do not use for |
|---|---|---|
| Daemon `done` fields (`prefill_tok_s`, `decode_tok_s`, `ttft_ms`) | Production-path throughput | — |
| `serve_harness.py` JSON | Native service-path ledger rows (when hashes present) | Unsigned exploratory if hashes missing |
| In-process `bench_qwen35_mq4` | Kernel profile, rocprof attach, speed-gate floor, bisect lower bound | Citing as production MoE decode (misses daemon AR hipGraph) |
| Probe-derived wall/gen tok/s | UX framing only | Kernel or format ranking |
| `eval_hipfire` trailing tok/s | Never as forward throughput | Kernel-speed claims (lm_head scoring loop dominates wall) |

On thinking models, probe TTFT can include the full think phase when
visible tokens are stripped — that is not prefill. Prefer daemon fields.

## Noise and confounders

There is **no single universal noise band** for every GPU, harness, and
campaign. Treat noise as something you **characterize on the fixture**,
not a constant copied from another arch.

1. **Declare** the comparison rule before measuring (e.g. “median of N
   fresh processes; accept only if |Δ| exceeds this run’s observed
   spread and transfers to a second bucket”).
2. **Investigate** before accepting or rejecting:
   - Stale binary (rebuild; delete the example binary if `ensure_build` is a no-op).
   - Cold DPM/JIT (warmup missing).
   - Thermal throttle after long sustained runs.
   - Firmware / SMU mismatch (`dmesg`; `/lib/firmware/updates/amdgpu` shadowing).
   - Multi-GPU contention (pin `HIP_VISIBLE_DEVICES`).
   - Different harness, KV mode, graph flag, or prompt bytes than the baseline.
3. Historical within-session notes (e.g. tight bands on a warm gfx1100
   fresh process, wider bands when cold-start was mislabeled as noise)
   are **scoped observations**, not global law. Campaign docs may set a
   stricter predeclared bar (Redline and LFM plans often use a ≥5% wall
   bar above local noise) — that bar is campaign-local.

A spread that is large relative to the claimed delta means more samples
or a cleaner fixture — not a hand-waved “noise” win.

## Profiling attribution

Wall tok/s alone does not name the hot kernel. Profiled runs are
**attribution-only** — not headline throughput.

1. End-to-end headline deltas and noise: **unprofiled** stationary
   production-path runs (daemon `done` fields or the production suite).
   Never use profiled tok/s for A/B acceptance or noise bands.
2. Internal timers: `HIPFIRE_PROFILE=1` / `HIPFIRE_PROFILE_DECODE=1` on
   an in-process bench when needed. These paths **synchronize after every
   kernel**, serializing launches and destroying normal async overlap —
   so their tok/s is not a valid performance A/B sample.
3. Device truth: `scripts/rocprof-wrap.sh` + `scripts/coverage-audit.py`
   ([`rocprof-coverage.md`](rocprof-coverage.md)).
4. Reconcile internal totals to rocprof. Blindspots (historical: large
   fraction of prefill time missing from internal profile) block
   confident attribution until timers land.
5. Do not blame “launch overhead” until rocprof shows it.

`hipfire profile --pp/--tg` drives synthetic `bench_prefill` /
`bench_decode` handlers — **not** a production `generate` / `done` path.
Treat those workload numbers as synthetic probes. Production-path
authority remains daemon `done` fields and the production suite
([`bench-suite.md`](bench-suite.md)).

Kernel Atlas collection and ISA views: [`kernel-atlas.md`](kernel-atlas.md).

## Spec-decode and output health

Perf without output checks can crown an attractor. For DFlash/spec
claims, pair timing with decoded-output health (unique-token /
repetition checks and eyeball) on the **same** fixture identity.
Retired `scripts/coherence-gate-*.sh` batteries are **historical
reproduction only** — not current acceptance
([`VALIDATION.md`](../VALIDATION.md) § Retired coherence-gate scripts).

Tight stddev on spec-decode is suspicious if acceptance noise should be
wider; inspect tokens.

Resident multi-row: `dflash_spec_demo --prompts-file` with fixtures
assembled from committed `.txt` files (see `benchmarks/prompts/`).
Per-row flag changes still need separate process groups.

## Disposition

Every kept result ends in one disposition:

| Disposition | Meaning |
|---|---|
| **Exploratory** | Local only; incomplete identity or single sample |
| **Measured** | Full identity + raw samples + declared rule; not a product default |
| **Historical** | Superseded config or methodology; retain, do not re-baseline product on it |
| **Rejected / null** | Failed rule or confounder; record and stop — do not accumulate speculative “wins” |
| **Promotion candidate** | Matched **fresh-process** A/B samples with declared ABBA/interleave **order** where order bias matters, gain above characterized noise, transfers across required buckets, intended path proven active (rocprof/profile on separate attribution runs), correctness route from VALIDATION still green | Still needs the owner’s admission/policy step — measurement alone does not write [`admissions.yml`](../admissions.yml) |

Redline-attributed promotion additionally requires the certification
ladder in [`REDLINE.md`](../REDLINE.md). Capture success ≠ timed-arm
route proof.

## Evidence durability

- Prefer append-only dated files under `docs/perf-checkpoints/` or a
  campaign ledger over editing old tables in place.
- Published snapshots live in [`BENCHMARKS.md`](../BENCHMARKS.md) and
  stay labeled historical/measured as that file states.
- Raw reports (JSONL, rocprof CSV, ABBA logs) keep paths and digests in
  the write-up; workstation-local paths are limitations, not secrets to
  omit.
- Stale baseline smell test: if `binary_md5` or model/prompt identity
  does not match the claim’s fixture, re-measure — do not argue from a
  weeks-old recalled number.

## Explicit non-goals

- No universal GPU gate and no restoration of retired coherence batteries
  as acceptance.
- No promotion by point estimate, best run, or harness success without
  the proof the claim needs.
- No copying mutable speed matrices into this file — link BENCHMARKS /
  checkpoints.
- No treating speed-gate floors or in-process lower bounds as production
  throughput.
