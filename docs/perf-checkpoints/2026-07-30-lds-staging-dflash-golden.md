# LDS-staged weight GEMMs on the 27B DFlash golden run — independent re-measurement

**Date:** 2026-07-30 · **Host:** hiptrx · **Arch:** gfx1201 (AMD Radeon AI PRO
R9700, RDNA4) · **Tree:** `bcd1ef0082e7f7f7a1eff9252da5d76e17d042b0` (beta tip)

> **Policy note.** [`AGENTS.md`](../../AGENTS.md) §5 makes git history the
> canonical home for perf numbers, and archived the `docs/perf-checkpoints/`
> tree on 2026-04-27. This file exists because the measurement was requested as
> a standalone document; the headline numbers are duplicated in the commit body
> so history remains authoritative. It is a **measurement record**, not an
> acceptance gate — [`docs/VALIDATION.md`](../VALIDATION.md) owns route
> selection and explicitly rejects a bench number as promotion evidence.

## What this confirms

`HIPFIRE_HFQ4G256_LDSSTAGE=1` is worth **+9.06%** decode on the 27B DFlash
golden run, reproducing a previously recorded +9.1% on a different tree, a
different day, and a different binary.

| ctx 2048, `HIPFIRE_Q8_PREFILL_M4=0` | LDS off | LDS on | ratio |
|---|---:|---:|---:|
| decode tok/s (3 fresh reps) | 255.53 / 254.88 / 254.84 | 278.24 / 277.97 / 276.92 | |
| **median** | **254.88** | **277.97** | **1.0906×** |
| spread | 0.27% | 0.47% | |
| prior record (older tree) | 254.21 | 277.19 | 1.0904× |
| delta vs prior record | +0.26% | +0.28% | +0.0002 |

The LDS-off arm also lands **0.02%** from the published golden baseline of
254.83 tok/s (`hipfire v0.2.0+3730b58`, `feat/dense-dflash-perfmaxx`
`3730b58bd3b5380eb1de672ec032b24016905458`).

## What this does NOT claim

- Not an admission, promotion, or acceptance. See `docs/VALIDATION.md`.
- Not bit-exactness. LDS staging **reorders f32 K-accumulation** by construction
  (8 waves × disjoint 64-K slices, fixed-order reduction). On this fixture
  τ/accept/tokens were identical, but that is one prompt at temp 0.0, not a
  quality result. The flag remains opt-in for exactly this reason.
- Not a serve number. `dflash_spec_demo` is an isolated kernel-path instrument;
  serve carries per-token host work and different prompts. The user-facing serve
  delta for this flag measured **+3.9%**, not +9%.

## Identity

Every arm below used **one binary**, verified by digest:

| item | value |
|---|---|
| binary | `target/release/examples/dflash_spec_demo` |
| binary sha256 | `a107cf8640abf1d7cb1b3aeec26f33a07a01b00abdab53956c29b6a7b83c1ab5` |
| binary size | 13 569 168 B |
| tree | `bcd1ef0082e7f7f7a1eff9252da5d76e17d042b0`, 0 dirty files |
| target trunk | `qwen3.6-27b-awq.mq4` → `qwen3-27b-3.6.mq4-awq.remote-mi300x` |
| trunk sha256 | `86a5f80fd29d545abb1093dead242725ced6d68b8607c6d566d897b1a82442dc` |
| trunk size | 14 984 158 208 B |
| draft | `qwen36-27b-dflash-mq4.hf4` → `…hfq` |
| draft md5 | `204c4c4ceab30cb9ebc118fa9d59a446` |
| draft size | 919 401 472 B |
| prompt | `benchmarks/prompts/merge_sort_thinking_off.txt` |
| prompt md5 | `253c7ac50857fe6d0e10fb0d2c5e35c0` (140 B) |
| HIP | 7.14.60850 · ROCm `/opt/rocm/core-7.14` · kernel 7.0.0-27-generic |
| GPU | `HIP_VISIBLE_DEVICES=3` of 4× gfx1201; 32 624 MB total, 17 034 MB used |

Fixtures are identified by digest, never filename: both 27B paths on this host
are symlinks, and lookalike AWQ/MQ4 files are not comparable.

## Reproduction

```bash
# 1. Tree at the measured tip. NOTE: the local clone on hiptrx predates the
#    push, so fetch the real remote, not `origin`.
git fetch https://github.com/warpfront/hipfire.git beta
git worktree add -f --detach ~/hf-beta-lds bcd1ef008
cd ~/hf-beta-lds

# 2. One binary for every arm; LDS and M4 are env flags, never rebuilds.
export PATH=/opt/rocm/core-7.14/bin:$PATH
cargo build --release --locked --features deltanet \
  -p hipfire-runtime --example dflash_spec_demo
sha256sum target/release/examples/dflash_spec_demo   # must be a107cf86…

# 3. Arms. Discard one warm-up run, then 3 fresh processes per arm.
export HIP_VISIBLE_DEVICES=3
export HIPFIRE_Q8_PREFILL_M4=0        # see "M4" below — this is load-bearing
export HIPFIRE_HFQ4G256_LDSSTAGE=1    # or unset for the baseline arm

./target/release/examples/dflash_spec_demo \
  --target /home/kaden/.hipfire/models/qwen3.6-27b-awq.mq4 \
  --draft  /home/kaden/.hipfire/models/qwen36-27b-dflash-mq4.hf4 \
  --prompt-file benchmarks/prompts/merge_sort_thinking_off.txt \
  --max 256 --temp 0.0 --no-chatml --kv-mode q8 --ctx 2048
```

**Adaptive B on purpose.** Do not add `--block-size N --no-adaptive-b`; the
pinned-B regime is a different measurement (~2.3× apart) and is not comparable
to this line.

**Routing proof, not inference.** An LDS-on arm must leave both objects in the
kernel cache, which is `$CWD/.hipfire_kernels` — *not* under `HOME`:

```
.hipfire_kernels/gfx1201/gemm_gate_up_hfq4g256_wmma_gfx12_ldsstage.hsaco   17 448 B
.hipfire_kernels/gfx1201/gemm_hfq4g256_residual_wmma_gfx12_ldsstage.hsaco  15 400 B
```

The first LDS-on run pays a JIT compile (its prefill reads ~30 tok/s); that is
why a warm-up run is discarded rather than averaged.

## Full grid at ctx 4096

25 timed runs total. A/B/B/A ordering within each driver so drift cannot alias
the effect; 6 reps per cell.

| cell | decode tok/s (6 reps) | median | spread |
|---|---|---:|---:|
| M4 on, LDS off | 241.93 / 242.20 / 241.94 / 241.79 / 241.48 / 241.67 | 241.86 | 0.30% |
| M4 on, LDS on | 262.29 / 262.42 / 262.49 / 261.99 / 262.16 / 262.15 | 262.23 | 0.19% |
| M4 off, LDS off | 252.05 / 252.41 / 252.35 / 251.26 / 251.89 / 251.47 | 251.97 | 0.46% |
| M4 off, LDS on | 274.04 / 274.00 / 273.28 / 273.77 / 273.05 / 273.32 | 273.55 | 0.36% |

Derived, all within-cell spreads ≤0.46%:

| effect | at ctx 4096 | at ctx 2048 |
|---|---:|---:|
| LDS staging, M4 off | **1.0856×** | **1.0906×** |
| LDS staging, M4 on | 1.0842× | not measured |
| M4 default-on cost, LDS off | **0.9599× (−4.01%)** | not measured |
| M4 default-on cost, LDS on | **0.9586× (−4.14%)** | not measured |

The LDS ratio is stable across the M4 axis (1.0842 vs 1.0856), so the two
changes are independent here rather than competing.

## Two findings that came out of the re-measurement

### 1. Allocated context costs decode even when the work is identical

Every run emits exactly 162 tokens with τ = 11.3846. Halving `--ctx` alone:

| arm (M4 off) | ctx 4096 | ctx 2048 | gain |
|---|---:|---:|---:|
| LDS off | 251.97 | 254.88 | +1.16% |
| LDS on | 273.55 | 277.97 | +1.62% |

Mechanism, from source rather than speculation: under hipGraph capture, scalar
kernargs are baked at capture time, so `qwen35.rs:8842` deliberately sets
`max_ctx_len = kv_cache.physical_cap` instead of the live `start_pos + n`,
sizing the attention kernel's LDS `scores[]` for the worst case. The kernel
still iterates the true per-row `positions[b] + 1` from a device buffer, so
output is unaffected — but the LDS/scan bound tracks the **allocation**, and
that is charged every cycle. `HIPFIRE_VERIFY_GRAPH` is default-on, so DFlash
verify is always in this regime.

This is unrelated to FireMap VMM KV (#549): `VmmArena` made the KV *backing*
lazy, while this value sizes per-workgroup LDS scratch, which has no VMM.

**[INFERENCE]** The prior +9.1% record was almost certainly taken at ctx 2048:
its absolutes (254.21 / 277.19) match this document's ctx-2048 arms to 0.26% /
0.28%, while the ctx-4096 arms sit ~1.2–1.6% lower. Measuring the reproduction
at ctx 4096 is what initially made the win look like +8.4% with "both arms 5%
low".

### 2. `HIPFIRE_Q8_PREFILL_M4` costs ~4% on the spec route and wins on AR

M4 (the four-query Q8 KV reuse tile, gfx12) was default-on at this tree and
costs **−4.01% / −4.14%** on this run. Its context floor
(`HIPFIRE_Q8_PREFILL_M4_MIN_CTX`, default 1024) does not exclude DFlash verify,
because the floor is compared against `max_ctx_len` — which, per finding 1, is
`physical_cap` under graph capture, never the true length. So the floor always
passes and the tile is always admitted into verify.

A follow-up sweep measured where the tile actually starts paying, on the AR
prefill path (DFlash off, LDS-GEMM off to isolate the attention axis). Three
GPUs, A/B/B/A, full pp curve per device:

| pp | mean M4-on / M4-off | verdict |
|---:|---:|---|
| 128 | 0.9964 | tie (inside 1.0–1.4% noise) |
| 512 | 0.9949 | tie |
| 1024 | 1.0000 | dead tie |
| 1536 | 1.0103 | M4, marginal |
| 2048 | 1.0193 | M4 |
| 3072 | 1.0339 | M4 |
| 3584 | 1.0405 | M4 |
| 4096 | 1.0471 | M4 |
| 4608 | 1.0457 | M4 |
| 5120 | 1.0455 | M4 |
| 8192 | 1.0491 | M4 |

Per-pp agreement across the three devices is ±0.0006; same-flag A/B/B/A drift
is ≤1.4% at low pp and ≤0.76% from 2048 up. **The crossover is ctx ≈ 1024, not
4096**, and there is no regime in 128–8192 where the single-pass LDS attention
kernel measurably *wins* on AR — the sub-1024 points are ties, not losses.

So route, not context, is the separator: the same kernel costs 4% on spec and
wins 1.9–4.9% on AR. The rule landed accordingly — a route flag
(`set_dflash_spec_active`, set by `spec_step_dflash`, the entry both the serve
and bench halves of the DFlash policy funnel through) consulted by
`q8_prefill_m4_eligible`. The 1024 ctx floor stays as-is because it now matches
the measured crossover; raising it to 4096 would forfeit 1.9–4.7% of AR prefill
across 2048–4096 to fix a problem the route flag already fixes.

Two notes for whoever picks this up next:

1. **The underlying defect is broader than M4.** `max_ctx_len` doubles as an
   LDS-sizing constant and a routing input. The same dispatch site keys its
   4096 crossover on it, so that decision is also made on the allocation rather
   than the live length under capture. M4 is now routed around it; the
   crossover is not, and its measured numbers were taken with this behaviour in
   place.
2. **The tile is being superseded by its own author.** Per #534, HUSRCF calls
   #554 "a very primary trial" and is rebasing toward a **16-query** KV share
   plus VOPD-friendly K-pair packing, projecting another 20–30%. The hoist
   ported into the 4-query tile proved the reuse and widening axes compose;
   it should not be defended as architecture past that rebase.

## Integrity checks performed

- **Non-empty guard before any comparison.** All 25 runs: `exit=0`,
  `emitted: 162 tokens`. A vacuous compare on empty output is the failure mode
  this guards.
- **Workload identity across every arm.** τ = 11.3846, accept_rate = 0.7590,
  cycles = 13, tokens = 162 — identical to four decimals in all 25 runs. Since
  τ is an acceptance count rather than a timing artefact, identical τ across
  arms proves the arms did the same work; a τ shift would have meant the
  comparison was invalid.
- **Same binary in every arm**, digest-checked at the start of both drivers.
- **No trace or profiling env.** `env | grep HIPFIRE_` was empty at driver
  start and is recorded in the telemetry file. A trace run can prove routing; it
  can never supply a number.
- **Warm-up discarded** after the build and on first LDS-on JIT.
- **Single-agent timing.** Verified no concurrent `pf_measure`/daemon activity
  on the host; the daemon `flock` is on `$HOME/.hipfire/daemon.pid` and is
  host-scoped, not device-scoped, so `HIP_VISIBLE_DEVICES` does not isolate it —
  only a private `HOME` does.

## Raw artifacts

In-repo: `benchmarks/results/hiptrx-gfx1201-dflash-lds-golden-2026-07-30/`
carries the six winner/baseline logs plus a standalone README.

On hiptrx, all under `~/`:

| directory | contents |
|---|---|
| `ctx2048-golden-dflash-ldson-m4off-2026-07-30` | the winning arm (277.97) |
| `ctx2048-golden-dflash-ldsoff-m4off-2026-07-30` | its baseline (254.88) |
| `ctx4096-golden-dflash-lds-ab-m4on-2026-07-30` | ctx-4096 grid, M4 default-on, + `telemetry.txt`, `routing.txt` |
| `ctx4096-golden-dflash-lds-ab-m4off-2026-07-30` | ctx-4096 grid, M4=0, + `integrity.txt` |
| `boundary-m4-vs-lds-pp2048-8192-gpu{0,1,2}-2026-07-30` | boundary sweep, upper curve |
| `boundary-m4-vs-lds-pp128-2048-gpu{0,1,2}-2026-07-30` | boundary sweep, sub-2048 |
| `verify-m4-rule-2026-07-30` | acceptance test for the route rule |

Drivers: `/tmp/lds-confirm.sh`, `/tmp/lds-m4.sh`, `/tmp/boundary.sh`,
`/tmp/boundlow.sh`, `/tmp/verify-rule.sh`.
