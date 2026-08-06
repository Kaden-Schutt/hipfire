# mq2r: a 12.3 GB A3B SKU at mq4r decode parity — campaign handoff

**Workload:** Qwen 3.6 35B-A3B, ordinary autoregressive decode, Q8 KV, no MTP or
speculative decoding.

**Hardware:** Radeon AI PRO R9700 (`gfx1201`), hiptrx, GPU 0 only.

**Branch:** `a3b-lowbit-work` @ `468954433`, on `beta` = `e2f7dd1a4`.

**Companion:** `docs/perf-checkpoints/2026-07-13-redline-mq4r-110-to-204.md`
(the mq4r 110 → 204 campaign). This document is the codebook-SKU counterpart and
assumes that one for Phases 2–4.

---

## Result

`mq2r` — **12.325 GB** — reaches **204.3 tok/s PM4 decode**, against `mq4r`'s
**201.9 tok/s at 18.65 GB**. Decode parity at **66% of the size**, with better
KLD than the shipping `.mq2` on both eval slices.

| SKU | size | wt2 KLD | agentic KLD | PM4 decode |
|---|---|---|---|---:|
| `.mq2` (ships today) | 11.61 GB | 0.2384 | 0.4692 | 135.9 |
| **`mq2r`** (this work) | **12.325 GB** | **0.2356** | **0.4631** | **204.3** |
| `mq4r` | 18.65 GB | — | — | 201.9 |
| `mq2p` (= P3) | 13.02 GB | **0.1841** | **0.4164** | 124.5 |
| mq2r-GL | 11.53 GB | 0.2099 | 0.4448 | 129.2 |

mq2r's 200.6 → **204.3** came from `75fe90fbb` (codebook resource contracts, § "Live
defects" item 2): +1.78% order-balanced over 12 runs, bit-exact, and it now *beats*
mq4r rather than tying it. Earlier rows in this table predate that fix.

`mq2r` composition: **all-MQ4 fixed tier** (attn + lm_head + embed + router) +
**routed gate_up MQ2G256Lloyd / down MQ3G256Lloyd**, `--no-kmap`.
File: `~/.hipfire/models/qwen3.6-35b-a3b.mq2r` (back-compat symlink at the old
working name `q36a3b.p1-fixmq4-gu2l-dn3l.hfq`). Encode ≈ 5 minutes.

---

## What actually moved the number

### 1. The model could not serve at all (`a28b99f9e`)

`forward_prefill_batch_with_pbs_opts` hard-errored on any `MQ3G256Lloyd` inside
a MoE layer, *before* computing `eligible`. The refusal protects the batched MoE
bodies (HFQ4-stride QKV matcher, `wo` hardcoded to `gemm_hfq4g256_residual` →
104/112-vs-136 stride corruption), but those bodies only run when the batched
path is eligible — and `moe_ffn_batched_admissible_for_dtypes` rejects
`MQ2G256Lloyd` in every arm. The refusal was pre-empting a correct per-token
fallback.

Fix: defer it until after `eligible`, gate on
`mq3_in_moe && (eligible || gdn_tape.is_some())`. The tape clause is
load-bearing — the `!eligible` fallback leaves a passed GDN tape **stale rather
than erroring**, so spec callers must still get the loud refusal.

Deliberately **not** changed: the predicate itself, and the sibling guard in
`forward_prefill_batch_single_chunk_captured_opts` — that entry point has no
eligibility check and no fallback, so its refusal is the only protection there.

### 2. PM4 lowering was gated by a file extension

`mq4r_redline_default()` (`crates/hipfire-runtime/src/config.rs`) admits the
Redline retained-PM4 path on a **file-extension test**:

```rust
matches!(gpu_arch, "gfx1100" | "gfx1151" | "gfx1201") && pp == 1 && tp == 1
    && extension.eq_ignore_ascii_case("mq4r")
```

So `mq4r` auto-ran PM4 while every codebook SKU ran HIP. **Every cross-SKU
comparison before this was HIP-vs-PM4 and is worthless.** `rocprofv3
--kernel-trace` also cannot see PM4 dispatches, which is why mq4r appeared to
"hide" 85% of its wall time.

`mq2r` lowers cleanly — stable capture, valid AQL contracts, route proof
`retained_rows=10 lifecycle_valid=True` — and PM4 is worth **+16.5%** to it
(vs +13.0% to mq4r). It is still opt-in: the extension is `mq2r`, which does not
match the `.mq4r` literal.

### 3. The routed down GEMV was running a naive decomposition

Per-dispatch attribution showed both SKUs issue **exactly 693 dispatches** and
land within 0.1% on total GPU time, with the whole difference in two kernels:

| projection | mq2r | mq4r |
|---|---|---|
| gate_up | **347.6 µs** (2.25 bpw wins by 302 µs) | 649.7 µs |
| down | 882.2 µs, grid 2048×8 | **585.3 µs**, grid 128×1 (`ninepath`) |

mq2r's bit advantage was real and it **gave all of it back on the down
projection** — not from dtype, from decomposition. The incumbent launched one
32-thread block per (row, krank): **16,384 single-wave workgroups**, eight fills
of a 2048-slot part, each re-reading the same rotated activation. 166 GB/s
effective against gate_up's 543.

Two fixes, both **default OFF**:

- **`68a70a175` — R2/R4 row tile** (`HIPFIRE_MQ3_DOWN_ROWS={2,4}`).
  R2: **193.8 → 201.1 PM4 (+3.8%)**, HIP 168.5 → 175.4. R4 regresses (94 VGPR
  against the 96-register 16-wave budget).
- **`468954433` — ninepath port** (`gemv_mq3g256_lloyd_moe_ninepath_d4`).
  One 256-thread CTA, 8 warps, activation staged into LDS **once**, expanded
  intermediate never leaves LDS, warp 0 folds in ascending krank order —
  **single owner per row, no atomics, deterministic under graph replay.**
  Down: **882 → 510 µs**, beating mq4r's 585 µs on the same projection.
  End-to-end 193.5 → 200.6 PM4 (+3.7%).

**Row tile and ninepath measure the same end-to-end** (200.6 vs 200.3, within
noise) despite ninepath being 42% faster on the kernel — the down projection
stops being the limiter well before 510 µs. Prefer **ninepath**: same speed, no
atomics, deterministic replay.

Also ported to `.mq2`'s MQ2L down (`9bd03ac38`, `44739d429`): **+3.0% PM4**
(135.9 → 139.9) on the shipping SKU.

---

## The defect class (reusable)

`quads = groups_per_row >> 2` is **zero** whenever `K < 1024`, so the
K4-unrolled main loop never executes. On a3b the down projection has
`K = moe_intermediate = 512`; gate_up is safe because its K is `hidden` (2048).

**It only hurts when the tail path also restages an LDS codebook per group.**
Swept every G256 kernel:

| kernel | tail restage | verdict |
|---|---|---|
| `gemv_mq3g256_lloyd_moe_down_indexed` | 6 sites | **fixed** |
| `gemv_mq2g256_lloyd_moe_down_indexed` | 5 sites | **fixed** |
| `gemv_mq2g256gl_moe_down_indexed` | 0 | dead loop, no restage — unmeasured |
| `gemv_mq3g256gl_moe_down_indexed` | 0 | dead loop, no restage — unmeasured |
| `gemv_hfq4g256_residual_scaled` (shared expert, also K=512) | 0 | **false positive** — no codebook, no LDS, no barriers |

**General rule:** any MoE with `moe_intermediate < 1024` hits this on the down
projection with a G256 codebook format. **ds4 (`moe_intermediate = 2048`) is
unaffected**, so `mq2rxt` does not inherit it.

---

## Falsified — do not retry

- **E8-SoA ALU reduction.** −39% VALU → −0.4%; −46% VALU → **−1.5% slower**.
  Monotonically wrong-way. Attention decode is not ALU-bound.
- **E8 AoS container.** Lower VALU/MAC (9.50 vs 10.19) *and* a 4-accumulator
  loop, measured **6.2% slower** (141.8 vs 151.1). Same KLD. SoA has the worse
  kernel but the better container; AoS's 17-byte blocks are never 16 B aligned.
- **`fused_qkvza` scalar-header lowering.** The 7 `global_load_b32` are at
  offsets 8/144/280/416 — stride 136, i.e. the *per-lane packed index data*. The
  `(scale, zp)` header at +0/+4 produces **zero** vector loads; already scalar.
- **atomicAdd as the down-GEMV cost.** `global_atomic_add_f32` is followed
  immediately by `s_endpgm` with **no `s_waitcnt`** — fire-and-forget, one per
  wave, ~2% of RDNA4 L2 atomic capability. The mq4r-style expanded-buffer
  alternative was quantified and rejected (+40 launches ≈ 160–340 µs against an
  atomic cost bounded at ~186 µs).
- **MQ3-vs-MQ2 decode cost.** One extra byte load per group per lane.
- **E8 lm_head over MQ4.** −0.45% KLD, free — a **null result**. E8 belongs in
  *attention* (−4.6% KLD for −7.5% speed), ~10× more valuable there.
- **Byte/traffic models as speed predictors.** Mispredicted in *both*
  directions across the session.

---

## Measurement methodology — read before profiling

**`rocprofv3 --kernel-trace` (direct per-dispatch timestamps) is the instrument
for per-kernel attribution.** The PM4 prefix-differencing profiler
(`redline_daemon_harness.py --profile-prefix-*` + `analyze_pm4_prefix_profile.py`)
is reliable for **totals and end-to-end only**.

Prefix differencing attributes *marginal* cost, so a dispatch's number absorbs
the wait/drain it exposes. With `steady_state=False` and `repeats=2` each sample
also resets and re-prefills, making every small-kernel number a difference of two
large noisy totals. Measured instability across two runs where only the down
kernel changed:

| kernel (untouched) | run A | run B | swing |
|---|---|---|---|
| lm_head (435 µs, isolated) | 435.0 | 434.9 | **0%** |
| router | 255.8 | 348.0 | +36% |
| `fused_qkv` | 119.2 | 271.6 | +128% |
| `sigmoid_mul_f32` | 55.0 | 181.2 | +230% |
| `fused_silu_mul_mq_rotate` | 44.4 | 387.8 | **+773%** |

The two instruments agree within a couple of points on large dispatches
(`fused_qkvza` 19.2% vs 19.4%, lm_head 9.2% vs 8.3%) and diverge as dispatches
shrink. **Any conclusion about small kernels drawn from prefix differencing is
unsupported** — including the "~28% of the token sits in tiny-grid kernels"
figure produced during this campaign.

Other traps, all cost real time here:

- `--timeout` defaults to **120 s** and covers the *entire* prefix sweep in one
  daemon request. A 693-prefix sweep takes ~10 min and dies at prefix ~149 with
  an opaque `Expecting value: line 1 column 1`. Pass `--timeout 3600`.
- The daemon requires a numeric **`attempt_id`** on every generate; a generation
  truncated inside a `<think>` span is rejected wholesale (`rolled_back`);
  `kv_mode`/`max_seq` must be inside **`params`** (default `max_seq` is 4096).
  **Use `scripts/serve_harness.py` / `tools.redline bench`, not a hand-rolled
  driver** — a mis-nested `kv_mode` is silently ignored and the run is not q8.
- `ssh host 'nohup bash x.sh'` is **non-login** → no `hipcc` → the engine warns
  and falls back to **UNVALIDATED prebuilt kernel blobs**, to the *daemon's*
  stderr which serve_harness does not capture. Use `ssh host 'bash -lc "..."'`
  and grep for `UNVALIDATED`.
- The stdio daemon survives `terminate()` and holds `~/.hipfire/daemon.pid`,
  blocking every subsequent daemon **box-wide**. Its cmdline carries no model
  name; identify by `/proc/<pid>/cwd`.

---

## Live defects found incidentally — not fixed

1. **`hipfire_gdn_reduce_sum_lane0_dpp` is wrong.**
   `kernels/src/gated_delta_net_q8_fast.hip:57` calls
   `__builtin_amdgcn_permlanex16(v, v, 0, 0, …)` with **zero lane-select
   DWORDs**, which broadcasts lane 16 to lanes 0–15 instead of swapping
   lane↔lane^16 (correct selects `0x76543210` / `0xfedcba98`). Its lane-0 sum is
   `sum(v0..v15) + 16*v16`. **Live on gfx1151** via
   `gated_delta_net_q8_compact2_dpp_gfx1151`
   (`rdna-compute/src/kernels.rs:4411`, selected at `norm.rs:2400`).

2. **`replay.rs` keys PM4 resource tracking on literal kernel-name strings** —
   `pointer_effects()` and `expected_kernarg_bytes()`. A renamed or new kernel
   falls out of both, `ResourceFrontier::independent()` goes false at every
   boundary, and `wait_compute_idle()` is forced **with no error**.
   **FIXED for the codebook family in `75fe90fbb`** — the MQ2/MQ3 G256-Lloyd
   routed-MoE kernels had *no* entries at all, which is why mq2r proved only
   90 of 692 boundaries independent against mq4r's 130 of 692. Registering the
   13 symbols took coverage 532 → **692/692** and independence 90 → **130**,
   worth **+1.78%** PM4 decode (200.67 → 204.27, order-balanced, 12 runs).
   `codebook_moe_symbols_have_resource_contracts` now fails if a codebook
   variant lands unregistered. **The general hazard remains** for every family
   without such a guard: any new kernel variant must be registered before a PM4
   A/B means anything, and mq4r itself still sits at `covered=612/692`.

3. **`HIPFIRE_FIXED_TIER` is silently ignored under `--no-q8-router`** —
   `main.rs:10537` gates the whole fixed-tier override behind
   `q8_router && is_q8_tensor(name)`. Cost an 11 GB byte-identical duplicate
   encode. Use `HIPFIRE_Q8_CLASSES=` (empty) instead. Always md5 a prefix of a
   new encode against its sibling.

---

## The ceiling — what is actually available

**Model size ratio ≠ per-token traffic ratio.** mq2r is 0.661× mq4r's size but
reads **0.873×** its bytes:

| | fixed tier | routed | total |
|---|---|---|---|
| mq4r | 1030.8 MB | 534.8 MB | 1565.6 |
| mq2r | 1030.6 MB | 335.5 MB | 1366.1 |

Both carry an **identical all-MQ4 fixed tier**, and a token reads **8 of 256
experts (3.1%)**. The entire 6.4 GB size advantage lives in weights that are
almost never read (mq4r routed storage 17.1 GB vs mq2r 10.7 GB).

So a 1.5× speedup was never on the table from size. A pure-bandwidth model
predicts **201.9 / 0.873 ≈ 231 tok/s**; measured is 200.6. **That ~15% is the
real remaining gap**, and it exists because both SKUs run the same 693-dispatch
chain with the same dependency structure — token time is set by the chain, not
the bytes. This is why every routed-bit experiment measured ~0% and every win
came from launch geometry.

Beyond ~231 on pure AR requires cutting the **fixed tier** (1030 MB = 75% of
per-token traffic, identical in both SKUs) or batching tokens.

**Candidate fifth phase — concurrency.** The PM4 wait audit already computes
independence and uses it only to *delete waits*; the tape still executes
strictly in order, single-stream. The July 2026 launch-set experiment measured
RADV overlapping independent dispatches (1.29× real, 3.6× on independent sets)
where ROCm hipGraph does not (0.77×). Two-queue was tried on 2026-07-11 and
lost, but that predates dependency-derived waits.

**Precondition now met (`75fe90fbb`).** The shared-expert vs routed-gate_up pair
the audit calls disjoint was *never actually proven* on mq2r — the codebook side
was unregistered, so 40 boundaries per token (one per layer) fail-closed. mq2r
now reports `covered=692/692, resource_independent=130`, so `pm4_phase_plan()`
has real antichain width for the first time; before, 160 unmodelled boundaries
capped it. `HIPFIRE_REPLAY_PM4_QUEUES=2` is therefore worth its first honest
attempt on **mq2r specifically** — note `CERTIFIED_PM4_POLICY`
(`tools/redline/product_bench.py:75`) hardcodes `QUEUES=1` for both SKUs with no
CLI override, so per-SKU routing wants the registry entry mq2r needs anyway.
Measure with **both arm orders**: a systematic slot bias of ±0.71% was measured
on mq4r, large enough to invert a sub-1% verdict on its own.

---

## State and open items

**Committed on `a3b-lowbit-work` (`468954433`), workspace clean, 487 lib tests
pass, pushed to `hiptrx`:**

| commit | what |
|---|---|
| `a28b99f9e` | prefill guard deferred — makes the SKU servable |
| `311c53174` | batched prefill for codebook MoE — **agent-authored, unreviewed, never executed** |
| `609ea0f8d` | that made opt-in (`HIPFIRE_MOE_CODEBOOK_BATCHED=1`) |
| `358803c19` | `--state-quant` on the redline harness |
| `68a70a175` | MQ3-down R2/R4 row tile (opt-in) |
| `9bd03ac38` + `44739d429` | same for MQ2-down (opt-in) |
| `468954433` | MQ3-down ninepath port (opt-in) |
| `75fe90fbb` | codebook MoE resource contracts — **default-on, bit-exact, +1.78%** |

**Open:**

- **Nothing here is default-on.** Row tile costs **+0.40% wt2 KLD** (0.235561 →
  0.236503; 4R accumulators change FP accumulation order) — above the KLD gate's
  ~0.1% resolution. Ninepath's numerics are unvalidated for KLD/coherence.
- **`311c53174` is unvalidated** — 855 lines, agent-authored, touches the
  quantizer for no stated reason, and its grouped-WMMA kernels have never run.
  Review or revert.
- **The shadow parity gate reports `exact=False` for the *certified* mq4r route**
  under my invocation, so it proved nothing about mq2r either. Cause not found;
  `--kv-mode` has no f32 option, so KV stays quantized. **Always run mq4r as the
  control.**
- **Registry:** `mq2r` has no entry, so `tools.redline bench`'s coherence gate
  cannot pin sampling and must be `--skip-coherence` (reports are then
  `coherence_skipped`, never `valid`). An entry is a prerequisite to certifying
  the SKU.
- **PM4 admission** still keys on the `.mq4r` extension — widening it is a
  certification decision, not a one-line flip.
- GL down kernels (dead loop, no restage) unmeasured.
- gfx1100 / gfx1151 untested — everything here is gfx1201.
