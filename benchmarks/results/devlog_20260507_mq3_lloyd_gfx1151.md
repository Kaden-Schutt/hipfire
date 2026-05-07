# Dev log 2026-05-07 — MQ3-Lloyd fast variants enabled on gfx1151 (Strix Halo APU)

**Branch:** `feat/mq3-lloyd-gfx1151` off `master @ 97ee9be`.
**Target hardware:** AMD Ryzen AI MAX+ 395 / Radeon 8060S
(gfx1151, RDNA3.5, ROCm 7.12).

## Change

Add `"gfx1151"` to all 5 MQ3-Lloyd fast-arm matchers in
`crates/rdna-compute/src/kernels.rs`:

- `gemv_mq3g256_lloyd_for_arch`
- `gemv_mq3g256_lloyd_residual_for_arch`
- `fused_gate_up_mq3g256_lloyd_for_arch`
- `fused_qkvza_mq3g256_lloyd_for_arch`
- `fused_qkv_mq3g256_lloyd_for_arch`

Plus a small fix to `crates/hipfire-runtime/examples/perplexity.rs`
to take `&mut hfq` for the `qwen35::load_weights` signature that
changed in master via PR #147 (`feat(loader+quantizer): UMA-safe
loader…`) and to print NLL/tok at 10-decimal precision so subtle
fp32-reorder drift is visible (master's 6-decimal print silently hid
the multi-acc drift surfaced in the MQ4-Lloyd investigation).

Coverage parity: gfx1151 now matches gfx1100/1101/1102's deployment
shape — all 5 fast variants firing during inference.

## Rationale: parity with the gfx1100 deployment, not a separate gating

When this enablement was first attempted (during the MQ4-Lloyd
investigation), I gated to **basic-GEMV-only** on gfx1151 because the
all-5 enablement produced a 0.9% PPL drift on Qwen3.5-9B.

The follow-up [`findings/mq4-lloyd-multiacc-investigation.md`](../../findings/mq4-lloyd-multiacc-investigation.md)
root-caused that drift to fp32 reorder noise from the K4
multi-accumulator pattern `(acc0+acc1)+(acc2+acc3)`, compounded
across ~200 GEMVs/token × 2K tokens × softmax non-linearity. **A
follow-up gfx1100 measurement (2026-05-07) confirmed the same
multi-acc drift reproduces on gfx1100** — the issue is universal
across the gfx11 family, not arch-specific.

**Implication: master's existing MQ3-Lloyd deployment on
gfx1100/1101/1102 has been carrying this latent compounding drift
since PR #115 / #181 landed.** It hasn't surfaced because no one
A/B'd PPL at 10-decimal NLL/tok precision; coherence and quality
gates are well within the drift envelope (0.9 % PPL is far below the
soft-flag thresholds in `coherence-gate-dflash.sh`).

Given that, gating gfx1151 to GEMV-only would be **artificially
inconsistent** with the gfx1100 deployment shape — both arches share
the same kernel family, the same drift, and the same coherence
behavior. Enabling all 5 on gfx1151 gives **deployment parity** with
gfx1100 and lets gfx1151 users get the same speedup the gfx1100
deployment ships with today.

Closing the drift universally is a separate follow-up: port MQ3-Lloyd
to single-accumulator (mirroring the production MQ4-Lloyd kernels
that already use single-acc and are byte-equal at 10dp). That's not
in scope for this small enablement PR.

## Conformance / drift snapshot

Qwen3.5-9B / `benchmarks/calib/calib-5m.txt` / ctx=2048 / warmup=8 /
offset=0 / gfx1151:

```
fast (default — all 5 MQ3-Lloyd matchers gfx1151):  NLL/tok = 3.2199992858  PPL = 25.0281
slow (HIPFIRE_LLOYD_FORCE_BASELINE=1):              NLL/tok = 3.2110607378  PPL = 24.8054
                                                                  Δ = +0.0089 NLL = +0.9% PPL
```

This is the documented universal MQ3-Lloyd multi-acc latent drift,
not a new gfx1151-specific issue. master's gfx1100 deployment carries
the same drift envelope (confirmed empirically 2026-05-07 by user).

For comparison, basic-GEMV-only enablement on gfx1151 produced
byte-equal PPL at 10dp:

```
GEMV-only enablement (residual + fused stay slow): NLL/tok = 3.2110607378  byte-equal
```

That's the "correctness-pure but smaller speedup" alternative this
PR rejects in favor of deployment parity — see `Performance` below.

## Performance — gfx1151

`bench_qwen35_mq4 <model> --prefill 128 --prefill-runs 3 --warmup 8 --gen 100`,
median of 3 prefill runs, single 100-token gen. Within-session noise
band on this host is ~±10–15% per
`docs/methodology/perf-benchmarking.md`.

### Full-coverage (this PR)

| Model                   | Mode | Prefill (median, tok/s) | Decode (gen, tok/s) | Effective BW |
|-------------------------|------|------------------------:|--------------------:|-------------:|
| `qwen3.5-4b.mq3-lloyd`  | slow |                    35.5 |                34.6 |    72.6 GiB/s |
| `qwen3.5-4b.mq3-lloyd`  | **fast** |                **74.5** |            **67.5** |  **141.7 GiB/s** |
| `qwen3.5-9b.mq3-lloyd`  | slow |                    18.3 |                18.2 |    77.2 GiB/s |
| `qwen3.5-9b.mq3-lloyd`  | **fast** |                **49.1** |            **46.4** |  **197.3 GiB/s** |

**Headline: ~2.0× decode on 4B, ~2.5× decode on 9B.** Effective BW
roughly doubles too — gfx1151 has shared LPDDR5x (~250 GB/s peak
shared with CPU); 197 GiB/s on 9B-fast represents ~78% of that peak
(very respectable).

### GEMV-only (alternative, rejected in favor of deployment parity)

| Model                   | Mode | Decode (gen, tok/s) |
|-------------------------|------|--------------------:|
| `qwen3.5-4b.mq3-lloyd`  | fast |                34.6 (+0.0%) |
| `qwen3.5-9b.mq3-lloyd`  | fast |                19.7 (+8.2%) |

GEMV-only sees minimal speedup because only the `wo` output projection
(per FA layer) plus the single lm_head GEMV per token routes through
the fast kernel. The bulk of decode (gate+up, QKV, QKVZA fused
projections) stays on slow generic. Going from "GEMV-only" to
"all 5" delivers the headline 2-2.5× win.

### gfx1100 reference (for comparison)

`docs/BENCHMARKS.md` documents Qwen3.5-9B MQ4 (uniform) at 132 tok/s
decode on gfx1100. PR #181's devlog
(`benchmarks/results/devlog_20260506_lloyd_mq4_extension.md` —
referenced by the proposal devlog) puts MQ3-Lloyd 9B on gfx1100
around 121.7 tok/s after the perf chain landed. gfx1151's 46.4 tok/s
is ~2.6× slower than gfx1100, consistent with shared LPDDR5x
(~250 GB/s) vs 7900 XTX GDDR6 (960 GB/s) memory-bandwidth ratio.

## Bench-host quirks

- gfx1151 SIGSEGVs during ROCm teardown after metrics print.
  Same as documented in
  `devlog_20260507_mq4_lloyd_implementation.md`. Doesn't affect
  bench numbers (printed before teardown). Exit code 139 on
  otherwise-successful runs.
- `bench_qwen35_mq4` is general-purpose — works on MQ3-Lloyd files
  too; dispatch is dtype-driven from per-tensor metadata. Naming is
  historical (the binary was added during MQ4 bring-up).

## Repro

```sh
source scripts/rocm-env.sh
cargo build --release -p hipfire-runtime --example bench_qwen35_mq4 --features deltanet
cargo build --release -p hipfire-runtime --example perplexity
source scripts/gpu-lock.sh && gpu_acquire "mq3-lloyd gfx1151"

# Performance:
./target/release/examples/bench_qwen35_mq4 \
  ~/.hipfire/models/qwen3.5-9b.mq3-lloyd \
  --prefill 128 --prefill-runs 3 --warmup 8 --gen 100

HIPFIRE_LLOYD_FORCE_BASELINE=1 ./target/release/examples/bench_qwen35_mq4 \
  ~/.hipfire/models/qwen3.5-9b.mq3-lloyd \
  --prefill 128 --prefill-runs 3 --warmup 8 --gen 100

# PPL drift envelope (10-decimal NLL/tok printf surfaces the latent drift):
./target/release/examples/perplexity \
  ~/.hipfire/models/qwen3.5-9b.mq3-lloyd \
  benchmarks/calib/calib-5m.txt --ctx 2048 --warmup 8 --offset 0
HIPFIRE_LLOYD_FORCE_BASELINE=1 ./target/release/examples/perplexity \
  ~/.hipfire/models/qwen3.5-9b.mq3-lloyd \
  benchmarks/calib/calib-5m.txt --ctx 2048 --warmup 8 --offset 0
```

## Cross-references

- `findings/mq4-lloyd-multiacc-investigation.md` — multi-acc drift
  root-cause + universal-on-gfx11 confirmation. (This findings doc
  lives on the MQ4-Lloyd branch; will be on master after that PR
  lands.)
- `benchmarks/results/devlog_20260507_mq4_lloyd_implementation.md` —
  sibling MQ4-Lloyd implementation devlog where the multi-acc drift
  was first surfaced.
- `docs/BENCHMARKS.md` — gfx1100 reference numbers for comparable models.

## Follow-up: universal MQ3-Lloyd → single-acc port

A future follow-up that ports the 5 MQ3-Lloyd fast kernels from
multi-accumulator to single-accumulator (mirroring the production
MQ4-Lloyd kernels) would close the latent ~0.9% PPL drift universally
across gfx1100/1101/1102/1151. Not blocking — current MQ3-Lloyd
deployments (master and this PR) carry the same well-understood drift
envelope, far below the coherence-gate thresholds.
