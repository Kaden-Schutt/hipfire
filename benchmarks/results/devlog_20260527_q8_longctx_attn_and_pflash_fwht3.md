# Devlog 2026-05-27 — Q8 long-ctx attention accel + PFlash fwht3 default

Branch `fix/q8-batched-masked-no-lds-cap`. gfx906 (MI50, target/dev0) +
gfx1031 (RX 6700 XT, drafter/dev1) hetero box.

## Context

This branch already carried (earlier sessions):
- `abd95245` — no-LDS-cap batched-masked Q8 flash attention (the cliff
  fix: Q8 prefill >15k no longer overflows LDS / per-position fallback).
  NIAH-PASS @ 21551 tok. tile_batched kernel + reuse of asym reduce.
- `59473843` — token-parallel Q8 FA experiment (falsified, off by default).
- `7ced6ca8` — gfx906 dp4a Q8 FA (v_dot4_i32_i8 Q·K), 1.24-1.27× over
  tile_batched, behind `HIPFIRE_Q8_DP4A=1` (default off).

## Today

### 1. Profiled the real serve prefill — overturned an earlier assumption

`profile_prefill_qwen35` on gfx906, B=13078 (the serve's compressed
length), Q8 KV, 27B AWQ. Per-kernel attribution:

| kernel | % prefill |
|---|---|
| **attention_q8_0_kv_batched** | **79.9%** (12.8 GiB/s ≈ 1% peak) |
| gemm_gate_up_hfq4g256_mmq_gfx906 | 10.3% |
| gemm_hfq4g256_residual_mmq_gfx906 | 6.3% |
| gemm_qkv_hfq4g256_mmq_gfx906 | 0.8% |
| (everything else) | <2% |

**Q8 attention is 80% of gfx906 prefill** — NOT the GEMMs. This falsifies
the earlier "attention isn't the e2e bottleneck, GEMMs dominate" claim.
The GEMMs already use the merged dp4a MMQ kernels (PRs #276/#281) and are
fast. The scalar batched-masked Q8 attention is the dominant cost and the
top lever — which means the dp4a Q8 *attention* work on this branch is
more impactful than previously credited.

### 2. PFlash drafter-KV fwht3: 1.80× compress win, wired as long-ctx default

The drafter compresses the FULL source (43286 tok) on gfx1031 — far above
its own 15k Q8 LDS cliff → per-position fallback. `HIPFIRE_PFLASH_DRAFTER_KV=fwht3`
(shipped, no LDS cap) A/B on the live serve, byte-identical 43k prompt:

| drafter KV | compress 43286→13078 | speedup |
|---|---|---|
| q8 (default) | 15257 ms | 1.0× |
| fwht3 | 8484 ms | **1.80×** |

Same kept-token count, same correct needle output. Wired as the default:
`DrafterKvMode::resolve(arch, max_kv_seq)` → fwht3 when arch ∈
{gfx906, gfx103x} AND max_kv_seq > 15000, else Q8. Env var still wins.

**Quality scope (important):** fwht3 here replaces ONLY the drafter's
*scoring* KV (throwaway, on dev1, used to pick which ~30% of source
tokens to keep). The TARGET's KV — what determines output — stays Q8,
untouched. So this is quality-safe for the served output by construction;
the only perturbation is *which source tokens get selected*. fwht3 is the
project's preferred scorer format anyway (Givens noise biases the
head-dim cosine; fwht's distributed noise doesn't). NOT yet proven:
kept-SET identity (only kept-count matched) and multi-needle scoring
parity — flagged for the measurement phase.

### Next (in progress)

- rocprofv2 (reported working on gfx906/gfx1031, unlike rocprofv3 which
  cores out) to refine the long-ctx Q8 attention kernel — it's 80% of
  prefill at ~1% peak BW, clearly issue-bound; want occupancy/VALU/issue
  attribution to guide the next kernel rev.
- Measure whether RDNA2 pflash compress is optimal even with fwht3
  (compress is still 8.5s for 43k → is the fwht3 attention itself now
  the floor, or is there more?), plus a multi-needle scoring-quality
  check on fwht3 vs q8.
