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

### 3. rocprofv2 on the long-ctx Q8 attention kernel (gfx906)

rocprofv2 WORKS on gfx906 (rocprofv3 cores out). PMC group must stay
under the TCC 4-counter limit — split FetchSize/L2/etc into separate
passes. VALU-group results (CTX=16000, N=64, B=64), per-kernel:

| kernel | VALUBusy | MemUnitBusy | VALUUtilization | VGPR | wg |
|---|---|---|---|---|---|
| tile_batched (default) | **92.1%** | 76.4% | **48.6%** | 32 | 32 |
| dp4a (HIPFIRE_Q8_DP4A=1) | 73.0% | **86.7%** | 48.6% | 32 | 32 |
| tokpar (falsified) | 40.9% | 56.9% | **99.8%** | 44 | 256 |

**Diagnosis:**
- tile_batched is **VALU-bound (92%)** — confirms issue/compute-bound,
  not memory. dp4a cuts VALUBusy 92→73% (sdot4 replaces scalar FMAs as
  designed) and shifts the bottleneck toward memory (MemUnitBusy 76→87%),
  hence its 1.25× and no more.
- **VALUUtilization = 48.6%** on both — the smoking gun. These kernels run
  on gfx906's **wave64** but are written wave32-style (`__launch_bounds__(32)`,
  wg=32, `__shfl_xor` over 32 lanes) → **the upper 32 lanes of every
  64-lane wave are idle.** ~2× VALU waste. tokpar (wg=256, full waves)
  hits 99.8% VALUUtil, proving the headroom is real.

**Next lever (measured, high-confidence): a true wave64 rewrite** of the
dp4a/tile kernel — 64 active lanes, `__shfl` over 64, head_dim split
across 64 lanes. Closes the 48→~96% utilization gap → potentially ~2×
VALU throughput. Matches the project's "wave32 kernel wastes upper 32
lanes on wave64-native gfx906" pattern (precompile registry has
`*_wave64` variants for exactly this on the GEMVs).

### 4. Wave64 dp4a attention — built, mechanism-confirmed, 1.5× over default

`attention_flash_q8_0_dp4a_wave64.gfx906.hip`: 64 active lanes, 4 dims/lane
= one sdot4/token, reduce over 64. Gated `HIPFIRE_Q8_DP4A_W64=1`,
head_dim%64==0.

VERIFY 3.741e-3 vs tile_batched (identical to wave32 dp4a — same math).
Timing (median of 5):

| | n=64/16k | n=512/16k | n=256/32k |
|---|---|---|---|
| TILE (default) | 18.61 | 148.84 | 151.49 ms |
| dp4a w32 | 14.92 | 117.90 | 119.52 ms |
| **dp4a w64** | **12.68** | **100.46** | **101.92 ms** |

- w64 vs TILE: **1.47–1.49×**; vs w32 dp4a: 1.17–1.18×; vs per-position: 3.0–3.3×.

rocprofv2 confirms the mechanism:

| kernel | VALUUtil | MemUnitBusy | VGPR |
|---|---|---|---|
| tile_batched | 48.6% | 76.4% | 32 |
| dp4a w32 | 47.7% | 86.5% | 32 |
| **dp4a w64** | **93.3%** | **91.2%** | 24 |

VALUUtilization 48→93% — the idle-upper-32-lanes waste is gone, exactly
the predicted fix. The 1.17× (not 2×) is because MemUnitBusy rose to
91.2% — **the kernel is now memory-bound**, so the wave64 fix consumed
the VALU slack and memory is the next wall. VGPR dropped 32→24 (more
occupancy headroom).

### 5. Memory-wall diagnosis: BATCH reload dominates (not GQA, not load width)

wave64 is memory-bound (91% MemUnitBusy). Investigated the DRAM traffic:

- **Load width: already optimal.** Disassembly shows the compiler already
  coalesces the 4 K codes into a single `global_load_dword offset:2`. No
  manual-packing win. (The K-code ptr is kb+2+bj → 2-aligned, so no clean
  dwordx4, but the dword load is already there.)
- **GQA reload (4×): NOT the dominant factor.** Predicted 133 MiB at
  N=64; measured FetchSize was 2123 MiB — 16× higher.
- **BATCH reload (N×): THE dominant factor — confirmed by scaling.**
  rocprofv2 FetchSize: N=16 → 547 MB, N=64 → 2177 MB = exactly 4×
  (linear in batch). The grid [heads, tiles, BATCH] gives each query row
  its own block that re-streams the tile's K/V from DRAM. K/V is read N
  times, once per query row.

**Batch K/V reuse — PROBED, NOT a clear win (do not build blind).**
Cheap rocprofv2 L2CacheHit probe before committing to the LDS-staging
rewrite: **L2CacheHit = 78%** on all three Q8 attention kernels (tile,
dp4a w32, dp4a w64). So the batch reuse is ALREADY mostly caught by L2 —
each tile's K/V (~34 KB/kv-head) is small enough that consecutive batch
rows re-hit it in the 16 MB L2. The 2123 MiB "FetchSize" is fetched-to-L2
traffic; only ~22% misses to DRAM. LDS staging would convert L2 reads →
LDS reads (relieve the 91% MemUnitBusy), BUT:
- the win is bounded (L2 already does most reuse, not a 4×/N× DRAM cliff),
- staging 34 KB/WG drops occupancy to 1 WG/CU, which HURTS latency-hiding
  on a memory-bound kernel (LDS study's own caveat).
Net is genuinely uncertain → high risk of a falsified-tokpar-style no-op
or regression. **Decision: do NOT build the LDS-staging rewrite blind.**
If pursued, must A/B it against the current wave64 with rocprofv2 proving
MemUnitBusy drops AND wall-time improves despite the occupancy hit.
The cheap probe (1 counter pass) saved the speculative rewrite.

### Next

- NIAH-gate wave64 at long ctx (DONE — PASS), committed 8c48c089.
- Next ceiling is MEMORY (91% MemUnitBusy). Per the gfx906 KV-read study
  (skyne98): HSD layout (we're already HSD-ish) + x4 (dwordx4/128-bit)
  vectorized K/V loads ≈ 7% mem win. The wave64 kernel reads 4 i8 + fp16
  scale per lane as scalar bytes; coalescing the 4 K codes into one dword
  load is the directional follow-up. Latency-hiding (issue N loads before
  wait) is a principle only — gfx906 rejects s_clause/s_waitcnt_depctr/
  s_delay_alu, so it's compiler-dependent and UNMEASURED; do NOT stack it
  speculatively (separate rev if pursued).
- RDNA2 pflash compress optimality + fwht3 vs q8 multi-needle scoring parity.
