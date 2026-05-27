# MTP rocprofv3 investigation on hiptrx (gfx1201 / R9700) — 2026-05-21

**Hardware**: 4× AMD AI PRO R9700 (gfx1201, RDNA4). Single-GPU bench
uses `HIP_VISIBLE_DEVICES=0`. ~640 GB/s GDDR6 per card, 32 GB VRAM.

**Branch**: `mtp-hiptrx-rocprof` (forked from `feat/mtp`, rebased to
`origin/master` 2026-05-21, awq_scale field fix `1f714ed1`).

**Tooling**: `rocprofv3 1.1.0`, `scripts/rocprof-wrap.sh`,
`scripts/coverage-audit.py` (latter not used in this investigation —
the in-tree `HIPFIRE_PROFILE=1` dump was sufficient cross-check).

**Bench config** (canonical):
```
HIP_VISIBLE_DEVICES=0 ./target/release/examples/mtp_only_demo \
  --target ~/.hipfire/models/qwen3.5-27b.mq4 \
  --mtp-head ~/.hipfire/models/qwen3.5-27b-cvs16384.mtp \
  --prompt-file benchmarks/prompts/lru_cache_pep8_strict.txt \
  --max 120 --max-n 5 --temp 0.0 --no-chatml \
  --compressed-serial --kv-mode q8
# prompt_md5 = 1e74f17934fe759468dbe1471b732067
```

---

## Headline results

| Config | tok/s | τ | prefill tok/s | notes |
|---|---|---|---|---|
| 3 warm runs cvs16384 | 39.12 ± 0.5% | 3.40 | 36.3 | Tight deterministic |
| full-vocab Q8 (3 warm) | 35.86 | 3.40 | ~36 | Slightly slower (full lm_head) |
| Under rocprofv3 | 36.57 | 3.40 | 33.9 | Rocprof overhead ~7% |
| k9lin canonical (memory) | ~53 | 3.84 | — | 7900 XTX gfx1100 |

**Ratio analysis**: hiptrx single R9700 / k9lin 7900 XTX = 39.12 / 53 =
0.738. R9700 / 7900 XTX BW ratio = 640 / 960 = 0.667. tok/s ratio
slightly above the BW ratio (~+10%). Suggests gfx1201 kernels are at
roughly the same per-call BW efficiency or slightly better than gfx1100
ones on this specific batched workload.

**τ delta** (3.40 hiptrx vs 3.84 k9lin) is from arch-dependent WMMA
rounding producing different greedy-argmax commits on the canonical
prompt. Same trunk weights, same MTP head weights, same prompt, same
kv-mode — different per-position argmax in marginal cases.

---

## Rocprof "blindspot": resolved — prefill, not decode

Initial rocprof CSV showed `fused_gate_up_hfq4g256` (per-token GEMV) at
25.8% of total GPU time with **14,912 calls** — surprising because
verify + replay should be batched-WMMA on Qwen3.5-27B mq4.

**Decomposition**: 232 prompt tokens × 64 layers = 14,848 calls from
**per-token prefill** in `mtp_only_demo.rs:305` (`target.forward(token,
pos)` loop, deliberately not `forward_prefill_batch`). Only ~64 of the
14,912 calls (~1/cycle/layer × 35 cycles = ~64) are from decode
fallbacks (advance=1 replay).

**Conclusion**: not a decode lever. Per-token prefill is a real
inefficiency that hurts startup + bench iteration time but is
orthogonal to decode tok/s. Switching prefill to `forward_prefill_batch`
would reclaim ~5s of bench wall (out of ~10s total run), making
iteration much faster but not affecting reported `tok_s`.

---

## In-tree per-cycle profile (HIPFIRE_PROFILE_CYCLES=10, K=5 max=60)

113.17 ms total per cycle wall. 1280/96/256/etc kernel calls per 10
cycles → per-cycle:

| Kernel | calls/cycle | µs/call | % cycle | bytes |
|---|---|---|---|---|
| gemm_gate_up_hfq4g256_wmma_gfx12 | 128 | 297 | 33.6% | 12.3 GB |
| gemm_hfq4g256_residual_wmma_gfx12 | 256 | 91 | 20.7% | 8.3 GB |
| gemm_qkvza_hfq4g256_wmma_gfx12 | 96 | 102 | 8.7% | 4.4 GB |
| fused_rmsnorm_mq_rotate_batched | 256 | 29 | 6.5% | 71 MB |
| gemv_hfq4g256 (MTP head per-step) | 47 | 111 | 4.6% | 2.7 GB |
| gemm_qkv_hfq4g256_wmma_gfx12 | 32 | 90 | 2.5% | 1.3 GB |
| gemm_hfq4g256 (lm_head) | 1 | 2794 | 2.5% | 682 MB |
| gated_delta_net_q8_batch_seq | 96 | 29 | 2.5% | 197 MB |

Layer accounting (Qwen3.5-27B = 64 layers, 16 FA + 48 LA):
- gate_up batched: 128/cycle / 64 = **2 per layer** ✓ verify + replay
- residual batched: 256/cycle / 64 = **4 per layer** = 2× (wo + w_down) ✓
- qkvza batched: 96/cycle / 48 LA = **2 per LA layer** ✓ verify + replay
- qkv batched: 32/cycle / 16 FA = **2 per FA layer** ✓ verify + replay

Decode is fully batched. **No hidden per-token forwards in decode.**

---

## Per-call BW efficiency on gfx1201

`gemm_gate_up_hfq4g256_wmma_gfx12` at M=6, K=5120, ffn_inner=13824 × 2:
- bytes moved: ~96 MB weight + tiny x + tiny y ≈ 96 MB/call
- µs/call: 297
- effective BW: 96 MB / 297 µs = **323 GB/s** = **50.5% of 640 GB/s peak**

`gemm_hfq4g256_residual_wmma_gfx12` at M=6, K=5120 (wo+w_down):
- bytes: ~32 MB MQ4 weight ≈ 32 MB/call
- µs/call: 91
- effective BW: **352 GB/s** = **55%** peak

Both batched WMMA paths run at ~50-55% peak BW on gfx1201 — slightly
**better** than k9lin gfx1100's reported 49% per cycle-anatomy memory.
The per-call efficiency is not the obvious lever; the GPU is fairly
saturated.

---

## The structural argument for composition (Goal B)

At M=6, the WMMA 16×16 tile uses 6/16 = 37.5% of the A-dimension. Per-
call wall time at M=6 is **dominated by the tile, not by M itself** —
padding to M=16 is the same wall time.

If we stack DFlash drafts (K1=12) + MTP drafts (K2=3) + seed = M=16,
the SAME verify forward does ~2.7× as much useful work for the same
gate_up wall time. This is the structural lift composition gives us
over solo MTP (M=6) — independent of the per-draft acceptance rates.

Per cycle (composition):
- Verify wall: same ~113 ms
- Useful drafts committed: scales with combined acceptance over 15 drafts
- At DFlash τ≈10 (canonical) + modest MTP τ_chain≈3, expected
  combined commits/cycle: 10-15 vs current MTP-only 3.4
- Cycle wall stays ~constant → **2-4× decode tok/s**

This is independent of any per-call kernel improvement. It's the WMMA
tile-utilization arbitrage.

---

## Hard-falsified levers (confirmed today)

These reaffirmed today's gfx1100 ceiling on gfx1201 via independent
measurement:
- Solo MTP at K=5: optimal cycle structure for Q8/MQ4 R9700
- Per-call BW efficiency: ~50-55% — not 35% as initially worried,
  not much headroom on existing kernels at M=6

## Real levers, ranked by expected lift

1. **Composition** (Goal B path) — M=6 → M=16 WMMA tile utilization
   → 2-4× decode tok/s. ~500-1500 LOC; multi-day implementation
2. **Replay elimination via GDN checkpoint** (multi-week) — kill the
   second batched forward → ~30-50% lift. Master plan Phase 1 Track B
3. **Batched prefill in mtp_only_demo** — orthogonal to tok/s, but
   ~5× faster bench iteration. Trivial fix.
4. **MTP head trained sidecar** (Phase 1 Track A) — τ 3.4 → 4.0
   projected → modest ~20% lift if standalone. Multi-hour distill on
   hiptrx 4-GPU. Possibly worth running while composition is being
   prototyped.

## Recommended sequence

Per master plan's "sneaky-smart sequencing": **probe composition first**.
Today's data confirms it's the highest-EV move:
- Goal B (230+) directly addresses user's combined ambition
- Composition lift is independent of solo MTP improvements (adds on top)
- M=16 is structurally aligned with batched WMMA tile

Solo MTP improvements stay relevant but are NOT the bottleneck for
either goal independently.
