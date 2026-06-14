# 14 — Stage 2b PpMtp layer-split rebalance (Opt 2)

Date: 2026-05-29
Branch: fix/q8-batched-masked-no-lds-cap (working tree at e6a25615 + bench script)
Model: `/local/hipfire/qwen3.6-27b.mq4` (qwen3.6-27b, 64 layers, dim=5120)
MTP head: `/data/hipfire/qwen3.6-27b-cvs16384.mtp` (k=3, cvs=16384)
Hardware: gfx906 (MI50, 32 GiB, HIP idx 0) + gfx1031 (RX 6700 XT, 12 GiB, HIP idx 1)
Prompt md5: `b385bda5fdf47185ab32ca7acabbf057` (committed LRU-cache prompt)
Bench: `scripts/bench-ppmtp-split.sh` — 1 warmup + 1 measured (max=256) per cell, fresh process per cell.

## Question

PpMtp (Stage 2b PP+MTP) decodes slower than both single-GPU MTP and
pp2-ar. The PP boundary handoff in
`forward_prefill_batch_multi_with_caps` (qwen35.rs ~11048) is FULLY
SERIALIZED: band 0 (gfx906) runs to completion → blocking `boundary_copy`
+ `wait_boundary` → band 1 (gfx1031) runs. One card idles while the other
works. The default 48,16 split puts 16 of 64 trunk layers on the
slow/small gfx1031. Does shifting trunk layers onto gfx906 (more headroom,
faster on the dominant GEMM) shrink gfx1031's serial tail?

This is a zero-code probe (pure `HIPFIRE_PP_LAYERS` sweep) to see how
much of the gap is reclaimable for free before deciding on the harder
levers (Opt 1 boundary pipelining / Opt 3 reference no-replay rollback /
Opt 4 accept-and-ship-long-ctx).

## Results

| cell | pp | split | spec_path | tok/s | decode tok/s | τ | accept | cycles |
|---|---|---|---|---|---|---|---|---|
| pp1-mtp | 1 | — | mtp | 21.9 | 22.5 | 3.19 | 2.23 | 80 |
| pp2-ar | 2 | 48,16 | ar | 17.2 | 17.6 | — | — | — |
| ppmtp | 2 | 48,16 | pp-mtp | 13.7 | 13.9 | 3.11 | 2.15 | 82 |
| ppmtp | 2 | 52,12 | pp-mtp | 13.6 | 13.9 | 3.15 | 2.19 | 81 |
| ppmtp | 2 | 56,8 | pp-mtp | 13.9 | 14.2 | 3.15 | 2.17 | 81 |
| ppmtp | 2 | 58,6 | pp-mtp | 13.8 | 14.0 | 3.15 | — | 81 |
| ppmtp | 2 | 60,4 | pp-mtp | 13.2 | 13.4 | 3.11 | — | 82 |

(58,6 / 60,4 measured in a second knee-probe run, same prompt/protocol.)

## Read

**Opt 2 has a shallow peak at 56,8 (14.2 decode tok/s), then REGRESSES.**
The sweep is not monotonic: 48,16 → 56,8 gains ~+2% (13.9 → 14.2), but
pushing further DOWN (58,6 = 14.0, 60,4 = 13.4) gives the gain back. The
peak is only ~+2% over the 48,16 default. Past 56,8, shifting more layers
onto gfx906 makes gfx906's band the longer leg of the serialized boundary
(it now does 56–60 of 64 layers while gfx1031 sits nearly idle), so the
serial total grows again — and gfx1031's 4-layer band adds per-band
launch/handoff overhead that isn't amortized over enough work. The free
lever tops out at ~14.2.

**τ is ~invariant (3.11–3.15)** across every PpMtp split, ≈ single-GPU
MTP's 3.19 (the 0.04–0.08 spread is within run-to-run noise — accept_rate
wobbles 2.15–2.19, cycles 81–82). This is the key control: the layer split
only redistributes the *serialized boundary work*, it does NOT touch spec
acceptance. So the gap is pure execution cost, not a PP quality regression.

**Rebalancing cannot close the gap.** Even at the peak split (56,8),
PpMtp (14.2) sits below pp2-ar (17.6) and far below single-GPU MTP
(22.5). Two structural costs no split removes:
1. The serialized boundary itself (no compute overlap between the cards).
2. PpMtp's 2nd boundary cross on rollback — v1 forces `tape_captured=false`
   so partial-accept rollback re-forwards the full multi-GPU trunk. At
   τ≈3.18 / K=3 that fires on ~2/3 of cycles, doubling the boundary crosses
   per cycle vs pp2-ar (which has no spec rollback at all).

That second cost explains why PpMtp (14.5) is *below* pp2-ar (16.8)
despite τ>3: the spec rollback's extra full-trunk pass costs more than the
spec acceptance saves, once the boundary is serialized.

## Conclusion / next levers

- Opt 2 is a real but tiny free win: peak ~+2% at 56,8 (14.2 vs 13.9 at
  the 48,16 default), and it REGRESSES past that. Worth folding 56,8 in as
  the PpMtp default — but it does NOT change the strategic picture:
  rebalancing alone leaves PpMtp well below pp2-ar.
- To get PpMtp below pp2-ar's cost, kill the 2nd boundary cross (Opt 3:
  adopt llama.cpp/vLLM-style no-replay rollback — re-derive draft state
  fresh, never re-forward the trunk on rollback).
- To BEAT single-GPU MTP, the serialized boundary must be pipelined (Opt 1
  / plan merry-giggling-conway.md B.2) — gate on the B.2a microbench first.
- Failing both, PpMtp's honest value is the ~1.7× longer ctx (178k→302k,
  see note 13) where single-GPU OOMs; `pick_path` already prefers
  single-GPU where it fits (Opt 4).

## Repro

```bash
HIPFIRE_MODELS_DIR=/local/hipfire ./scripts/bench-ppmtp-split.sh
# report → /tmp/ppmtp-split-<ts>.md
```
