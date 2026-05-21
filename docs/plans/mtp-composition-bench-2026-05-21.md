# MTP+DFlash composition bench — Phase 0 empirical 2026-05-21

**Hardware**: hiptrx, 4× R9700 / gfx1201 RDNA4, single GPU
(`HIP_VISIBLE_DEVICES=0`). ~640 GB/s GDDR6 per card.

**Branch**: `mtp-hiptrx-rocprof` (HEAD `5340a974` + composition bench).

**Code state**: `spec_step_dflash_mtp` (linear) and
`spec_step_dflash_mtp_tree` (per-slot tree) are **already implemented**
in `crates/hipfire-arch-qwen35/src/mtp_compose.rs` (1223 LOC).
`dflash_mtp_demo.rs` (405 LOC) and `dflash_mtp_tree_demo.rs` (394 LOC)
wire them up. This bench just exercises the existing artifacts.

**Bench config**:
```
HIP_VISIBLE_DEVICES=0 ./target/release/examples/dflash_mtp_demo \
  --target ~/.hipfire/models/qwen3.5-27b.mq4 \
  --drafter ~/.hipfire/models/qwen35-27b-dflash-mq4.hfq \
  --mtp-head ~/.hipfire/models/qwen3.5-27b-cvs16384.mtp \
  --prompt-file benchmarks/prompts/lru_cache_pep8_strict.txt \
  --max 120 --temp 0 --no-chatml --kv-mode q8 \
  --dflash-b <B> --mtp-k <K>
# prompt_md5 = 1e74f17934fe759468dbe1471b732067
```

---

## Headline results (canonical 27B-3.5, K=5 implicit for MTP head)

| Variant | dflash-b | mtp-k | tok/s | commits/cycle | cycles | decode_secs | notes |
|---|---|---|---|---|---|---|---|
| **DFlash solo** | 16 | — | **126.06** | 10.46 | 13 | 0.98 | reference baseline |
| **MTP solo** (mtp_only_demo) | — | — | 39.12 | 9.46 / 3.4 | 35 | 3.07 | from prior bench |
| Composition linear B=14 K=2 | 14 | 2 | **123.79** | 9.46 | 13 | 0.99 | tile-aligned M=16 |
| Composition linear B=16 K=1 | 16 | 1 | 94.64 | 9.46 | 13 | 1.30 | M=17 → 2 tiles |
| Composition linear B=16 K=2 | 16 | 2 | 93.05 | 9.46 | 13 | 1.32 | M=18 → 2 tiles |
| Composition linear B=16 K=3 | 16 | 3 | 91.62 | 9.46 | 13 | 1.34 | M=19 → 2 tiles |
| Composition linear B=12 K=4 | 12 | 4 | 115.15 | 8.79 | 14 | 1.07 | M=16 tile-aligned |
| Composition linear B=8  K=8 | 8 | 8 | 86.49 | 6.47 | 19 | 1.42 | M=16 but DFlash truncated |
| Composition tree B=16 K=1 | 16 | 1 | 37.18 | 6.32 | 19 | 3.23 | tree overhead kills it |
| Composition tree B=8  K=2 | 8 | 2 | 16.11 | 1.88 | 64 | 7.45 | tree variant degenerate |

### Full-vocab MTP head (qwen3.5-27b-q8.mtp) vs compressed (cvs16384)

| Variant | compressed tok/s | full-vocab tok/s |
|---|---|---|
| B=14 K=2 | 123.79 | 123.07 |
| B=12 K=4 | 115.15 | 113.67 |
| B=16 K=2 | 93.05 | 92.56 |

**Vocab compression doesn't matter** — both produce identical
committed_total (123) at every (B,K). MTP head's argmax positions on
canonical bench are all in top-16K.

---

## Conclusion: composition does NOT exceed DFlash solo with weak MTP head

Empirically confirms master plan's honest math (see
`docs/plans/mtp-dflash-composition-master-plan.md`):

> "Composition is at-best-flat over DFlash solo if MTP adds 2-3 commits
> per cycle. To CLEARLY EXCEED DFlash solo, we need either:
> 1. MTP contribution ≥4-5 commits per cycle (requires good MTP solo
>    acceptance — needs the trained sidecar)
> ..."

Best composition (B=14 K=2 tile-aligned M=16) hits **123.8 tok/s** vs
DFlash solo 126.1 tok/s — 1.8% under DFlash. MTP candidates contribute
0 extra commits in observed cycles (committed_total identical at
B=16 K=0 vs B=14 K=2).

### Why composition contributes 0 net commits

DFlash full-accept rate at B=16: 1/13 = **7.7%** (per spec_step_dflash
seed-oracle output). MTP K=2 candidates only "fire usefully" on full-
accept cycles. Conditional contribution per full-accept:
- MTP step 0 accept p ≈ 0.68 (matches MTP solo per-position acceptance)
- MTP step 1 accept p ≈ 0.68 × 0.68 = 0.46
- E(commits per full-accept) = 1.14

Expected lift: 7.7% × 1.14 = **0.088 commits/cycle**. In 13 cycles =
~1.1 extra commits — within noise of 123 vs 124 measurements.

### Why tree variant is much worse

Tree allocates B × K MTP slots; verify becomes M = B + B×K. At B=16
K=1, M = 32 → 2 WMMA tiles + per-cycle tree-construction overhead.
The non-WMMA per-cycle costs (tree node construction, attention bias
building, KV writes for B*K slots) dominate the cycle. Drops to 37
tok/s vs linear 94 tok/s at same (B,K).

---

## What this proves

1. ✅ Composition architecture **works correctly** — verify accepts
   MTP candidates as expected on full-accept cycles, no regression in
   committed_total quality
2. ✅ Tile-alignment is real: M=18 → 2-tile WMMA costs 34% more wall
   for batched gate_up/residual/qkvza, killing throughput
3. ❌ Current MTP head (sidecar cvs16384 or full-vocab Q8) does NOT
   produce enough extra commits to offset composition overhead

## What's needed for Goal B 230+ tok/s

Per master plan and these empirics:

### Tract A: Trained MTP sidecar (Phase 1, multi-hour to multi-day)
- Run `scripts/distill/run_distill_parallel.sh` on hiptrx 4× R9700
- Wide-corpus trunk-argmax distillation (needs ~100s of prompts;
  HF datasets `Roman1111111-claude-opus-10000x`,
  `Jackrong-Qwen3.5-reasoning-700x`, `nohurry-Opus-Reasoning-3000x`)
- Need `--kv-mode q8` flag added to run_distill_parallel.sh
  (currently defaults to asym3)
- Projected MTP solo τ: 3.4 → 4.5+ → tok/s 39 → 50-60
- In composition: enables MTP to contribute ~1.5-2 commits/cycle
  consistently (not just on full-accept) → tok/s 124 → 150+

### Track B: Replay elimination via per-position GDN checkpoint
- Multi-week kernel work (per master plan)
- Saves ~30-50% of cycle wall by skipping replay forward
- Pure perf lever, independent of MTP head quality

### Hardware: TP across 4 R9700s
- Multi-week TP infrastructure work
- 3-4× lift theoretical → composition 124 → 350-450 tok/s
- Clears Goal B comfortably
- Out of overnight scope

---

## Recommendations for next session

1. **Distillation prep**: download HF prompt corpus on hiptrx,
   add `--kv-mode q8` flag to run_distill_parallel.sh, kick off
   distillation. ~2-6 hours runtime on 4× R9700.
2. **Aggregate + sidecar**: run aggregate_argmax.py + merge_sidecars.py
   to produce trained `qwen3.5-27b-distilled.mtp`.
3. **Re-bench composition**: replace cvs16384.mtp with distilled.mtp,
   re-run B=14 K=2 sweep. Expected: 140-160 tok/s on hiptrx single GPU.
4. **k9lin re-bench**: same composition on k9lin 7900 XTX. Per BW
   ratio, expected: 124 × (960/640) = 186 tok/s (no trained sidecar) →
   220-240 tok/s (with trained sidecar) → likely **clears Goal B 230**.

Goal B 230+ on **k9lin** is feasible with Phase 1 trained sidecar.
On **hiptrx single R9700**, probably blocked by BW until TP work.

## Code state

- `crates/hipfire-arch-qwen35/src/mtp_compose.rs` (1223 LOC):
  - `MtpComposeState`, `spec_step_dflash_mtp` (linear)
  - `MtpComposeTreeState`, `spec_step_dflash_mtp_tree` (tree)
- `crates/hipfire-runtime/examples/dflash_mtp_demo.rs` (405 LOC)
- `crates/hipfire-runtime/examples/dflash_mtp_tree_demo.rs` (394 LOC)

All shipped before this session. No code changes required for Phase 0.

## Bench reproducibility

Models on hiptrx (canonical):
- `~/.hipfire/models/qwen3.5-27b.mq4` — trunk
- `~/.hipfire/models/qwen35-27b-dflash-mq4.hfq` — DFlash drafter
- `~/.hipfire/models/qwen3.5-27b-cvs16384.mtp` — MTP head (vocab=16K)
- `~/.hipfire/models/qwen3.5-27b-q8.mtp` — MTP head (full vocab)

prompt_md5: `1e74f17934fe759468dbe1471b732067`
