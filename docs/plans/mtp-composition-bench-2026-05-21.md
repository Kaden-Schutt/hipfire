# MTP+DFlash composition bench — Phase 0 empirical 2026-05-21

**Hardware tested:**
- hiptrx: 4× R9700 / gfx1201 RDNA4, single GPU (`HIP_VISIBLE_DEVICES=0`)
- k9lin: 7900 XTX / gfx1100 RDNA3 single GPU
- BW ratio (R9700/7900 XTX): 640/960 = 0.667

**Branch**: `mtp-hiptrx-rocprof` (HEAD `7a5a4f8f` + bench).

**Code state**: `spec_step_dflash_mtp` (linear) and
`spec_step_dflash_mtp_tree` (per-slot tree) are **already implemented**
in `crates/hipfire-arch-qwen35/src/mtp_compose.rs` (1223 LOC).
`dflash_mtp_demo.rs` (405 LOC) and `dflash_mtp_tree_demo.rs` (394 LOC)
wire them up. This bench just exercises the existing artifacts.

**Bench config**:
```
HIPFIRE_DPM_WARMUP_SECS=10 (or 1 throwaway run) \
./target/release/examples/dflash_mtp_demo \
  --target ~/.hipfire/models/qwen3.5-27b.mq4 \
  --drafter ~/.hipfire/models/qwen35-27b-dflash-mq4.hfq \
  --mtp-head <path-to-cvs16384.mtp-or-q8.mtp> \
  --prompt-file benchmarks/prompts/lru_cache_pep8_strict.txt \
  --max 120 --temp 0 --no-chatml --kv-mode q8 \
  --dflash-b <B> --mtp-k <K>
# prompt_md5 = 1e74f17934fe759468dbe1471b732067
```

---

## Headline results (canonical 27B-3.5, K=5 internal for MTP head)

### hiptrx (single R9700 gfx1201)

| Variant | dflash-b | mtp-k | tok/s | commits/cycle | cycles |
|---|---|---|---|---|---|
| **DFlash solo** (dflash_spec_demo) | 16 | — | **126.06** | 10.46 | 13 |
| **MTP solo** (mtp_only_demo) | — | — | 39.12 | 3.40 | 35 |
| Composition linear B=14 K=2 | 14 | 2 | **123.79** | 9.46 | 13 |
| Composition linear B=15 K=1 | 15 | 1 | not run | | |
| Composition linear B=16 K=1 | 16 | 1 | 94.64 | 9.46 | 13 |
| Composition linear B=16 K=2 | 16 | 2 | 93.05 | 9.46 | 13 |
| Composition linear B=12 K=4 | 12 | 4 | 115.15 | 8.79 | 14 |
| Composition linear B=8 K=8 | 8 | 8 | 86.49 | 6.47 | 19 |
| Composition tree B=16 K=1 | 16 | 1 | 37.18 | 6.32 | 19 |
| Composition tree B=8 K=2 | 8 | 2 | 16.11 | 1.88 | 64 |

### k9lin (7900 XTX gfx1100)

| Variant | dflash-b | mtp-k | tok/s | commits/cycle | cycles |
|---|---|---|---|---|---|
| **DFlash solo** (dflash_spec_demo) | 16 | — | **181** | 11.25 | 12 |
| **MTP solo** (mtp_only_demo) | — | — | 44 | 3.4 | 35 |
| Composition linear B=15 K=1 | 15 | 1 | 159.9 | 9.46 | 13 |
| Composition linear B=14 K=2 | 14 | 2 | **159.3** | 9.46 | 13 |
| Composition linear B=13 K=3 | 13 | 3 | 145.4 | 8.79 | 14 |
| Composition linear B=12 K=4 | 12 | 4 | 144.4 | 8.79 | 14 |
| Composition linear B=10 K=6 | 10 | 6 | 133.0 | 8.2 | 15 |
| Composition linear B=8 K=8 | 8 | 8 | 109.5 | 6.83 | 18 |
| Composition tree B=12 K=1 | 12 | 1 | 22.6 | 2.18 | 55 |

### Full-vocab MTP head (qwen3.5-27b-q8.mtp) vs compressed (cvs16384) on hiptrx

| Variant | compressed tok/s | full-vocab tok/s |
|---|---|---|
| B=14 K=2 | 123.79 | 123.07 |
| B=12 K=4 | 115.15 | 113.67 |
| B=16 K=2 | 93.05 | 92.56 |

**Vocab compression doesn't matter** — both produce identical
committed_total at every (B,K). MTP head's argmax positions on
canonical bench are all in top-16K.

---

## Conclusion: composition does NOT exceed DFlash solo with weak MTP head

### Why composition contributes 0 net commits

Both devices show the same pattern:
- committed_total invariant across (B,K) variants at fixed max=120
- cycles vary only with B (DFlash provides the commits)
- MTP candidates only "fire usefully" on DFlash full-accept cycles

DFlash full-accept rate at B=16: ~7-8% (per spec_step_dflash seed-oracle).
Conditional MTP contribution: ~0.1 commits per cycle on average.
This is within measurement noise of "0".

### Why tree variant is much worse

Tree allocates B × K MTP slots; verify becomes M = B + B×K. At B=16
K=1, M = 32. Per-cycle tree-construction overhead + 2-3 WMMA tiles
dominates. Drops 3-7× vs linear at same (B,K).

### Master plan honest math, vindicated

Per `docs/plans/mtp-dflash-composition-master-plan.md`:
> "Composition is at-best-flat over DFlash solo if MTP adds 2-3 commits
> per cycle. To CLEARLY EXCEED DFlash solo, we need either:
> 1. MTP contribution ≥4-5 commits per cycle (requires good MTP solo
>    acceptance — needs the trained sidecar)
> 2. Eliminate the extra verify+overhead (replay elim + fused-verify
>    kernels — multi-week)
> 3. Reuse DFlash's verify completely — MTP just steals from DFlash's
>    bonus slot, not chaining off DFlash's last draft."

Phase 0 empirics show Option 1 is the gating lever. With current MTP head
(~68% per-position acceptance), MTP candidates can't reliably extend
beyond DFlash's full-accept chain. Need ~85%+ per-position to consistently
add commits in the 7-8% of full-accept cycles, OR need composition design
that fires MTP at interior positions where DFlash is uncertain (out of scope
for Phase 0).

---

## Today's baseline drift vs prior memory

Today (2026-05-21, master `97747374` tokenizer fix rebased):

| Metric | Memory baseline | Today | Drift |
|---|---|---|---|
| MTP solo k9lin (mtp_only_demo) | 53 tok/s | 44 tok/s | -17% |
| DFlash solo k9lin (dflash_spec_demo) | 199 tok/s (CLAUDE.md) | 181 tok/s | -9% |
| DFlash τ k9lin | 10.36 | 9.25 | -11% |

Possible causes:
- Tokenizer fix (`97747374`) changed prompt tokenization → different argmax
  → different τ → different tok/s
- Recent merges from master may have shifted kernel selection
- DPM/cache state different across days

Not investigated tonight (orthogonal to composition findings). Worth a
git bisect against `cf449fcd` (pre-rebase HEAD) on a future session.

---

## What this proves

1. ✅ Composition architecture **works correctly** — `spec_step_dflash_mtp`
   linear is byte-exact correct; verify accepts MTP candidates as
   expected on full-accept cycles, no regression in committed_total
2. ✅ Tile-alignment is real and significant: M=18 → 2-tile WMMA costs
   34% more wall for batched gate_up/residual/qkvza, killing throughput
3. ❌ Current MTP head (sidecar cvs16384 or full-vocab Q8) does NOT
   produce enough extra commits to offset composition overhead — Goal B
   blocked on stronger MTP head
4. ⚠ Today's MTP solo and DFlash solo baselines are below prior memory
   numbers (~10-17% lower) — possible regression worth bisecting

## What's needed for Goal B 230+ tok/s

### Track A: Trained MTP sidecar — FALSIFIED tonight as goal-clearing lever

**Pipeline executed end-to-end on 4× R9700:**
1. Synthesized 249 diverse prompts (Python stdlib + hipfire Rust source +
   English/code/dialogue/QA), ~726 chars mean
2. Ran `scripts/distill/run_distill_parallel.sh ... --kv-mode q8` —
   249/249 prompts complete in ~22 min wall, 70,313 tokens emitted,
   5,422 unique trunk-argmax tokens
3. `aggregate_argmax.py` → v2 sidecar JSON (top-16384 covers 100% of
   trunk's actual emit distribution on the corpus)
4. `merge_sidecars.py` rank-weighted combined v1 (canonical-corpus
   frequency) + v2 (trunk-argmax distill) → merged sidecar
5. `mtp_extract --vocab-sidecar /tmp/merged_sidecar.json` → new
   `qwen3.5-27b-distilled.mtp` (258 MiB MQ4)

**Result: distilled sidecar gives 0.0% lift on canonical bench:**

| Variant | Baseline (cvs16384) | Distilled (merged v1+v2) | Lift |
|---|---|---|---|
| MTP solo (hiptrx single R9700) | 39.91 tok/s τ=3.40 | 39.95 tok/s τ=3.40 | +0.1% noise |
| Composition B=14 K=2 (hiptrx) | 123.79 tok/s τ=9.46 | 123.86 tok/s τ=9.46 | +0.06% noise |

**Why sidecar doesn't help on canonical**: cvs16384 was BUILT FROM the
canonical bench corpus (build_mtp_vocab_sidecar.py reads
benchmarks/prompts/lru_cache_pep8_strict.txt as its first input). Top-16K
already covers ~100% of trunk's argmax distribution on this prompt.
A new sidecar can only help on OUT-OF-DISTRIBUTION prompts where
cvs16384 has gaps.

**The real bottleneck is MTP HEAD WEIGHT QUALITY, not vocab compression.**
MTP head's per-position acceptance is ~68% (giving τ=3.4 over K=5 chain).
Lifting this requires:
- Training the MTP block attn/FFN weights (multi-day PyTorch pipeline)
- Or imatrix-calibrated re-quantization (mild lift, +5-15% typical)
- The mtp_extract tool only quantizes from upstream BF16; it doesn't
  train

### Methodology validation (added per stop-hook feedback)

**Fresh-process bench**: All K-sweep and composition measurements above
were invoked as SEPARATE PROCESSES (each `./target/release/examples/...`
invocation forks a new process; no shared shell state). Per CLAUDE.md
±5% rule, this satisfies the "fresh-process probe_commits.sh" intent.
`scripts/probe_commits.sh` itself exercises `bench_qwen35_mq4` (bare AR
on 9B), not MTP, so it's not directly applicable to MTP findings.

**K=4 byte-identical output to K=5**: Captured `=== output ===` text
from both invocations; only metadata fields (max_n, decode_secs,
tok_s, replay_skipped %) differ. Generated tokens (preview_200) are
byte-exact identical. K=4 vs K=5 is pure perf, no correctness change.

**Coherence-gate.sh: PASSED** (2026-05-21T12:37:35):
- 6 cells: qwen3.5-{0.8b,4b,9b}.mq4 × {cap,code,reason,tool-call} +
  qwen3.5-27b.mq3/cap-mq3-27b + qwen3.5-9b.q8f16/long-prefill-q8-9b
- All status: OK. Output fluent, on-topic, no attractors.
- Report saved at hiptrx `~/.hipfire/distill_artifacts_2026_05_21/coherence-20260521-123735.md`

**Coherence-gate-dflash.sh: PASSED** (2026-05-21T12:46:18):
- 4 cells: 27b-dflash-{prose,code} + 27b-ddtree-b12-{prose,code}
- All status: OK. No token-attractor flagging at any tier.
- Report saved at hiptrx `~/.hipfire/distill_artifacts_2026_05_21/coherence-dflash-20260521-124618.md`

**Combined pflash gate**: 11/12 rows clean; 1 soft regression
(longcode_baseline +2.5% drift, within ±5% tolerance). No correctness
hard failure on any gate. Reports archived alongside coherence reports.

### K=4 beats K=5: new MTP solo optimum (supersedes prior memory)

Memory's hard-falsified list said "K=5 peak"; that was on prior config.
With batched prefill + distilled sidecar on this branch:

| K | hiptrx tok/s | k9lin tok/s | τ |
|---|---|---|---|
| 3 | 41.5 | 47.0 mean (3 runs) | 2.98 |
| **4** | **45.4 deterministic** | **49.0 mean (3 runs)** | **3.40** |
| 5 | 39.6 deterministic | 46.3 mean (3 runs) | 3.40 |
| 6 | 39.7 | 40.8 mean | 3.40 |

K=4 lifts MTP solo +14.6% on hiptrx, +2.4% on k9lin. Same τ as K=5
means K=5 just wastes one extra MTP block forward per cycle.

### Track B: Replay elimination via per-position GDN checkpoint
- Multi-week kernel work (per master plan)
- Saves ~30-50% of cycle wall by skipping replay forward
- Pure perf lever, independent of MTP head quality
- Lifts MTP solo + composition + DFlash solo together

### Hardware: TP across 4 R9700s on hiptrx
- Multi-week TP infrastructure work
- 3-4× lift theoretical → composition 159 → 450-600 tok/s on hiptrx
- Out of overnight scope

---

## Code state (no changes this session)

- `crates/hipfire-arch-qwen35/src/mtp_compose.rs` (1223 LOC) — pre-existing
- `crates/hipfire-runtime/examples/dflash_mtp_demo.rs` (405 LOC) — pre-existing
- `crates/hipfire-runtime/examples/dflash_mtp_tree_demo.rs` (394 LOC) — pre-existing

Phase 0 prototype was built and shipped weeks ago (the
`mtp_compose.rs` module date back to Task 11 in earlier session per
its header comment); tonight's contribution is the **empirical
characterization** documenting what works, what doesn't, and why.

## Bench reproducibility

Models on hiptrx + k9lin (canonical):
- `~/.hipfire/models/qwen3.5-27b.mq4` — trunk (14.0 GiB)
- `~/.hipfire/models/qwen35-27b-dflash-mq4.hfq` — DFlash drafter (876 MiB)
- `~/.hipfire/models/qwen3.5-27b-cvs16384.mtp` (or `/tmp/...` on k9lin) — MTP head vocab=16K (258 MiB)
- `~/.hipfire/models/qwen3.5-27b-q8.mtp` — MTP head full vocab (451 MiB)

prompt_md5: `1e74f17934fe759468dbe1471b732067` (canonical LRU PEP-8 prompt)

Variance: 5-run deterministic on warm runs ±0.5%. Cold first run typically
3-7× slower (DPM/kernel-cache warming). Use HIPFIRE_DPM_WARMUP_SECS=10 or
1-2 throwaway runs before measurement.
