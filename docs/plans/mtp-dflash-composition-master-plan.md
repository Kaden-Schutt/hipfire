# MTP + DFlash composition master plan

**Date:** 2026-05-18 (original) — **2026-05-21 EMPIRICAL UPDATE BELOW**
**Branch:** `mtp-hiptrx-rocprof` (HEAD `f1dfa1ef`+, was `feat/mtp` `d10c0906`)
**Status:** **Phase 0 EXECUTED + EMPIRICALLY VALIDATED — STRUCTURALLY BLOCKED on current MTP head quality**
**Goal:** Match Unsloth/Atlas-class solo MTP (60-80 tok/s on 27B-3.5), then
STACK DFlash composition on top for **250-350+ tok/s** — exceeding DFlash
solo (199 tok/s on canonical).

This document is the load-bearing handoff for the multi-phase MTP+DFlash
composition work. Read this before re-investigating already-falsified
levers (see "Negative results" section).

## 2026-05-21 EMPIRICAL UPDATE — Phase 0 + Phase 1 Track A results

**TL;DR**: Composition architecture works correctly but does NOT exceed
DFlash solo with current MTP head. Phase 1 Track A (sidecar swap) gives
ZERO lift on canonical. The structural bottleneck is MTP head WEIGHT
QUALITY (not vocab compression, not composition design). Clearing
either Goal A (60-80+ tok/s solo) or Goal B (230+ composition) requires:
- (1) MTP block weight fine-tuning — multi-day PyTorch pipeline, OR
- (2) Replay elimination kernel work — multi-week, OR
- (3) TP across 4× R9700s on hiptrx — multi-week infrastructure

Empirical bench numbers (canonical 27B-3.5 K=4 Q8 greedy --no-chatml):

| Variant | hiptrx 1× R9700 | k9lin 7900 XTX | Goal A target | Goal B target |
|---|---|---|---|---|
| MTP solo | 45.4 tok/s | 49.0 tok/s (mean) | 60-80+ | — |
| DFlash solo | 126.1 | 181 | — | — (baseline) |
| Composition B=14 K=2 (best) | 123.8 | 159.3 | — | 230+ |

Composition is 1.7% under DFlash solo on hiptrx, 12% under on k9lin.
MTP solo is 18-27% under Goal A target.

See `docs/plans/mtp-composition-bench-2026-05-21.md` and
`docs/plans/mtp-hiptrx-rocprof-2026-05-21.md` for full data + reproducibility.

### What was tested and falsified tonight

| Lever | Result | Why |
|---|---|---|
| Composition linear B+K sweep across 5 configs | Best 159 k9lin / 124 hiptrx | MTP candidates only fire on DFlash full-accept (~7%) |
| Composition tree variant | 3-7× worse than linear | B×K MTP overhead dominates |
| Track A sidecar swap (v1 + v2 + merged) | 0.0% canonical lift | cvs16384 already ~100% coverage |
| MQ4 vs Q8 distilled head | Same tok/s, Q8 has τ=3.50 vs 3.40 | Both at similar wall, Q8 quant lifts accept marginally |
| K-sweep K=3,4,5,6,7 | K=4 +14.6% hiptrx, +2.4% k9lin | K=5 wastes one MTP block forward |
| Aggressive M (B=16 K=8, B=24 K=8) | Much worse | 2-tile overhead > extra commits |
| rocprof on cycle | No hidden kernels | Decode already batched, 50-55% peak BW |

### What's still NOT tested (out of overnight scope)

1. **MTP block weight fine-tuning** (multi-day): use the 249-prompt /
   70K trunk-argmax corpus captured tonight as supervised labels. Train
   block.attn + block.ffn weights with PyTorch + Transformers.
   Distilled head + bench at end. Could plausibly lift τ 3.40 → 5.0+.
2. **Imatrix-aware quantization**: requires `mtp_extract` extension to
   weight error minimization by activation magnitude. Multi-hour code
   work. Mild lift (+5-15% typical).
3. **Replay elimination via per-position GDN checkpoint kernel**:
   multi-week kernel + plumbing work. -30-50% cycle wall. Direct lift
   for both Goal A and Goal B.
4. **Pipelined MTP-DFlash overlap**: run MTP K-chain ON ANOTHER STREAM
   concurrently with DFlash drafter. Saves ~5-10 ms/cycle if streams
   actually overlap on R9700. Speculative.
5. **TP across 4× R9700 on hiptrx**: multi-week TP infrastructure.
   3-4× lift theoretical, ~350-500 tok/s on hiptrx (clears all goals).

## Headline ambition (ORIGINAL — kept for context)

| Layer | Current tok/s (27B-3.5 canonical) | Target |
|---|---|---|
| DFlash solo | 199 (CLAUDE.md) / 181 (today's measurement) | (baseline) |
| MTP solo | 53 (1.17× AR) → 49 today w/ K=4 | 60-80 (1.3-1.8× AR, Unsloth/Atlas class) |
| **DFlash + MTP composition** | **159 today (k9lin) / 124 (hiptrx)** | **250-350+** (1.25-1.75× DFlash solo) |

Sneaky-smart sequencing: **probe composition FIRST** with current weak MTP
to validate the architecture in days, not weeks. Composition gains are
ADDITIVE — if composition gives +2 commits/cycle on top of DFlash, that's
already 230-250 tok/s. Investing in solo MTP improvements then MULTIPLIES
the base.

> **Validation outcome (2026-05-21)**: composition contributes
> ~0 net commits with current MTP head. Master plan's honest math
> ("Composition is at-best-flat over DFlash solo if MTP adds 2-3
> commits per cycle") proven correct empirically.

## Proposed architecture: "MTP-extended verify"

Share a SINGLE batched trunk verify forward across DFlash drafts AND
MTP-chained candidates. No second verify forward — that's what made the
prior `[[mtp-native-head-deferred-2026-05-15]]` composition lose −32.7%.

```
super-cycle:
  1. DFlash drafter proposes K1=12 candidates from cur_pos
  2. MTP head proposes K2=3 candidates CHAINED off draft_11 (DFlash's last)
     - MTP K2-step uses bundled .mq4-mtp's head + trunk's lm_head
     - Cheap: 3 × MTP block forward ≈ 6 ms (overlappable with DFlash drafter)
  3. Trunk verify ONE batched forward over:
        [last_committed, draft_0..draft_11, mtp_0..mtp_2] = 16 positions
     - Modest BW increase from 13 → 16 positions
     - Replay still happens DFlash-style (see "replay handling")
  4. Accept longest matching prefix:
        - First check DFlash's drafts (argmax_per_pos[k] == draft[k] for k=0..11)
        - If all DFlash accepted, continue into MTP's chained drafts
          (argmax_per_pos[12+k] == mtp[k] for k=0..2)
        - First disagreement stops the chain
  5. Commit accepted prefix + bonus
  6. cur_pos += advance
```

### Why this should work better than prior failed composition

Prior `linear-chain composition (-32.7%)` failed because:
- Used the LOSSY continuous-embedding chain (`next_token_embed = Some(prev_hidden)`)
- MTP head OOD on drafter-generated hidden states
- Two SEPARATE trunk verify forwards (DFlash verify + MTP verify) doubled cost

This proposal differs:
- Uses DISCRETE-TOKEN chain (`spec_step_mtp_compressed_serial` pattern)
- MTP chain anchored on DFlash's drafter output's last DISCRETE TOKEN (draft_11)
  - The MTP head's `prev_hidden` is captured from trunk's verify position 11
    (after DFlash's draft_11 is verified), which IS in trunk's distribution
- SINGLE trunk verify over combined candidates → no extra forward cost

### Projected economics (rough)

Per super-cycle on 27B-3.5 + DFlash drafter:
- DFlash drafter forward: ~5 ms (small drafter)
- MTP K2 chain: ~5-6 ms (3 × MTP block @ 1.3 ms; can overlap with DFlash drafter)
- Trunk verify over 16 positions: ~52 ms (vs 50 ms for 13 positions; +4% BW)
- Replay (when partial accept on DFlash side, current pattern): ~30-40 ms
- Total wall: ~95 ms (vs DFlash-alone ~80 ms; +18% overhead)

Tokens committed:
- DFlash alone today: τ ≈ 10-13 (canonical)
- Composition: DFlash accept_count_1 + MTP accept_count_2 + 1 bonus
  - If MTP head accepts well after DFlash's accepted prefix: +2-3 commits
  - Realistic: 12-15 commits per super-cycle

Throughput projection:
- 13 commits / 0.080 s (DFlash alone) = **162 tok/s** (current observed: 199, so this back-of-envelope is conservative)
- 15 commits / 0.095 s (composition) = **158 tok/s** (basically flat vs DFlash solo)

WAIT — that's not above DFlash solo. The +2 MTP commits don't offset the +15ms verify+overhead cost.

### Honest reality check

The math above suggests composition is at-best-flat over DFlash solo if
MTP adds 2-3 commits per cycle. To CLEARLY EXCEED DFlash solo, we need
either:

1. **MTP contribution ≥4-5 commits per cycle** (requires good MTP solo
   acceptance — needs the trained sidecar)
2. **Eliminate the extra verify+overhead** (replay elim + fused-verify
   kernels — multi-week)
3. **Reuse DFlash's verify completely** — MTP just steals from DFlash's
   bonus slot, not chaining off DFlash's last draft. Sub-mechanism:
   - DFlash commits its accepted prefix from the verify
   - The bonus slot becomes a "MTP-extended" prediction: if DFlash's
     verify at position K1 had a HIGH-PROB argmax, that's the bonus
   - Otherwise fall back to plain DFlash bonus
   - Zero extra cost — bonus is already in verify_logits

Option 3 is the cheapest. But it's not really "composition" — just smarter
bonus picking. May give +5-10% by avoiding low-confidence bonus tokens.

For real composition wins, need MTP solo to be Atlas-class first.

## Phased plan

### Phase 0: Composition probe with current MTP (~3-5 days)

Build the "MTP-extended verify" prototype. Bench on canonical. Expected
outcome: small lift or flat. Either way, we KNOW the architecture's
viability.

Files to create/modify:
- `crates/hipfire-arch-qwen35/src/spec_step_dflash_mtp.rs` (NEW, ~500 LOC)
- `crates/hipfire-runtime/examples/dflash_mtp_demo.rs` (already exists,
  needs updating)
- Composition spans `dflash.rs` + `mtp_spec.rs` + new spec_step

Implementation notes:
- DFlash drafter proposes K1 candidates (existing path)
- After DFlash drafter forward, capture drafter's hidden at last position
- Run MTP head K2 times chained off DFlash's draft_K1-1 (discrete-token
  chain semantics, not lossy embedding override)
- Build combined verify_tokens = [last_committed, draft_0..draft_K1-1,
  mtp_0..mtp_K2-1] (K1+K2+1 positions)
- Trunk forward_prefill_batch over combined tokens
- Accept rule: greedy prefix match over the combined chain
- KV/DN snapshot+replay as DFlash does today

Risks:
- MTP head's prev_hidden source: trunk's hidden at verify position K1-1
  (captured from `verify_hidden[K1-1]`) — should be in distribution since
  it IS trunk's actual hidden state
- BUT: trunk's hidden at position K1-1 was computed assuming draft_K1-1 is
  REAL. If draft_K1-1 is rejected during accept, MTP head's chain was
  built on a hidden state that won't be in the final committed sequence.
  Need to think through whether this matters for correctness.

### Phase 1: MTP solo to Unsloth/Atlas class (1-2 wks)

If Phase 0 shows the composition architecture is sound, invest in solo
MTP perf. Two parallel tracks:

**Track A: Trained sidecar on hiptrx**
- `scripts/run_distill_parallel.sh` exists from earlier session work
- Currently defaults to `--kv-mode asym3`; needs `--kv-mode q8` flag added
  to match production deployment
- Run trunk-argmax distillation across wide corpus on 4× R9700
- Output: a sidecar with semantic vocab tail coverage (vs current
  sequential-id padding past the 3870-token corpus frequency tail)
- Projected: τ 3.84 → 4.3+ → tok/s 53 → 60-65 on canonical

**Track B: Replay elimination (kernel-level)**
- Per-position GDN (gated-delta-net) state checkpoint kernel
- Modify `gated_delta_net_q8_batch_seq` to optionally emit intermediate
  state per-position
- Plumb K+1 snapshot buffers through spec_step
- Projected: ~50% cycle wall reduction → tok/s 53 → 80+ on canonical
- Effort: 1-2 weeks careful kernel + dispatch + engine work

Track A is cheaper, Track B has higher ceiling. Could do A first.

### Phase 2: Productionize composition with strong MTP (~1-2 wks)

With MTP solo at 60-80 tok/s, redo composition bench. Expected:
- DFlash 199 + MTP contribution scaled up → 230-300+ tok/s
- Tune K1/K2 split for best τ × wall ratio
- Possibly adaptive split based on online acceptance rate (like DFlash's
  adaptive-B today)

### Phase 3: Stretch — multi-stream parallel drafting (~2-3 wks)

If composition validates and we have headroom:
- Run DFlash drafter forward AND MTP K-chain on PARALLEL streams
- Overlap with trunk verify of PREVIOUS cycle's drafts
- Atlas's "pipelined verify" pattern (which we previously falsified on
  RDNA via PR 5 path_d.md, but worth re-attempting with composition's
  different memory access pattern)
- Projected: another 20-30% lift if BW allows

## Negative results to NOT re-investigate

Hard-empirically falsified TODAY (2026-05-16 to 2026-05-18):

| Lever | Result | Why it fails |
|---|---|---|
| `--mtp-p-min` confidence cutoff (compressed AND full-vocab) | −3 to −19% at every threshold | MTP head's softmax intrinsically diffuse on code; cuts too many accept-candidates |
| Sampling vs greedy (host AND GPU) | −10 to −49% at every (K, temp) | Code workload too argmax-aligned; sampling injects pointless entropy |
| GPU sampling kernels (replace host overhead) | +1.6% noise | Host overhead was ~1 ms/cycle not ~7 (my estimate was wrong) |
| K-sweep K ∈ {2, 3, 4, 5, 6, 7} | K=5 stays peak | Cycle wall dominated by trunk verify, not K-dependent |
| Asym3/Fwht4 KV mode for MTP head | within noise of Q8 | KV cache is tiny, mode barely matters |
| `HIPFIRE_GATE_UP_VARIANT=ldsx` | −12% | LDS staging doesn't help at M=6 batch |
| `__launch_bounds__(32, 8)` bump | −1% noise | Compiler didn't change VGPR allocation; already at ceiling |
| 3.6-27B for MTP | −7.5% vs 3.5-27B | 3.6's MTP head trained for different distribution; recommend AR for 3.6 on code |
| `--temp=0.6` Unsloth coding default | still worse than greedy | Same sampling-vs-greedy issue on code |

Per-cycle profile (canonical 27B-3.5 cvs=16K bundled K=5 greedy):
- ~512 GEMM calls/cycle (~2 trunk forwards: verify + replay)
- HFQ4G256 WMMA GEMMs dominate (62% of cycle wall)
- Per-call BW efficiency: ~49% peak (468 of 960 GB/s)
- Kernels: VGPR=73, SGPR=20-26, scratch=0, LDS=0, wg=32 — comfortable
- NO register/LDS pressure issue. We're at per-call kernel efficiency ceiling.

## Today's shipped commits on `feat/mtp`

| SHA | Lines | What |
|---|---|---|
| `d344af5b` | +750 | Bundle .mq4-mtp format + skip-replay + p_min + kv-mode matrix |
| `4feae90d` | +370 | Host-side residual sampling (falsified default-off) |
| `615a4a67` | +311 | GPU sampling kernels (falsifies "missing kernels") |
| `d10c0906` | +61 | Per-kernel profile dump in mtp_only_demo |

## Bundled artifacts ready to use (~/.hipfire/models/)

```
qwen3.5-9b.mq4-mtp                   5.07 GiB  full-vocab MQ4 (best UX on 9B)
qwen3.5-27b.mq4-mtp                  14.16 GiB full-vocab MQ4
qwen3.5-27b-compressed.mq4-mtp       14.24 GiB compressed 32K sidecar (canonical bench)
qwen3.5-27b-cvs{4K,8K,16K}.mq4-mtp   14.17-14.20 GiB (cvs sweep variants)
qwen3.6-27b.mq4-mtp                  14.16 GiB full-vocab MQ4 (note: slower than 3.5)
```

Best canonical-bench combination so far:
```
HIPFIRE_DPM_WARMUP_SECS=10 cargo run --release \
  -p hipfire-runtime --example mtp_only_demo -- \
  --target ~/.hipfire/models/qwen3.5-27b-cvs16384.mq4-mtp \
  --prompt-file benchmarks/prompts/lru_cache_pep8_strict.txt \
  --max 120 --max-n 5 --temp 0.0 --no-chatml \
  --compressed-serial --kv-mode q8
# Expected: ~52-53 tok/s, τ=3.8
```

Profile a cycle anatomy:
```
HIPFIRE_PROFILE=1 HIPFIRE_PROFILE_CYCLES=10 [same args above]
```

## Inspection sources for next session

- **Atlas MTP deep-dive (CUDA reference)**:
  `https://raw.githubusercontent.com/Avarok-Cybersecurity/atlas/main/book/src/deep-dives/mtp.md`
  Key takeaway: K=2 + GREEDY accept + NVFP4 + pipelined verify = 131 tok/s
  on 35B-A3B (~1.87× their AR baseline)
- **Atlas scheduler code**:
  `crates/spark-server/src/scheduler/mtp_step.rs` — 4 specialized verify
  paths by K (k2/k3/k4/dflash)
- **Unsloth MTP guide**: `https://unsloth.ai/docs/models/qwen3.6`
  Recommended config: temp=1.0/0.6, top_p=0.95, top_k=20, K=2-3
- **Our prior MTP-Unsloth calibration**:
  `docs/plans/mtp-unsloth-target-2026-05-15.md`

## Related memories (read these for context)

- `[[mtp-native-head-deferred-2026-05-15]]` — prior composition falsification details
- `[[mtp-session-state-2026-05-15-compaction]]` — pre-bundle MTP state
- `[[mtp-unsloth-target-2026-05-15]]` — Unsloth/llama.cpp #22673 calibration
- `[[mtp-qualcomm-probe-v1-aborted-2026-05-15]]` — early hipGraph attractor bug
- `[[feedback_jinja_dflash_falsified_2026_05_13]]` — DFlash chatml collapse mechanism (same applies to MTP)
- `[[feedback_pr_gating_policy]]` — additive flag mergeable freely
- `[[project_pr5_pipelining_session_2026_05_08_part2]]` — pipelined verify falsified on RDNA, may revisit with composition
