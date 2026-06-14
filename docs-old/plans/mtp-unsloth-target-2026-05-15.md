# MTP — Unsloth / llama.cpp #22673 / AtomicBot turboquant as target reference

**Date:** 2026-05-15
**Status:** Research, not implemented. Calibrates hipfire `feat/mtp` against the
public state of the art.
**Sources:**

- Unsloth Qwen3.6 MTP guide: https://unsloth.ai/docs/models/qwen3.6
- llama.cpp MTP PR (am17an): https://github.com/ggml-org/llama.cpp/pull/22673
- HF discussion: https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF/discussions/6
- AtomicBot turboquant fork (TBQ KV + NextN MTP): https://github.com/AtomicBot-ai/atomic-llama-cpp-turboquant

## TL;DR

Their MTP is architecturally close to ours (single-transformer-block head,
separate KV, serial K-step draft, parallel verify). We've already independently
landed several of their pieces (in-graph batched argmax, separate KV, compressed
draft head). Their two material levers we haven't pulled:

1. **`--spec-draft-p-min 0.75`** — early-exit the K-step draft chain when
   draft top-1 probability drops below threshold. Trades nothing on full-accept
   cycles; saves wasted MTP compute + verify slots on low-confidence chains.
   **Estimated lift on canonical 27B-3.5 LRU: +5-15%.** Implementation cost: low.
2. **K=2 sweet spot (their data: 83% accept @ K=2, 75% @ K=3, 50% @ K=4).**
   Our K-sweep concluded K=5 was the canonical winner — but that sweep was run
   under the now-corrected ±10-15% "DPM noise" framing that masked real signal.
   Worth re-benching K∈{2,3,5,7} under the ±1-3% warm-cache protocol.
   **Estimated lift if K=3 wins: 0-10%.** Implementation cost: trivial (CLI flag).

Plus one strategic confirmation:

3. **AtomicBot's `turbo3` (WHT-rotated 3-bit) KV + NextN MTP combo on
   Qwen3.6-35B-A3B yields +33-36%** on M4 Max Metal. That's a direct
   validation of the `feat/tbq` + `feat/mtp` stack we already have queued.
   Once `feat/tbq` lands TBQ KV on master, swap MTP head KV from Q8_0 → TBQ3
   and re-bench. **Estimated lift: +5-15% on top of MTP baseline.** Implementation
   cost: depends on `feat/tbq` ship date (currently a seed branch with a plan doc).

The X-post claim "Q8_0 had best MTP hit rate 76.45%" empirically validates the
Q8 KV swap we shipped at `d3c97d57` (was justified architecturally; now
externally confirmed as the right format choice for this class of head).

## What they have that we have

| Feature | hipfire `feat/mtp` | Unsloth/PR #22673 | Notes |
|---|---|---|---|
| Single transformer-block MTP head (NextN overlay) | ✓ | ✓ | Native Qwen3.5/3.6 head |
| Separate KV cache for MTP head | ✓ | ✓ | We just shipped Q8_0; they default Q8_0 too |
| Shared lm_head between trunk and draft | ✓ (compressed) | ✓ (full) | They use full vocab; we slice top-K |
| Serial K-step draft + parallel verify | ✓ (K=5) | ✓ (K=2 default, max ~3) | Different K |
| In-graph batched argmax over K draft positions | ✓ (mtp_spec.rs:438) | ✓ | We D2H 4·K bytes, not F32[vocab] |
| In-graph batched argmax over verify positions | ✓ (mtp_spec.rs:542) | ✓ | Both D2H 4·(K+1) bytes |
| KV rollback on partial accept | ✓ | ✓ | Equivalent semantics |

## What they have that we don't

### Lever 1: `--spec-draft-p-min` confidence early-exit

llama.cpp PR #22673 plus `--spec-draft-p-min 0.75` (recent llama.cpp main):
during the K-step draft chain, capture top-1 probability per step. If the
draft's confidence drops below threshold (e.g. 0.75), truncate the chain
**before** trunk verify. Saves:

- Remaining MTP block forwards (each ~1.3 ms on our cycle)
- Corresponding verify slots (lm_head GEMM rows)
- KV rollback work for slots that were never going to accept anyway

Our `spec_step_mtp_compressed_serial` always runs K=5 + verifies all 5 +
rolls back. With τ=3.7, ~26% of slots are wasted compute. p_min would cut
most of them.

**hipfire delta to implement:**
- `mtp_head_forward_compressed` currently writes K argmax-only ids to host.
  Needs to also write K top-1 probabilities (one extra `argmax_max_f32_batched`
  variant or compute `softmax_top1_prob` in-graph).
- `spec_step_mtp_compressed_serial` adds an early-exit branch in the
  K-loop after each draft step's prob check.
- New flag `--mtp-p-min` (default 0.0 = disabled, opt-in 0.75 to match
  llama.cpp).

### Lever 2: K-sweep re-bench under proper methodology

Their K=4 acceptance crash from 83→50% suggests our K=5 may be past the
acceptance cliff for code-corpus prompts too. We landed K=5 as canonical
from a sweep that:
- Used the bad ±10-15% noise band (real signal getting hand-waived)
- Did not warm DPM/kernel cache
- Did not run multi-process probe_commits.sh

Per [[mtp-session-state-2026-05-15-compaction]] the F32 vs Q8 -3% delta is
also unconfirmed. Re-run K∈{2, 3, 5, 7} under:
```
HIPFIRE_DPM_WARMUP_SECS=10
./scripts/probe_commits.sh <baseline> <head>  # multi-process aggregation
```

If K=3 lands within 1% of K=5, prefer K=3 (lower cycle wall, less rollback
risk on long-ctx OOD where attractor manifests). If K=2 within 3%, prefer
K=2 (matches Unsloth canonical, simplifies tuning surface).

### Lever 3: TBQ-rotated KV for MTP head (cross-branch)

AtomicBot fork: `-ctk turbo3 -ctv turbo3 -fa on` for Qwen3.6-35B-A3B
NextN gives **+33.8% over `turbo3`-base**, comparable to **+35.9% over f16-base**.
The 3-bit WHT-rotated KV doesn't degrade NextN acceptance.

This is the `feat/tbq` + `feat/mtp` stack we already have queued. When TBQ
KV lands on master:
1. Swap `Qwen35MtpHeadKvCache::new_gpu_q8` → `new_gpu_turbo3` (or our `tbq3`)
2. Verify acceptance doesn't drop (rotation matters for cross-attn between
   draft trunk-hidden and MTP head's KV)
3. Re-bench

Currently blocked on `feat/tbq` shipping (it's a seed branch — see
`docs/plans/tbq4-kv-cache-plan.md`).

## What they have we should NOT chase

### Cross-cycle async MTP overlap (`llama_decode_mtp_async`/`wait`)

PR #22673 implements depth-2 overlap where MTP draft compute for cycle N+1
runs concurrently with target verify for cycle N. **This is exactly the
PR 5 path_d.md D0-D3b speculative prefetch we already empirically falsified
on hipfire** (-1.7% to -7.16% across hipx/hiptrx). Per
[[project_pr5_pipelining_session_2026_05_08_part2]]: BW-saturation pattern
holds — concurrent draft + verify streams contend for the same memory bus
rather than overlapping. RDNA3/4 + ROCm 7.2 doesn't show this lever.

Don't re-implement for MTP. The architecture is identical to PR 5; result will
be the same.

### UD-Q4_K_XL "dynamic" quant

Their quant format applies per-block scale + selective important-layer upcast.
Our MQ4 already does the equivalent (per-block scale, lm_head/embed F16, no
WMMA path for MFP4G32 means selective upcast is implicit). Per
[[mfp-hfp-dead-use-mq4-q8]] MQ4 is the canonical winner on hipfire — no need
to rewrite as UD-style.

## Comparative numbers

Caveats:
- Hardware not directly comparable (RTX 5090 ~1.0 TB/s vs 7900 XTX 960 GB/s)
- Model not directly comparable (Qwen3.6 is structurally different from 3.5)
- Quant not directly comparable (UD-Q4_K_XL vs MQ4)

| System | Model | Quant | Hardware | Mode | tok/s |
|---|---|---|---|---|---|
| Unsloth+llama.cpp | Qwen3.6-27B | UD-Q4_K_XL | RTX 5090 | MTP K=2 | 140 |
| Unsloth+llama.cpp | Qwen3.6-27B | UD-Q4_K_XL | RTX 5090 | MTP K=2 (X claim) | ~120 |
| Unsloth+llama.cpp | Qwen3.6-27B | Q8_0 | RTX 5090 | MTP K=2 (X claim) | ~90 |
| Unsloth+llama.cpp | Qwen3.6-27B | Q8_0 | Tesla P40 ×2 | MTP K=3 | 14-15 |
| AtomicBot | Qwen3.6-35B-A3B | f16 / turbo3 KV | M4 Max Metal | NextN K=2 | 89-95 |
| **hipfire master** | Qwen3.5-27B | MQ4 | 7900 XTX | DFlash | **199** |
| **hipfire feat/mtp** | Qwen3.5-27B | MQ4 + Q8 KV | 7900 XTX | MTP K=5 (compressed) | **~47** |
| **hipfire AR baseline** | Qwen3.5-27B | MQ4 | 7900 XTX | AR | ~45 |

The only column where hipfire wins clearly is DFlash. Our MTP path at 47 tok/s
is barely above AR baseline — **far** below where llama.cpp MTP lands relative
to its own AR baseline (~2x lift). The Unsloth target shows MTP should be
roughly 1.5-2x AR, which on hipfire would be 67-90 tok/s. We're 30-50% below
where the lever should land.

This isn't quite an apples-to-apples gap because:
- DFlash already eats most of the speculative-decode lift; MTP is a different
  decomposition trying to capture the same headroom
- Our compressed-vocab approach is lossier than full-vocab (smaller draft
  head means lower acceptance per step, harder to scale K)
- Unsloth's K=2 fits their compute budget better; our K=5 ratio is wrong

**Strategic implication:** MTP-as-DFlash-replacement is not the right framing.
DFlash already wins. MTP's value is **stacking with DFlash** — composing as
the drafter inside DFlash's tree, OR as orthogonal bonus tokens. See
`docs/plans/dflash-mtp-composition-orthogonal.md` for the composition analysis
that already concluded "no clear +20% gate without trained head".

## Recommended next actions

Priority order (cost ascending, expected lift descending within tier):

1. **K-sweep re-bench under ±1-3% methodology** (~1 hour). Free signal on
   whether K=5 was the right call. Use `probe_commits.sh` + `HIPFIRE_DPM_WARMUP_SECS=10`.
2. **Coherence-gate + probe_commits.sh on Q8 swap** (~30 min). Outstanding
   from prior session. Resolves whether F32 KV is genuinely 3% better than Q8
   or if Q8 -3% was real-noise/cold-start. If F32 is materially better, roll
   back the Q8 swap (X-post 76.45% Q8 hit-rate validates Q8 directionally,
   but local data wins).
3. **Implement `--mtp-p-min` early-exit** (~3-4 hours). Plumb top-1 prob from
   draft GEMM through to chain truncation logic. Estimated +5-15%.
4. **Re-distill on hiptrx with KV mode matching deployment** (~6 hours; user's
   pre-compaction question). If we land Q8 as canonical, re-distill under Q8;
   if we land F32 as canonical, re-distill under F32. Currently distill defaults
   to asym3 (script never updated post-Q8-swap).
5. **Wait for `feat/tbq` to ship TBQ KV, then swap MTP head KV → TBQ3**
   (~unknown; blocked on other agent's `feat/tbq` work). Estimated +5-15%
   stacked on MTP baseline per AtomicBot data.

Out of scope:
- Cross-cycle async MTP overlap (already falsified on RDNA + ROCm 7.2,
  see PR 5 path_d.md history).
- Re-architecting as full-vocab head (lm_head GEMM is already the cycle's
  dominant cost; full-vocab regresses, not helps).

## Related memories

- [[mtp-session-state-2026-05-15-compaction]] — current branch state, outstanding gates
- [[mtp-native-head-deferred-2026-05-15]] — why native head v1 was deferred
- [[project_pr5_pipelining_session_2026_05_08_part2]] — async cross-cycle overlap falsification
- [[mfp-hfp-dead-use-mq4-q8]] — quant format guidance
- [[feedback_pr_gating_policy]] — additive `--mtp-p-min` flag is mergeable freely
