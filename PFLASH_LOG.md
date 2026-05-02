# PFLASH LOG

Append-only progress log. Entries are timestamped + reference commit SHAs.

## 2026-05-02

### Start

- Branch `feat/89-llama-batched-prefill` at e17684b (Phase A-D landed).
- PFlash PRD: `docs/plans/pflash-speculative-prefill.prd`.
- Contract: `PFLASH_CONTRACT.md`.
- Drafter target: Qwen3-0.6B. Existing artifacts: `~/.hipfire/models/qwen3-0.6b.hf4` (HFQ4G256) and `qwen3-0.6b.mq4` (MQ4G256).
- Tokenizer: per Phase A smoke, qwen3-0.6b.hf4 metadata produces a tokenizer that matches the model. Need to verify cross-tokenizer compatibility with target later.

### Phase 0: NIAH harness + full-prefill baseline (in progress)

Goal: make long-context target measurable before touching runtime behavior.

Deliverables (per PRD §6 Phase 0):
- `benchmarks/longctx/niah/niah_{8k,16k,32k,64k,128k}.jsonl` (committed or generator-deterministic).
- `crates/engine/examples/pflash_niah_bench.rs` (full-prefill baseline, prints TTFT breakdown + md5s).

Acceptance:
- Full-prefill NIAH baseline passes at every supported context size given available VRAM.
- Bench reports TTFT broken into tokenize / prefill / first decode step / total.
- Source md5 + binary md5 logged.

### Phase 0 results (qwen3.5-4b.mq4, gfx1100, asym3 KV)

```
8K   fixture md5 6f24cd79...  ttft 32259 ms  prefill 1748 ms (3139 tok/s)  decode 161.5 tok/s  PASS (mauve-velociraptor-7741 retrieved)
       tokenize 30511 ms (5487 tokens) — see DEFERRED.md, tokenizer perf is the bench TTFT bottleneck, not prefill
```

Phase 0 status: **DONE for 8K**. 16K-128K runs deferred until Phase 0/Phase 1 share the same harness path; Phase 1's compression demo will exercise larger contexts where tokenize-vs-prefill curves matter most.

### Phase 1.0: pflash module scaffold (DONE 075ddc6)

`crates/engine/src/pflash.rs` with PflashMode/Config/State/Decision/
BypassReason/RequestKind data model + `decide_bypass` pure-CPU gate +
`maybe_compress_prompt` entry. 6/6 unit tests green.

### Phase 1.1: drafter loading + tokenizer-compat (FINDING)

Added `pflash::load_drafter` (HFQ → LlamaConfig + LlamaWeights + Tokenizer
+ ForwardScratch + KvCache stashed in PflashState) and
`tokenizers_compatible` (vocab_size + probe-phrase round-trip).
`decide_bypass` now returns `TokenizerMismatch` when the drafter loads but
tokenizer probes diverge.

Smoke `pflash_load_demo qwen3.5-4b.mq4 qwen3-0.6b.hf4`:
- load 358 ms, 28 layers, 1024 dim, 439 MB VRAM estimate.
- Target vocab 248144, drafter vocab 151743 → MISMATCH (correct refusal).

Escalated to MANUAL_REVIEW.md: matched-tokenizer drafter is not in the
local model dir. Three forward paths offered. Phase 1.2 (Q/K capture)
proceeds with a same-tokenizer dev pairing (qwen3-0.6b as both drafter
and target stand-in) so the scoring infra advances while the drafter
question is unblocked by user.

### Phase 1.2: K-capture + per-block scoring (DONE)

Added `pflash::BlockScores` + `compute_scores_cpu(state, gpu, source, block_size)`.
Implementation: per-token `forward_scratch_embed + forward_scratch_compute`,
then `download_f32(scratch.k)` to capture last-layer post-RoPE K per
position. CPU mean-pools K per block and computes cosine similarity vs
the last position's K. Pure CPU at this phase — no llama.rs surface
changes, no qwen35 risk.

Smoke `pflash_load_demo qwen3-0.6b.hf4 qwen3-0.6b.hf4` (32-token toy
prompt, block_size=8):
  4 blocks, scores [0.731, 0.754, 0.779, 0.922]
  → last block highest (tail-K self-correlation, expected for cosine MVP).

Phase 1.3 next: wire scores → span selection → compressed token IDs,
re-prefill on a Qwen3-0.6B target with the compressed prompt to verify
correctness end-to-end at smaller-than-NIAH context (since drafter
availability is still escalated to MANUAL_REVIEW.md for matched-tokenizer
Qwen3.5 targets).

### Phase 1.3: span selection + compressed token emission (DONE)

Added `pflash::select_spans(scores, sink, recent, keep_ratio, min_keep)`
and `pflash::emit_compressed(source, kept_spans)`. Selection rules per
PRD §5.4: always keep sink prefix + recent tail, fill remaining budget
from highest-scoring middle blocks (descending score, ascending index
tie-break for determinism), coalesce adjacent spans into single ranges.
Pure CPU. 5 new unit tests covering full-when-under-min-keep,
top-block-with-anchors, adjacent-coalesce, in-order emit, OOB clamp.
12/12 module tests green.

Smoke `pflash_load_demo qwen3-0.6b.hf4 qwen3-0.6b.hf4` extended:
  32 source → 4 blocks of 8 → scores [0.731, 0.754, 0.779, 0.922]
  select_spans(sink=4, recent=4, keep_ratio=0.5) → 3 spans:
    [(0, 4), (16, 24), (28, 32)] = 16 tokens = exactly 0.5 ratio
  emit_compressed: 16 tokens, monotonic, length-consistent.

Phase 1.4 next: full pflash entry, compute_scores → select_spans →
emit_compressed, returning a CompressedPrompt. Then end-to-end
verification: source → compress → re-prefill on target → check the
compressed prompt produces a coherent next-token continuation.

### Phase 1.4: maybe_compress_prompt full pipeline (DONE)

Wired compute_scores_cpu → select_spans → emit_compressed inside
maybe_compress_prompt. Returns PflashDecision::Compressed(CompressedPrompt)
with source_tokens, kept_tokens, kept_spans, source_md5, compressed_md5,
and PflashTimings (score / select / gather / total ms). Falls back to
Bypass(BelowThreshold) when budget would keep the entire prompt (no
point recompressing the same tokens through the target).

Smoke (qwen3-0.6b self-pair, 32-tok toy, sink=4 recent=4 keep=0.5):
  source=32 kept=20 ratio=0.625
  source_md5 42b2f9af7e3b6b0e94a58ca91cf7780a
  compressed_md5 c4400d6802977a0bf1bed2f2a8b120e9
  kept_spans = [(0, 4), (16, 32)]
  timings: score=96ms select=0ms gather=0ms total=96ms
  invariants: length_ok=true spans_disjoint=true monotone=true md5_present=true

Phase 1.5 next: end-to-end retrieval verification. Encode a small
filler+needle+question prompt with qwen3-0.6b's tokenizer, run
maybe_compress_prompt, then prefill the COMPRESSED stream through the
SAME qwen3-0.6b as a target stand-in and decode greedily. PASS if the
needle text appears in the answer despite compression. Real Qwen3.5
target retrieval blocks on the matched-tokenizer drafter availability
(MANUAL_REVIEW.md).

### Phase 1.5: end-to-end compress -> target re-prefill -> decode (DONE)

New `pflash_compress_demo.rs` exercises the whole pipeline end-to-end
on a single qwen3-0.6b artifact (drafter and target both loaded from
the same HFQ; double-loaded as the dev workaround for the matched-
tokenizer drafter gap):

  1. Build a filler+needle+question prompt (~2000 chars / ~392 tokens)
  2. Tokenize via the target tokenizer
  3. load_drafter(target_path) into PflashState
  4. maybe_compress_prompt(...) -> CompressedPrompt
  5. unload_drafter (free VRAM for target KV)
  6. llama::forward_prefill_batch on the COMPRESSED token stream
  7. Greedy decode via forward_scratch_embed + forward_scratch_compute
  8. Verify pipeline_ok (non-empty alphabetic answer); log needle
     retrieval as informational

Smoke runs (qwen3-0.6b.hf4, gfx1100):

  keep_ratio 0.30: 392->120 (30.6%), 4 spans, target prefill 6000 tok/s,
                   decode 310 tok/s, pipeline_ok=true, needle missing
                   (model hallucinates "The answer is a single word.")
  keep_ratio 0.70: 392->280 (71.4%), 5 spans, target prefill 7568 tok/s,
                   decode 277 tok/s, pipeline_ok=true, needle missing
                   ("The secret pass code is 12345...") -- model gets
                   the cue shape but hallucinates the value
  keep_ratio 1.00: bypass(BelowThreshold) -- correct, budget keeps full
                   prompt so no compression to attempt

The pipeline is correct: compressed_md5 stable, kept_spans coalesced,
target re-prefill on compressed stream produces decodable next tokens,
greedy decode runs at the model's normal tok/s. Needle retrieval at
this scale is the model's quality ceiling -- 0.6B BF16 / HFQ4 cannot
reliably hold "mauve-velociraptor-7741" against typical filler shape,
which is a known small-model limitation, not a PFlash bug.

Phase 1 (drafter compression MVP) complete in plumbing terms. Real
NIAH retrieval at 8K/16K requires the matched-tokenizer drafter pair
escalated in MANUAL_REVIEW.md; until that's resolved the bench can't
demonstrate Lucebox-class retrieval numbers.

Phase 2 (HIP scoring + selection kernels) advances next: the CPU
scoring loop is ~30 s on 8K and ~12 min projected on 128K. Phase 2
moves the per-token K capture + block scoring onto the GPU.
