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
