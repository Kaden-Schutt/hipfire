# DFlash Port Plan — hipfire 0.1.6

**Theme:** Block-diffusion speculative decoding. Port z-lab/dflash from
Python/CUDA to native Rust+HIP on top of hipfire's existing Qwen 3.5
engine. Expected payoff: **2-4× decode tok/s** on every target we ship
(4B / 9B / 27B / 35B-A3B). Drafts are pre-trained and on HuggingFace —
no training work required.

**Constraint:** Port must be Rust-native. No PyTorch, no vLLM, no
SGLang runtime. Same "single binary, no Python in the hot path"
invariant as everything else in hipfire.

**Status:** planning. Sanctioned 2026-04-13 for an overnight build.
Phase 1 must complete and surface the scope answer before Phase 2 begins.

## Reference materials (read first)

- **Paper:** arXiv:2602.06036 — architecture + training + eval numbers
- **Repo:** https://github.com/z-lab/dflash
- **Models:** https://huggingface.co/z-lab — `Qwen3.5-{4B,9B,27B,35B-A3B}-DFlash`
- **Blog:** https://z-lab.ai/projects/dflash/

Key files in the upstream repo to study:
- `dflash/model.py` — draft model forward pass
- `dflash/generate.py` — speculative decoding loop
- `dflash/spec/` — verification logic (if present)
- The vLLM integration under `dflash/vllm/` for an optimized batched path

## Architecture summary (from HuggingFace model cards)

- Draft model: **1B params, BF16**, Qwen3.5 architecture with added diffusion machinery
- Block size: **8 or 16 tokens** (configurable; paper favors 16)
- Speedup claim: **up to 4.4×** decode over pure AR
- Target model: stock Qwen 3.5 at any size (9B/27B/35B-A3B published)
- Drafts have **independent weights** (not shared with target)
- Tokenizer: shared with target (Qwen 3.5 tokenizer)

Unknowns to resolve in Phase 1:
- Exact diffusion process (continuous Gaussian vs discrete masking?)
- Number of denoising steps per block (1-shot or iterative?)
- Whether draft has its own KV cache that persists across spec steps
- Whether draft consumes target's hidden states (EAGLE-style) or runs standalone
- Head layout — is the "diffusion head" a lightweight MLP or a full decoder pass?

## Phase 1 — architecture scope check (HARD GATE)

**Goal:** answer "is this port 2 weeks or 6 weeks?" before committing to
implementation.

**Actions:**

1. Clone `z-lab/dflash` locally to `.dflash-reference/` (gitignored).
2. Read `dflash/model.py` and transcribe the forward-pass architecture
   into pseudocode.
3. Download `z-lab/Qwen3.5-9B-DFlash` safetensors + config.json. Inspect:
   - Tensor list: compare against stock Qwen 3.5 9B tensor layout
   - Number of layers / hidden_size / n_heads — confirm it's Qwen-arch
   - Look for non-standard tensors (diffusion schedules, noise
     projectors, denoise-specific heads)
4. Read the paper's algorithm section. Summarize:
   - Noise schedule (if any)
   - Number of denoising steps
   - Input representation at each step
5. Study `dflash/generate.py`. Transcribe:
   - The spec_generate loop
   - Draft forward call signature
   - Verification + acceptance logic
   - Rejection sampling (if temp > 0)

**Deliverable:** `docs/DFLASH_ARCHITECTURE.md` with:
- Pseudocode of draft forward
- Pseudocode of speculative loop
- Component inventory: what reuses hipfire's existing code, what's new
- Scope estimate: weeks of work, risk flags
- **Go / no-go recommendation** for Phase 2

**Do NOT start Phase 2 until this gate is passed.**

## Phase 2 — draft weight converter (if Phase 1 says "go")

**Goal:** load DFlash draft weights into hipfire's native tensor format.

**Outputs:**

1. `crates/hipfire-quantize/src/bin/dflash_convert.rs` — binary that
   reads HF safetensors + config.json, writes `.hfq` (stock Qwen-arch
   tensors) + any dflash-specific tensors as a new section.
2. Target file layout: `.dflash` or extend `.hfq` with a new header
   section — decide based on how different the dflash-specific tensors
   are. Prefer extending `.hfq` if it's just "Qwen + a few extra
   tensors".
3. Leave draft weights in BF16 initially. MQ4 quantization is Phase 7.

**Verify via:** load converted weights + run a single-token forward,
compare logits to upstream PyTorch reference (dump logits from a known
input, compare bitwise-close).

## Phase 3 — draft forward pass

**Goal:** native Rust+HIP forward pass for the 1B draft, producing B
tokens per call.

**New files:**
- `crates/engine/src/dflash.rs` — `DflashWeights`, `DflashScratch`,
  `dflash::draft_forward(target_hidden, prev_tokens) -> [B; u32]`
- `kernels/src/dflash_*.hip` — any new kernels (noise injection,
  schedule application, denoise step) — scope depends on Phase 1
  answers.

**If draft is "Qwen + simple head" (best case):**
- Reuse `forward_prefill_batch` with B positions as the "prompt"
- Replace the argmax head with draft's diffusion-trained head
- ~2-3 days work

**If draft has iterative denoising with N steps:**
- Run draft's transformer N times per block
- Each step gets progressively-denoised embeddings as input
- ~1-2 weeks work, new kernels

**Verification:** run the Rust forward, compare output token
distribution to upstream Python draft on the same context.

## Phase 4 — batched verification (target side)

**Goal:** target model emits logits at all B positions given the B
draft tokens, so we can check which prefix the target agrees with.

**Most of this already exists.** `forward_prefill_batch` runs batched
forward over N positions and writes `scratch.logits_batch[N × vocab]`.
Need:

1. Make `logits_batch` available on the daemon's verify path (might be
   internal-only today)
2. Add `verify_spec(draft_tokens: &[u32; B]) -> accepted_length`
   - For each position i in 0..B:
     - argmax(target_logits[i]) ?= draft_tokens[i]
     - if match: continue
     - if mismatch: break, return i + 1 (target's argmax at i is the
       next token; +1 because we always accept at least one)
3. Advance both KVs by `accepted_length`

**Temp > 0 rejection sampling:** speculative sampling is more
complicated. For 0.1.6 ship, greedy-only is fine. Rejection sampling
is a Phase 7+ polish.

## Phase 5 — speculative daemon loop

**Goal:** new daemon message type that orchestrates draft + target.

**Protocol addition (`daemon.rs`):**

```
→ {"type":"spec_load", "target":"...mq4", "draft":"...dflash", "params":{...}}
← {"type":"spec_loaded", "target_arch":..., "draft_arch":..., "block_size":16}

→ {"type":"spec_generate", "id":"...", "prompt":"...", "temperature":0, "max_tokens":100, "block_size":16}
← {"type":"token","id":"...","text":"..."}   (streamed)
  ...
← {"type":"done","id":"...","tokens":N,"accept_rate":0.78,"tok_s":350}
```

**Daemon loop:**

```rust
// Initial prefill
target.prefill(prompt);
// draft.prefill(prompt);   // if draft is stateful

loop {
    let draft_tokens = draft.diffuse(B);    // Phase 3
    let accepted = target.verify(&draft_tokens);  // Phase 4

    for i in 0..accepted {
        emit_token(draft_tokens[i]);
    }
    // target's next_token (from verification): always ONE more
    emit_token(target_next_token);

    if stop_condition { break; }
    // advance both KVs by accepted + 1
}
```

## Phase 6 — CLI / serve integration

**CLI changes:**
- `hipfire pull qwen3.5:9b --dflash` — pulls target + draft together
- `hipfire run qwen3.5:9b` auto-uses DFlash when draft is available
- `hipfire config set dflash on|off|auto` — global toggle (auto = on if
  draft present, off otherwise)
- `hipfire config set spec_block_size 16`

**Serve changes:**
- `/v1/chat/completions` gets accept-rate in `usage.accept_rate`
- Accept-rate < threshold (e.g. 0.4) logs a warning — probably a bad
  domain for this draft

## Phase 7 — quantization + shipping

**Actions:**

1. Quantize draft BF16 → MQ4 via existing quantizer. Verify
   accept-rate doesn't tank (expect 5-10% drop at most).
2. Upload MQ4 drafts to `schuttdev/hipfire-qwen3.5-{size}-dflash`
3. Register new tags: `qwen3.5:9b-dflash` etc
4. Update quality gate: greedy output of dflash path must be byte-
   exact with non-dflash greedy (invariant — spec decoding is
   distribution-preserving)

## Phase 8 — benchmarks + docs

**New docs:**
- `docs/SPECULATIVE_DECODING.md` — how it works, how to enable, when
  it helps (long outputs, coding, math), when it doesn't (short
  responses, high-temp creative)
- `docs/BENCHMARKS.md` — add dflash columns. Target numbers:

| Model | target tok/s | dflash tok/s | speedup | accept rate |
|---|---:|---:|---:|---:|
| 7900 XTX 4B  | 180 | 400-540 | 2.2-3.0× | ~0.75 |
| 7900 XTX 9B  | 125 | 300-450 | 2.4-3.6× | ~0.75 |
| 7900 XTX 27B | 47  | 130-180 | 2.8-3.8× | ~0.78 |
| V620 9B      | 65  | 150-220 | 2.3-3.4× | ~0.75 |
| BC-250 4B    | 77  | (skip — APU is BW-bound, spec won't help much) |

## Risk catalog

| Risk | Likelihood | Mitigation |
|---|---|---|
| Phase 1 says "6 weeks, complex noise schedule" | medium | Honor gate; defer to 0.1.7 if scope is too big |
| Draft forward needs new HIP kernels we don't have (Gaussian noise RNG, diffusion schedules) | medium | Port upstream kernels if CUDA-only; write from scratch if needed |
| Target verification path has subtle bugs that only manifest at temp>0 | high | Ship greedy-only first; temp>0 is Phase 7+ polish |
| Byte-exact greedy parity with non-spec path breaks | medium | Quality gate is the backstop; if it fails, pause + bisect |
| Draft + target together blow VRAM on 12 GB cards | medium | Per-arch config default — disable dflash on < 16 GB VRAM; quantize draft to MQ4 early |
| Accept-rate is domain-dependent and low on conversational prompts | likely | Document expectations; runtime `dflash: auto` disables on low accept |

## Out of scope for 0.1.6

- Training drafts for non-Qwen-3.5 targets (e.g., carnice, qwopus)
- MoE targets (Qwen3.5-35B-A3B, 122B-A10B) — skip until single-expert
  version is solid
- Multi-draft ensembles
- Tree-based verification (EAGLE-3 style)
- Rejection sampling at arbitrary temperatures (greedy only in 0.1.6)

## Immediate next steps for overnight run

1. Clone `z-lab/dflash` into `.dflash-reference/` (add to .gitignore)
2. Download `z-lab/Qwen3.5-9B-DFlash` config + one safetensors shard
3. Run Phase 1 architecture dig — write `docs/DFLASH_ARCHITECTURE.md`
4. **STOP at the Phase 1 gate.** Do not start implementation until the
   scope answer is in the doc and sanity-checked against the paper.

If Phase 1 says "simple head, 2 weeks" → continue through Phase 8 in
sequence, quality-gating each phase before advancing.

If Phase 1 says "complex noise schedule, 4+ weeks" → stop, open an
issue summarizing findings, queue for 0.1.7.

## Invariants that must not break

- Greedy decode temp=0 on the NON-dflash path stays byte-exact (no
  changes to forward_prefill_batch / forward_scratch signature).
- `hipfire run` / `hipfire serve` still work without a draft loaded
  (dflash is additive).
- Quality gate (`./scripts/quality-gate.sh`) still passes on every
  commit.
- BC-250 / V620 bench numbers don't regress (speed-gate).
- MQ4 / MQ6 target model uploads to HF are untouched.
