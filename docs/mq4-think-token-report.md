# MQ4 Vocabulary Gap: Missing `<think>` / `</think>` Special Tokens

**Date:** 2026-05-11
**Impact:** Thinking=ON produces "Here's a thinking process:" visible monologue instead of hidden `<think>` blocks. Two-pass app-level reasoning workaround functional but slower.

**Hipfire branch:** `feature/think-policy-ar-dflash` on `DrBearJew/hipfire` (commit `f0fe2b9`)

---

## Root Cause

**The MQ4 conversion dropped `<think>` (ID 151657) and `</think>` (ID 151658) from the vocabulary.**

Qwen 3.6 27B was trained with these tokens in its vocabulary. llama.cpp works because it loads them from the GGUF's `tokenizer.ggml.tokens` array. The MQ4 conversion only preserved tokens 0..151642 (native Qwen vocab, 151643 entries) plus 3 added tokens (`<|endoftext|>` at 151643, `<|im_start|>` at 151644, `<|im_end|>` at 151645). Tokens 151646+ were dropped.

Evidence:

```
$ python3 -c "
import json
with open('~/.hipfire/models/tokenizer_config.json') as f:
    config = json.load(f)
added = config.get('added_tokens_decoder', {})
for k,v in added.items():
    if 'think' in str(v).lower():
        print(k, v)
"
# Output: (empty — no think tokens registered)
```

Compare with the GGUF source (`Qwen3.6-27B-Q4_K_M.gguf`), which has `<think>` at 151657 and `</think>` at 151658 in its vocabulary.

---

## What We Implemented (Correct, Needs MQ4 Fix to Activate)

### 1. Hardcoded Token IDs (daemon.rs, line ~3186)

```rust
// BUG: tokenizer.special_token_id("<think>") returns None because
// the MQ4 vocab doesn't include these tokens.
// WORKAROUND: hardcode the IDs Qwen 3.6 uses in its native vocab.
let think_start_id: u32 = 151657;  // <think>
let think_end_id: u32 = 151658;    // </think>
```

### 2. Prefix Injection Before Prefill (daemon.rs, line ~3193)

```rust
if thinking_allowed {
    new_tokens.push(think_start_id);           // 151657
    new_tokens.push(newline_token);           // \n
    // Model now sees <think>\n as input prefix
}
```

### 3. Think State Machine (sampler.rs)

- `ThinkState` struct: `in_think`, `think_blocks_seen`, `visible_answer_started`, `tokens_inside_think`
- `merge_think_bans()`: bans `<think>` reopen, bans `</think>` when not in think, **forces 64+ tokens inside think** before allowing `</think>`
- `update_think_state()`: tracks token transitions

### 4. Vocab Size Bump (llama.rs, line 1933)

```rust
logits: gpu.alloc_tensor(&[config.vocab_size.max(151659)], DType::F32)?,
```

This pads the logits buffer so bans at token IDs 151657/151658 don't overflow.

### 5. Three Sample Sites Wired

| Site | Location | merge_think_bans | update_think_state |
|------|----------|-----------------|-------------------|
| First token | ~3367 | ✓ | ✓ |
| Budget-alert skip | ~3584 | ✓ | ✓ |
| Main loop | ~3667 | ✓ | ✓ |

### 6. Prefix Scan

After prefill, scans `conversation_tokens` for injected `<think>` and seeds state:
```rust
if depth > 0 {
    think_state.in_think = true;
    think_state.think_blocks_seen = 1;
}
```

---

## What Breaks

The model's embedding matrix (`token_embd`) is allocated at the MQ4's vocab_size (151646). Token IDs 151657/151658 are **out of bounds**:

- The embedding lookup returns garbage/zero vectors
- The model receives garbage input for `<think>` and cannot enter a think block
- The ban on `</think>` works (correctly writes -INF to logits buffer offset) but the model can't reason inside a think block because it never properly entered one

**Symptoms:**
- With prefix injection: model emits 1 token (EOS) — confused by garbage embedding
- Without prefix injection: model writes "Here's a thinking process:" as visible text

---

## Required Fix: MQ4 Vocabulary Patcher

### What needs to happen:

1. **Add tokens 151646–151658** to the MQ4's vocabulary metadata
2. **Add embedding rows** for each new token to `token_embd.weight` in the MQ4 file
3. Update `config.vocab_size` in the MQ4 header to 151659

### New token assignments:

| ID | Token | Purpose |
|----|-------|---------|
| 151646–151656 | (reserved) | Padding — Qwen's native vocab has these as rare tokens |
| 151657 | `<think>` | Think block opener |
| 151658 | `</think>` | Think block closer |

### Patching approach:

The MQ4 format stores `token_embd` as the first weight (Q8_0 quantized). For Qwen 3.6 27B:
- hidden_dim = 5120
- token_embd currently: 151646 rows × 5120 elements × Q8_0 packing
- Needs: 151659 rows (13 more)

The patch tool should:
1. Read the MQ4 header JSON, bump `vocab_size` from 151646 to 151659
2. Append 13 rows of Q8_0-quantized embedding data to `token_embd`
3. For rows 151646–151656: zero-out or copy noise (unused)
4. For row 151657 (`<think>`): initialize from a reasonable embedding (e.g., average of other special tokens)
5. For row 151658 (`</think>`): same
6. Update any weight offset tables

### Verification:

After patching:
```bash
# Should return Some(151657) and Some(151658)
curl -s localhost:11435/v1/chat/completions -d '{
  "model":"qwen3.6:27b",
  "messages":[{"role":"user","content":"47*83=?"}],
  "temperature":0,
  "max_tokens":200,
  "max_think_tokens":4096
}' | jq '.choices[0].message.content'

# Expected: hidden reasoning inside <think>...</think>, then visible answer
# Actual (current): "Here's a thinking process:" monologue or empty output
```

### Alternative: GGUF Remux

Instead of patching the MQ4, re-run the GGUF→MQ4 conversion with a patched GGUF that includes think tokens in its metadata. This is safer than binary-patching the MQ4.

---

## Files Modified

| File | Change |
|------|--------|
| `crates/hipfire-runtime/src/sampler.rs` | `ThinkState`, `merge_think_bans()`, `update_think_state()` |
| `crates/hipfire-runtime/examples/daemon.rs` | Hardcoded IDs, prefix injection, 3 sample sites, vocab_size bump |
| `crates/hipfire-runtime/src/llama.rs` | Logits buffer padded to 151659 |
| `kernels/src/repeat_penalty_argmax_batched.hip` | GPU kernel (DFlash path, working) |
| `crates/hipfire-arch-qwen35/src/speculative.rs` | GPU penalty path (DFlash, working) |
| `crates/rdna-compute/src/dispatch.rs` | Kernel dispatch (DFlash) |
| `crates/rdna-compute/src/kernels.rs` | Kernel registration (DFlash) |
| `cli/index.ts` | Budget-aware think cap (DFlash) |
| `docs/ar-think-policy-bug-report.md` | Initial bug report (pre-diagnosis) |

All on branch `feature/think-policy-ar-dflash`, NOT master.

---

## Next Steps

1. Write MQ4 vocabulary patcher (or fix GGUF→MQ4 conversion to preserve all tokens)
2. Apply patch to `~/.hipfire/models/qwen3.6-27b.mq4`
3. Rebuild daemon with `thinking_allowed` check removed (no longer need hardcoded IDs)
4. Test: thinking=ON → hidden `<think>` reasoning → clean visible answer
5. Merge feature branch to master
