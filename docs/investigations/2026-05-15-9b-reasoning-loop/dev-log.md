# Issue #258 investigation log — 9B Qwen3.5 chat-mode reasoning loop

Status: **RESOLVED — root cause: daemon's default `repeat_penalty=1.3`.** Fix: flip default to `1.0` (matches llama.cpp + HF transformers). See PR #267 and the chronology below for the trail of dead-end hypotheses.

This log is structured so other agents can: (1) reproduce the bug, (2) understand which hypotheses are falsified and why, (3) avoid re-walking the dead ends.

---

## Goal

Trace the cause of issue #258's "chat-mode reasoning loop" pathology on Qwen3.5-9B and produce a minimal fix. Failure mode: on multi-step math/reasoning prompts with chat template + thinking enabled, hipfire produces "Let me re-read the prompt carefully... did I miss a digit?" self-doubt loops that never converge to an answer. llama.cpp on the same model + same prompt at the same temperature produces clean structured chain-of-thought.

## Canonical reproducer

P6 prompt (committed at `benchmarks/prompts/trains-meet.txt`):

```
If a train leaves station A at 9am traveling 60 mph, and another leaves
station B at 10am traveling 80 mph toward A, and the stations are 280
miles apart, at what time do they meet? Show your work briefly.
```

Correct answer: **11:34 AM**.

Quants used during the investigation (all hipfire-native, built on /local since they're large):

- `qwen3.5-9b.mq4` — May 6 pre-AWQ plain MQ4
- `qwen3.5-9b.mq4-awq-current` — F2 AWQ at α=0.5
- `qwen3.5-9b.mq6`, `qwen3.5-9b.mq8` — higher bit-rate MQ family
- `qwen3.5-9b.q8` — unrotated Q8 (binary-compatible block layout with GGML Q8_0), built via `hipfire-quantize --format q8`
- `qwen3.5-9b.q8-f16lm` — Q8 weights + F16 lm_head (`HIPFIRE_QUANTIZE_LM_HEAD_F16=1`)
- `qwen3.5-9b.q8f16-fresh` — fresh Q8F16 with F16 embeddings

Llama.cpp ground truth: `Qwen3.5-9B-Q8_0.gguf` from `/data/models/unsloth/Qwen3.5-9B/`, run via `mixa3607/llama.cpp-gfx906:b9010-rocm-6.3.3` docker.

Repro recipe (master at the time of this writeup):

```bash
./target/release/hipfire-quantize \
    --input /local/hipfire/Qwen3.5-9B-BF16-st \
    --output /local/hipfire/qwen3.5-9b.q8 --format q8

cargo build --release --example daemon --features deltanet

# JSON over stdin to the daemon. Send a load message, wait, then generate
# at temp=0.3 (or 0.0), jinja chat template on, thinking on.
# See drive scripts referenced in the Reproducibility section.
```

With default `repeat_penalty=1.3` (pre-PR-#267 master): infinite self-doubt loop, no answer in 1500 tokens.
With `repeat_penalty=1.0` (post-PR-#267): correct structured chain-of-thought, byte-equivalent to llama.cpp Q8_0 output.

---

## Chronology of hypotheses (each with the experiment that falsified it)

### Day 0 — Initial observation

User reports: hipfire chat-mode "trains meet" prompt loops; llama.cpp BF16 on the same model produces a coherent answer. Reproduces on every quant level (MQ4/MQ6/MQ8/Q8). The loop is a semantic loop ("let me re-read", "did I miss a digit?"), NOT a verbatim token loop — coherence_probe's n-gram detectors don't catch it.

### Day 1 — Variable elimination (each ruled out cheaply)

- **Tokenization parity (PR #232 baseline)**: hipfire's chat-template tokens match what HF's `apply_chat_template` produces. ChatFrame (Plain) and Jinja produce the same byte-level token sequence on a small canonical 44-token P5 reproducer.
- **Quant choice**: tested MQ4-AWQ, MQ4 plain, MQ6, MQ8, Q8F16. All loop. **Not weight quantization.**
- **AWQ pre-scaling**: tested pre-AWQ May-6 quants. Same loop. **Not AWQ.**
- **Engine knobs**: temperature 0.0/0.3/0.8, prompt-normalize on/off, KV mode q8/asym3/asym4. All loop. **Not engine config.**
- **Issue #247 (flash-attn boundary)**: flash triggers at ctx ≥ 2048; P5/P6 context < 200. Not active here.

### Day 1 — Cross-engine ground truth

llama.cpp BF16 on the same model + same prompt + same MI50 hardware **answers correctly**. Implies the loop is hipfire-side, not a model property. **Confirmed via Q4_K_M, Q8_0, and BF16 GGUF builds — all three llama.cpp variants answer correctly.**

### Day 1 — Logit-parity dump (early breakthrough that turned out to be a red herring)

Built `dump_logits_trajectory` (in the PR #232 worktree): teacher-forces hipfire on a llama.cpp greedy trajectory, dumps top-20 logits per position. Initial finding on MQ4 + asym3 KV chat-mode prompt: position-0 KLD(L‖H) = 26.75 with hipfire argmax = `'\n'` vs llama.cpp `'Thinking'`. Concluded "position-0 catastrophic divergence." Most of Phases 4–6 were built on this premise — see below for the actual diagnosis (the position-0 divergence had two causes, one of which was a separate PR #232-specific bug, and neither was the dominant loop driver).

### Day 2 — Phase 4: chat-template vs raw completion

Hypothesis from the position-0 finding: chat-template special tokens (`<|im_start|>`, `<|im_end|>`, `<think>`) trigger something in the forward pass. Tested raw-completion (no chat scaffold) on the SAME token IDs vs chat-templated: raw completion produces near-llama-identical logits (pos-0 KLD = 0.02), chat-templated diverges. Encoding the chat-templated string via `tokenizer.encode()` (not ChatFrame) and feeding into raw-mode produces the same 27-nat divergence — **so the token IDs alone triggered it**, not the ChatFrame code path.

### Day 2 — Phase 5–6: hidden-state per-layer comparison vs HF transformers BF16

Built a one-off HF hidden-state dump script (CPU since gfx906 isn't in ROCm 6.4's rocBLAS) and diffed it against hipfire's `forward_scratch_with_hidden` output per-position per-layer. That one-off script is historical: it relied on `output_hidden_states[N]` semantics that were later retracted in Phase 7.7. Use the corrected tooling listed under Reproducibility for any new hidden-state comparison.

Hypothesis-by-hypothesis falsification chain (each from per-layer or per-channel diffs):

- **RMSNorm precision on low-magnitude rows** — falsified: trailer-token magnitudes are 80-95% of body, not lower.
- **Rotated-embedding-row asymmetry on token 74455 (`assistant`)** — falsified: L0 cos is uniform 0.996 across all positions.
- **RoPE accumulated phase precision** — falsified: drift is NOT monotonic with position. Pos 43 (`<think>`) is less affected than 41/42.
- **Cumulative DeltaNet recurrence drift** — falsified: the bug appeared to manifest at position 0 of decode (before recurrence accumulates) — and later turned out to be falsified for a different reason; see Phase 7.7.

### Day 2 — Phase 6.2: sub-block (attn vs MLP) bisect via HiddenStateRingBuffer

Extended `HiddenStateRingBuffer` with optional `attn_bufs` / `mlp_bufs` channels (production paths leave them None). Captured post-attn and post-MLP residual streams separately per layer × position. Wrote HF reference via `forward_pre_hook` on `post_attention_layernorm`.

Initial reading: mid-stack layers L11+ had attention-step "trailer-vs-body asymmetry" up to −0.035 cos per layer, peaking at L22 (−0.13). Concluded attention sub-block was the bug, propagating through linear-attention recurrence.

**This conclusion was wrong.** See Phase 7.7 for the measurement-bug retraction.

### Day 2 — Phase 6.3: MQ8 + Q8 KV "control" looked clean, then user pointed out Q8 loops too

Re-ran the sub-block diff with MQ8 + Q8 KV. Asymmetry collapsed to ±0.0005 — looked like proof that MQ4 quant noise was the bug. **Retracted** when user pointed out Q8F16 ALSO loops per their earlier gfx1151 result. Verified locally: Q8 weights + Q8 KV cache + Jinja + temp=0.3 on P6 → still loops with hallucinated numbers + self-doubt texture.

### Day 2 — Phase 6.5: the `<think>\n` discovery (PR #232 worktree bug, NOT master)

While building HF reference dumps for the cross-test, found: HF's `apply_chat_template(messages, add_generation_prompt=True)` emits **67 tokens** ending in `[..., <think>, \n]`. PR #232's daemon-side `<think>` injection emits only **66 tokens** ending in `[..., <think>]` (no trailing newline). On a 66-vs-67 test in HF: 66-token sequence's last-position top-1 is `'\n'` (the model wants to emit the missing newline), 67-token sequence's top-1 is `'Here'` / `'Thinking'` (a content token).

Fix: introduced `prompt_frame::open_think_tokens()` returning the canonical `[<think>, \n]` sequence, refactored PR #232's injection sites to use it. End-to-end Q8 + Q8 KV on P5 went from `\n\n<|im_end|>` (2 tokens, immediate-EOS) → "Thinking Process: 1. **Analyze the Request**: ..." (structured response). Position-0 KLD dropped 26.66 → 5.04.

**Scope**: PR #232 worktree only. Master doesn't have this code path (it uses `AssistantPrefix::OpenThink` or Jinja, both of which already emit `<think>\n` correctly). End-to-end master Q8 doesn't catastrophically fail at first decode — but **still loops mid-trace**. That's "bug B."

### Day 3 — Phase 7.1–7.3: bug B (master) candidates ruled out

After Phase 6.5 closed the PR #232-specific bug, focused on master's bug B. Cross-implementation control on the same MI50 + Jinja + temp=0:

| Setup | Result |
|---|---|
| llama.cpp Q4_K_M | ✅ Correct: 11:34 AM, structured CoT |
| llama.cpp Q8_0 | ✅ Correct |
| HF transformers BF16 (CPU greedy) | ✅ Correct (matched llama.cpp prefix for first 181 tokens) |
| hipfire master MQ4 / Q8 / Q8 + F16 lm_head, any KV mode | ❌ Loop |

llama.cpp at 4-bit cleanly answers; hipfire at 8-bit fails. **Bug is hipfire's forward-pass / decode loop, not model nor quant.**

Falsified:
- Sampling-step noise — greedy (temp=0) loops identically to temp=0.3
- KV cache quantization — Q8 KV reproduces the same way as asym3
- Cumulative attention drift — teacher-forced KLD stays at noise floor through pos 399
- lm_head precision — cherry-picked PR #242 (F16 lm_head) + gfx906 dequant-on-load fix. Same teacher-forced KLD (0.0076 vs 0.0075). Same loop end-to-end.

### Day 3 — Phase 7.2: teacher-forced trajectory diff (solid result, but partial picture)

Captured llama.cpp Q8_0 greedy trajectory via `/v1/chat/completions` with `logprobs=true, top_logprobs=5, temperature=0`. Ran hipfire `dump_logits_trajectory` teacher-forced on those 400 token IDs.

Result:
- **398/400 argmax agreement (99.5%)** hipfire vs llama.cpp
- Mean KLD(L‖H) = **0.0075** (effectively noise floor)
- 2 argmax disagreements at near-tied positions (pos 210, 227)

Example at pos 210:

| | P('.') | P(' when') |
|--|------:|----------:|
| llama.cpp Q8 | 0.4302 | 0.4298 |
| hipfire Q8   | 0.4082 | 0.4427 |

Hipfire shifts ~3.5% softmax mass between top-1 and top-2 of a tied pair. Concluded: small per-position bias (~0.01–0.1 nat) flips argmax at near-ties, cascading under AR decode.

**This finding is correct — the 99.5% agreement is real. But it didn't explain the loop end-to-end** because the actual AR divergence is at pos 25, not pos 210 (see Phase 8.1).

### Day 3 — Phase 7.5–7.6: per-channel hidden-state diff (later retracted)

Followed the bias hypothesis: dumped pre-output-norm `s.x` per position from hipfire, diffed against HF's `hidden_states[-1]`. Saw uniform 1.77× per-channel scaling with channel-3994 outlier at ~4×. Bisected to L31's MLP showing ratio 2.07× while L0–L30 showed ratio 1.000.

### Day 3 — Phase 7.7: RETRACT 7.5/7.6 — measurement bug

`output_hidden_states=True` in HF returns a tuple of length N+1. **`hidden_states[N]` (the LAST index, N=num_layers) is POST-final-norm in Qwen3.5's TextModel**, because the model's forward pass applies `self.norm(hidden_states)` after the layer loop. Hipfire's `s.x` dump at the end of L31 is PRE-final-norm. The 2× ratio was the magnitude difference between pre-norm and `model.norm`-applied state, not a kernel bug.

Verified by monkey-patching `Qwen3_5DecoderLayer.forward` at L31 to capture the true post-block tensor: hipfire's L31 post-MLP matches the true HF post-MLP at **cos 0.99943, ratio 1.0011**. All 32 layers are correct.

**All Phase 6.2 / 7.5 / 7.6 per-layer / per-channel localization claims are RETRACTED.** The forward pass IS correct (consistent with Phase 7.2's logit-space measurement). The bug is elsewhere.

Lesson: `output_hidden_states` semantics vary across model families and transformers versions. **Always verify hidden-state-tuple semantics by monkey-patching one layer's forward**. Don't trust pre-hooks on layernorms or the tuple alone for cross-implementation comparisons of pre-norm residual streams.

### Day 3 — Phase 8.1: ROOT CAUSE FOUND — `repeat_penalty=1.3` default

Switched to direct AR-greedy comparison instead of teacher-forced: run hipfire master Q8+Q8KV at temp=0 on P6, capture the actual generated token IDs, diff against llama.cpp's 400-token Q8_0 greedy IDs.

Also ran HF transformers BF16 greedy on the same prompt (CPU, ~13 min for 400 tokens) for an independent reference.

| comparison | first AR divergence | agreement rate |
|---|---:|---:|
| HF BF16 greedy vs llama.cpp Q8_0 greedy | pos **181** (stylistic `' *'` vs `' **'`) | 181/400 prefix exact |
| hipfire master Q8 AR-greedy vs llama.cpp Q8_0 greedy | pos **25** | only 7% match |
| hipfire master Q8 AR-greedy vs HF BF16 greedy | pos **25** (same token) | only 7% |

HF and llama.cpp produce identical structured chain-of-thought through pos 180. Hipfire produces the same first 25 tokens, then at pos 25 picks `' Train'` (id 25291) instead of `' **'` (id 2972).

**The smoking gun:** at pos 25 with the same first-25-token prefix, hipfire's teacher-forced top-1 IS `' **'` (logit 26.88, with `' Train'` second at 24.75 — a clean 2-logit gap, no near-tie). Under AR-greedy the same forward at the same position emits `' Train'` instead. **Same input, same forward, different output → post-forward logit modification.**

`daemon.rs:859`:

```rust
let repeat_penalty = msg.get("repeat_penalty").and_then(|v| v.as_f64()).unwrap_or(1.3) as f32;
let repeat_window = msg.get("repeat_window").and_then(|v| v.as_u64()).unwrap_or(128) as usize;
```

Token 2972 (`' **'`) appeared at output pos 15 (`1.  **Analyze...`). At pos 25 it's 10 tokens later, well inside the 128-token window. Repeat penalty divides logit 26.88 by 1.3 → 20.68. Meanwhile `' Train'` (id 25291, unpenalized) stays at 24.75. **Now `' Train' > ' **'` → argmax flips → trajectory derails into the self-doubt attractor.**

llama.cpp default `--repeat-penalty 1.0` (off). HF `generate()` default `repetition_penalty=1.0` (off). Both produce clean output.

Verified: setting `"repeat_penalty": 1.0` explicitly in the daemon's generate request makes hipfire produce byte-equivalent structured CoT to llama.cpp + HF.

**Fix: change daemon default `1.3` → `1.0`.** Single-line patch, comment added. Shipped as PR #267.

---

## Summary of falsified hypotheses (for future agents who think they've found one of these)

- ❌ Kernel correctness bug in any single sub-block (Phase 7.2 + 7.7 — forward pass matches HF/llama.cpp at noise floor)
- ❌ Weight bytes differ between hipfire Q8 and GGUF Q8_0 (Phase 6.5 sanity — algorithm bit-near-identical)
- ❌ KV cache quantization (Q8 / asym3 / asym4 all loop the same way)
- ❌ lm_head precision (Q8 vs F16 lm_head — same loop)
- ❌ Cumulative drift over many decode positions (teacher-forced KLD constant through pos 399)
- ❌ Sampling-step noise / temperature (greedy reproduces)
- ❌ Cos-direction hidden-state diff localizes a layer (all such measurements either matched HF or were comparing wrong references — see Phase 7.7)
- ❌ Embedding lookup precision (post-embed cos 0.9999 vs HF at all positions)
- ❌ Mid-stack attention bias on trailer tokens (Phase 7.7 retract — measurement artifact)

## Still-open small issues (not bug-B-blocking)

- Hipfire-vs-llama.cpp residual logit bias of ~0.01–0.1 nat per position at near-tied positions, even after the repeat-penalty fix. Consistent with kernel-implementation precision differences (hipfire's fp32 inner-loop reduction vs llama.cpp's CUDA reduction order). At `repeat_penalty=1.0` these no longer flip argmax on standard reasoning prompts, but a future agent investigating perfect-parity may want to revisit.
- L31 final RMSNorm precision: separate (small) effect at the end of the residual stack — body and trailer both drop ~0.04 cos at the final post-MLP step. Doesn't affect bug B; flagged for future audit.

---

## Reproducibility

### Files in this directory

- `dev-log.md` — this document (single source of truth for the investigation)
- `p5_chat_tokens.json` — canonical 44-token chat-template encoding of P5 (single-train prompt). Bytes-exact reproducer for the Phase 6 hidden-state work

The Phase 6.2 / 7.5 / 7.6 diff scripts (per-channel cos, layer×position grid, 3-channel decomposition, MLP sub-stage bisect, etc.) were ~20-50 line numpy programs that became misleading once Phase 7.7 retracted their underlying comparison. They are not committed here. The HFHS binary format they consumed is documented below for historical context and for reuse with the corrected dump tools already on master.

### Current hidden-state dump tooling

- `crates/hipfire-runtime/examples/dump_qwen35_hidden_states.rs` — hipfire forward through `forward_scratch_with_hidden`, reusing `HiddenStateRingBuffer` with `extract_layers` set to every layer.
- `scripts/dump_hf_hidden_states.py` — HF transformers BF16 oracle. It uses a forward-pre-hook on the final model norm so the last layer is pre-final-norm and matches hipfire's capture point. Do not replace this with a plain `hidden_states[1:]` dump; `hidden_states[n_layers]` is post-final-norm for Qwen3.5.
- `scripts/compare_hidden_states.py` — offline per-layer cosine + relative-L2 comparator for two HFHS dumps.

### HFHS binary format

```
magic 8B = b"HFHS\0\0\0\0"
n_layers u32
n_pos    u32
hidden_dim u32
_reserved u32
body: n_layers × n_pos × hidden_dim f32 row-major
```

The current dump tools write that format. A minimal reader template:

```python
import struct, numpy as np

def load_hfhs(path):
    with open(path, 'rb') as f:
        m = f.read(8); assert m == b"HFHS\0\0\0\0"
        nL = struct.unpack('<I', f.read(4))[0]
        nP = struct.unpack('<I', f.read(4))[0]
        hd = struct.unpack('<I', f.read(4))[0]
        f.read(4)  # reserved
        return np.frombuffer(f.read(), dtype=np.float32).reshape(nL, nP, hd)

a = load_hfhs(hipfire_dump_path)
b = load_hfhs(hf_dump_path)
# then per-position or per-layer numpy cos / rms / diff
```

For the FA-stages dump format (used in Phase 7.6 MLP-bisect, also retracted), each record was `32B header (8× u32: layer_idx, pos, stage_id, n_elems, 0, 0, 0, 0)` then `n_elems` f32 values. The hipfire-side writer (`dump_fa_stage`, env-gated via `HIPFIRE_DUMP_FA_STAGES` + `HIPFIRE_DUMP_DN_LAYER`) was branch-local historical tooling and is not on current master.

### Key reproducer scripts (build on master, no hipfire patches needed)

```bash
# 1. Quantize Qwen3.5-9B to Q8 (binary-compatible with GGML Q8_0 blocks)
./target/release/hipfire-quantize \
    --input /local/hipfire/Qwen3.5-9B-BF16-st \
    --output /local/hipfire/qwen3.5-9b.q8 --format q8

# 2. Build daemon
cargo build --release --example daemon --features deltanet

# 3. Drive daemon (jinja chat template + thinking on + greedy decode)
DAEMON=./target/release/examples/daemon
PROMPT_JSON=$(python3 -c "import json; print(json.dumps(open('benchmarks/prompts/trains-meet.txt').read()))")
(
  echo "{\"type\":\"load\",\"model\":\"/local/hipfire/qwen3.5-9b.q8\",\"params\":{\"max_seq\":4096,\"kv_mode\":\"q8\"}}"
  sleep 35
  echo "{\"type\":\"generate\",\"id\":\"r1\",\"prompt\":$PROMPT_JSON,\"temperature\":0.0,\"max_tokens\":1500}"
  sleep 300
  echo "{\"type\":\"unload\"}"
  sleep 2
) | env HIPFIRE_JINJA_CHAT=1 "$DAEMON" 2>&1

# With repeat_penalty=1.3 default (pre-PR-#267 master): loop pathology
# With "repeat_penalty": 1.0 in the generate request: clean output
```

Llama.cpp ground truth via docker (single-turn `-no-cnv -st`):

```bash
docker run --rm --device /dev/dri --device /dev/kfd \
    --group-add video --group-add render \
    -e HIP_VISIBLE_DEVICES=0 -e ROCR_VISIBLE_DEVICES=0 \
    -v /data/models/unsloth/Qwen3.5-9B:/models:ro \
    --entrypoint /app/llama-cli \
    mixa3607/llama.cpp-gfx906:b9010-rocm-6.3.3 \
    -m /models/Qwen3.5-9B-Q8_0.gguf \
    --jinja --temp 0.3 -ngl 99 -st \
    -n 1500 -p "$(cat benchmarks/prompts/trains-meet.txt)" --no-display-prompt
```

HF transformers BF16 greedy (CPU, since ROCm 6.4 dropped gfx906 from rocBLAS):

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained('/local/hipfire/Qwen3.5-9B-BF16-st')
model = AutoModelForCausalLM.from_pretrained(
    '/local/hipfire/Qwen3.5-9B-BF16-st', dtype=torch.bfloat16,
    low_cpu_mem_usage=True).eval()

user = open('benchmarks/prompts/trains-meet.txt').read().rstrip('\n')
text = tok.apply_chat_template(
    [{"role":"user","content":user}], tokenize=False, add_generation_prompt=True)
ids = tok(text, add_special_tokens=False)['input_ids']
with torch.no_grad():
    out = model.generate(
        torch.tensor([ids]), attention_mask=torch.ones(1, len(ids)),
        max_new_tokens=400, do_sample=False, use_cache=True)
print(tok.decode(out[0][len(ids):], skip_special_tokens=False))
# ~13 min wall on 30 GB RAM. Produces correct structured CoT.
```

---

## Lessons for future bug hunts on hipfire

1. **AR-decode path has post-forward logit modifications** (sampler, repeat penalty, think-state masking). Always test these BEFORE diving into the forward pass.
2. **Teacher-forced trajectory dumps don't apply the sampler** — they show the raw forward top-K. AR-greedy CAN diverge from teacher-forced top-1. If you see that, look at the daemon's sampler config, not the forward pass.
3. `output_hidden_states[N]` from HF transformers can be post-final-norm depending on the model family's behavior. **Verify by monkey-patching one layer** before trusting cross-implementation hidden-state diffs.
4. Cos-direction diff is direction-only; it can be 0.998 while the logit projection has 26-nat KL divergence. **Compare in logit space (or argmax space) for chat-mode behavior issues, not in hidden-state space.**
5. Cross-implementation control (llama.cpp + HF) early and often. If both agree but hipfire disagrees, the bug is on the hipfire side — and the early variable-elimination steps are cheaper than per-layer hidden-state archaeology.

Last updated: 2026-05-16, bug closed via PR #267.
