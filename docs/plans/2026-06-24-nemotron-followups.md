# Nemotron-H follow-ups + quantized-serving roadmap

Status as of 2026-06-25: **nemotron_h (Nano-4B) serves end-to-end** (arch_id 14,
loads and streams tokens), but coherent chat output remains blocked by FU1. The
original mq4 artifact flips a close Hipfire-f32/q8 `<|im_end|>` boundary into a
newline loop; q8 tracks Hipfire f32 but still stops immediately or samples
incoherently. `/home/sadara/vllm0.22.1` is now a coherent external reference for
the byte-identical closed-think 2+2 prompt: it generates `2 + 2 equals 4.` with
first token `2`, while Hipfire/native-HF choose immediate `<|im_end|>`. The Lyra
ROCm stack with real installed `mamba_ssm` kernels now corroborates the
Transformers/Hipfire boundary, so the remaining FU1 question is vLLM-specific
rather than a broken local Mamba fallback. `benchmarks/nemotron/dump_hf_reference.py`
remains the repeatable local Python/native-Mamba first-token and per-layer
reference for Hipfire's current boundary, with `--mamba-import real` available
for Lyra fast-kernel checks; `benchmarks/nemotron/run_vllm_reference.py` captures
the vLLM reference. `--format mq4` now protects Nemotron residual writers as Q8,
so fresh Nano-4B protected-mq4 artifacts follow Hipfire f32/Q8 instead of the
original newline flip. See
`docs/plans/2026-06-24-nemotron-fu5-status.md` for the current evidence and
`docs/plans/2026-06-24-nemotron-h-mamba2.md` for N0–N5.
This doc plans the six follow-ups, each self-contained and grounded in the
current code/checkpoints. Effort tags: **S** ≈ hours, **M** ≈ 1–2 days, **L** ≈
several days.

## Recommended sequencing (dependencies)

```
1. chat-template refinement      (S)  ── done; FU1 coherence now needs vLLM-vs-Hipfire bisect
2. HF-ref numeric bisect         (M)  ── lock correctness before scaling
3. quantizer compatibility       (M) ─┐ the "real-use memory" track
4. loader compatibility          (L) ─┘ (16GB f32 → ~4GB mq4); 4 depends on 3
5. N6 chunked-prefill + q8 state (L)  ── throughput (independent of 3/4)
6. Nano-30B (MoE 'E' block)      (L)  ── new model/feature (independent)
```
1, 2, 5, 6 are mutually independent; 4 depends on 3. The chat-template sub-work
in 1 is done; the remaining FU1 coherence work should use the vLLM reference to
bisect the first-token divergence before more sampling or prompt-policy tuning.
Then pick the track by priority: correctness (2), memory (3+4), speed (5), or
scale (6).

---

## 1. Chat-template refinement (S)

**Current state (2026-06-25).** The concrete template items below are done in the
live code: arch 14 resolves EOS from tokenizer `<|im_end|>`, Jinja is the default
when the embedded template exists, and `hipfire chat` forwards thinking-off
controls (`closed_think`, `max_think_tokens = 1`) just like the HTTP path.

This did **not** resolve coherence. Current `target/debug/hipfire-daemon` with
the original mq4 Nano-4B artifact emits newline-only output. A q8 artifact tracks
Hipfire f32 at the first-token boundary and avoids the newline loop, but greedy
closed-think generation stops immediately and a sampled reasoning-on prompt is
still incoherent. The local Python/native-Mamba reference now matches Hipfire f32
on the closed-think 2+2 prompt (top-5 `[11, 1010, 1058, 1050, 1319]`, logit
relative delta 0.0221). A Lyra real-Mamba Transformers run with
`--mamba-import real --mamba-reference remote` produces the same first-token
top-5, so the local Mamba fallback is not the cause. Fresh protected-mq4
artifacts also match that boundary because Nemotron residual writers are
promoted to Q8. vLLM 0.22.1 gives the coherent production-style boundary for
the same prompt ids: first-token top-5 `[1050, 31035, 1052, 2757, 16489]` and
generated text `2 + 2 equals 4.`. Treat the remaining FU1 task as "bisect vLLM
vs Transformers/Hipfire," not as another chat-template tweak.

**Problem (root-caused).** The N5 serve test stopped after 8 tokens ("We need to
answer in one sentence.") — an early/odd halt. Cause is concrete:
- Tokenizer ids: `2 = </s>`, **`11 = <|im_end|>`**, `10 = <|im_start|>`.
- `config.eos_token_id = 2` (`</s>`), but the **chat template delimits assistant
  turns with `<|im_end|>` (11)**, and `tokenizer_config.eos_token = <|im_end|>`.
- `generate_nemotron` stops on `cfg.eos_token_id = 2`, NOT 11. So the model's
  real turn-end token (11) is not a stop, and token 2 (`</s>`) fires at the wrong
  place. Nano-3 is also a **reasoning** model (the template has think/tool macros)
  — its output is a `<think>`…`</think>` trace then the answer.

**Approach.**
1. Resolve the serving EOS from the tokenizer's `<|im_end|>` id (11), not
   `config.eos_token_id`. Add `<|im_end|>` (and keep `</s>`) to the stop set.
   Plumb through `NemotronModel::config().eos_token_id` → set it to the
   `<|im_end|>` id at load (look it up via the tokenizer / `added_tokens`), or add
   `stop_sequences`/extra-EOS in `generate_nemotron`'s `GenerateCtx`.
2. Default arch-14 to the **jinja chat template** (it ships a good one) rather
   than the Plain `ChatFrame`. Today `generate_nemotron` only uses jinja under
   `HIPFIRE_JINJA_CHAT=1`; flip the default for arch 14 (the template is present
   and is the correct ChatML/`<|im_start|>`/`<|im_end|>` framing).
3. Think-mode: route/strip the `<think>` block via the existing think-mode
   infra (`max_think_tokens`, `strip_think`), as qwen35 does — Nano-3 emits
   reasoning traces.

**Files.** `crates/hipfire-serving-core/src/generate_arch.rs::generate_nemotron`;
`crates/hipfire-arch-nemotron/src/lib.rs` (eos resolution) or the load path
(`load.rs` arch-14 branch) to set the chat-eos id.

**Validation.** The old validation expectation ("serve 3-4 prompts and expect
clean answers") is not met by Hipfire on ROCm today. New validation target: use
vLLM as the coherent external reference, then compare Hipfire prompt IDs,
first-token logits, and generated text against that reference.

---

## 2. HF-reference numeric bisect (M)

**Goal.** Definitively confirm hipfire's nemotron forward matches HF
`NemotronHForCausalLM` per-layer (beyond the coherent-serve + gpu-vs-cpu evidence,
which only validate against *our own* CPU oracle / conventions).

**Current state.** `benchmarks/vision/dump_hf_reference.py` exists but is
vision-scoped. No text/nemotron HF-ref path.

**Approach.**
1. Python (tooling only, not hot path): load HF model with `trust_remote_code`,
   register forward hooks on each `NemotronHBlock` output and the final logits
   for a fixed token sequence; dump to `.npz`. This now lives in
   `benchmarks/nemotron/dump_hf_reference.py`.
2. Rust example in `hipfire-arch-nemotron` (gpu, gated on the checkpoint): run the
   same tokens through `NemotronModel`, dumping each block's input/output (add a
   debug hook to `forward_gpu`), compare against the HF dump, report the
   **first-divergence layer** + max abs/rel delta. Mirror the multi-family bisect
   pattern from `dump_hf_reference.py`.

**Reuse.** `model.rs`'s per-block structure (already a clean loop); the existing
HF-dump harness shape.

**Risks.** HF runs in bf16; compare with tolerance (hipfire is f32 → expect HF as
the lower-precision side). 4B model is heavy to run once (acceptable, one-shot).

**Validation.** Expect ≤ ~1e-2 rel per-layer (bf16 noise). A clean match retires
all "convention re-verify" caveats; a divergence pinpoints the exact block/op.

---

## 3. Quantizer compatibility (M)

**Goal.** `hipfire-quantize` produces a nemotron_h `.hfq` (e.g. mq4) so the model
serves at ~4 GB instead of ~16 GB f32.

**Current state.** `crates/hipfire-quantize/src/main.rs` auto-detects `arch_id`
(stamped into the HFQ header) and quantizes linear weights via
`quantize_hfq_source_tensor` (Oq4/mq4/…), keeping norms/scales as sidecars. arch
14 is not yet handled in the tensor-classification logic.

**Approach.**
1. Ensure arch detection stamps **arch_id 14** for a nemotron config (mirror
   `derive_arch_id`; the quantizer reads the same architecture string).
2. Tensor classification for nemotron (by name):
   - **Quantize (linear, the bulk):** `*.mixer.in_proj/out_proj.weight`,
     `*.mixer.up_proj/down_proj.weight`, `*.mixer.q_proj/k_proj/v_proj/o_proj.weight`,
     `lm_head.weight`, `backbone.embeddings.weight`.
   - **Keep f16/f32 (small + recurrence-sensitive):** `*.conv1d.weight`,
     `*.conv1d.bias`, `*.A_log`, `*.D`, `*.dt_bias`, `*.norm.weight` (block +
     mixer/gated norm), `backbone.norm_f.weight`.
3. **Sensitivity:** the Mamba-2 `in_proj` drives dt/B/C, and the projection-back
   writers feed the residual stream. Current local evidence says uncalibrated
   mq4 is too noisy at the first generated token for Nano-4B, while q8 matches
   f32. The live `--format mq4` policy therefore promotes Nemotron residual
   writers (`out_proj` / `down_proj` / `o_proj`) to Q8 until a calibrated
   imatrix/AWQ/Lloyd policy proves true 4-bit coherence.

**Reuse.** The existing quantize codecs/mq4 pipeline; per-arch name-pattern skip
logic.

**Risks.** Quantizing a recurrence tensor (conv1d/A_log) would corrupt the SSM —
the keep-list above is the guard. Verify against the f32 serve.

**Validation.** Quantize Nano-4B → mq4 `.hfq`; serve (needs #4); coherence parity
vs the f32 serve (coherence-gate). Effort **M**, but only useful paired with #4.

---

## 4. Loader compatibility (HFQ / quantized) (L)

**Goal.** Load nemotron from a `.hfq` (quantized) and serve with the weights kept
quantized on-GPU (the actual memory win), not dequantized to f32.

**Current state.** `loader.rs` reads **BF16 safetensors → f32**
(`NemotronWeights`), and `NemotronModel` uploads plain f32 `GpuTensor`s; the block
structs (`Mamba2BlockGpu`, `MlpRelu2Gpu`, `NemotronAttnGpu`) hardcode
`gemv_f32`. The HFQ path (`hfq.rs::load_weights_hfq`) is separate and per-arch;
nemotron has none.

**Approach (staged).**
- **4a (cheap, low value):** read `.hfq` tensors via `ModelSource` (it already
  abstracts hfq vs safetensors) and **dequantize to f32** into `NemotronWeights`.
  Works immediately, but keeps the 16 GB f32 footprint — only useful to validate
  the `.hfq` round-trips.
- **4b (the real win, invasive):** make the block GPU structs hold **runtime
  quantized linear weights** (the llama/qwen35 `dispatch_ref()` weight type) and
  call the dispatched gemv (mq4/q8) instead of `gemv_f32`. The recurrence tensors
  (conv1d/A_log/D/dt_bias/norms) stay f32. Run `preflight_gemv_dtypes` on the
  linear set. This is a refactor of `block_gpu.rs`/`mlp.rs`/`attn.rs` to a generic
  weight type + the dispatched gemv (the `execute_steps`/`GemvInput` path llama
  uses).

**Reuse.** llama/qwen35 quantized-gemv path (`hipfire_dispatch` `Step::Gemv` +
`dispatch_ref`), `preflight_gemv_dtypes`, the `ModelSource` hfq reader.

**Risks.** Invasive (every block's linear call changes); the f32 gpu-vs-cpu tests
must be re-pinned to the quantized path with looser tolerance. Keep the f32 path
as the reference floor.

**Validation.** Load mq4 `.hfq`; serve; coherence vs f32; ~4× smaller resident
footprint. Depends on #3.

---

## 5. N6 — chunked-SSD prefill + q8 SSM state (L)

**Goal.** Throughput: replace the per-token prefill loop and shrink recurrent
state bandwidth.

**Current state.** `SimpleAr::prefill` now routes through
`NemotronModel::prefill_batched` for f32 and supported HFQ artifacts. The
protected mq4 HFQ gate matches the per-token decode loop
(max|Δlogit|=1.29e-5, argmax match). A release fresh-process gfx1151 benchmark
on `/tmp/nano4b-mq4-protected.hfq` measured:
- seq=128: batched 51.7 tok/s vs decode-loop 38.1 tok/s, speedup=1.36x.
- seq=256: batched 50.2 tok/s vs decode-loop 38.1 tok/s, speedup=1.32x.

The remaining bandwidth item has an opt-in implementation:
`HIPFIRE_NEMOTRON_SSM_STATE=q8` stores the Mamba-2 SSM state as int8 plus
per-row f32 scales. It preserves the prefill/decode handoff on the q8 path
(synthetic block max|Δ|=2.98e-8; protected HFQ model max|Δlogit|=2.38e-4,
argmax match) and protected HFQ still tracks safetensors under q8 state
(cosine 0.999910, argmax match). FP32 remains the default until longer-generation
quality evidence justifies default promotion.

**Approach.**
1. **Batched prefill** (done for the current N6 path): the Mamba/conv/MLP blocks
   and NoPE attention now have batched prefill coverage, and HFQ `gemm_seq`
   supports the protected mq4 path.
2. **q8 SSM state** (opt-in done): quantize `h` to q8 between steps (mirror the
   GDN q8-state pattern) to cut state memory/bandwidth. Remaining decision:
   default promotion after longer-generation evidence.

**Reuse.** qwen35 `pflash` chunked prefill structure; GDN q8-state pattern;
`attention_*_batched_masked` for the attention blocks; `conv1d` batched variant.

**Risks.** Any future true inter-chunk recurrence **must be bit-faithful to the
decode recurrence** — the discriminator is gpu-vs-cpu of chunked-prefill logits
vs the validated per-token `forward` over the same tokens. (See the SSD CPU
oracle `ssd.rs` / `block.rs` — extend it with the chunked form as the test
oracle.)

**Validation.** Batched prefill logits now match the per-token decode loop for
f32, protected mq4 HFQ, and opt-in q8 SSM state. The fresh-process benchmark
entrypoint is
`cargo run --release -p hipfire-arch-nemotron --example bench_prefill_hfq_gpu`;
the q8-state gate is
`PATH=/usr/lib/llvm-21/bin:$PATH cargo run -p hipfire-arch-nemotron --example test_block_q8_state_gpu`.

---

## 6. Nano-30B — MoE ('E') block (L)

**Goal.** Serve NVIDIA-Nemotron-3-Nano-30B-A3B (present in `/srv/huggingface`).

**Current state (grounded from its config).** Same `nemotron_h` arch but
hidden=2688, 52 layers, pattern **`MEMEM*EMEM…`** — introduces a **new block
char `E` (MoE FFN)** replacing the dense `-` MLP. MoE: `n_routed_experts=128`,
`num_experts_per_tok=6` (top-6), `moe_intermediate_size=1856`, plus
`n_shared_experts=1` / `moe_shared_expert_intermediate_size=3712` (shared
expert). `BlockKind::Moe`, MoE config parsing, safetensors name mapping, and an
isolated decode-first `MoeRelu2Gpu` block are now implemented. The live 30B
safetensors index uses split 2D per-expert tensors:
`backbone.layers.L.mixer.experts.E.{up_proj,down_proj}.weight`, plus
`mixer.gate.weight`, `mixer.gate.e_score_correction_bias`, and
`mixer.shared_experts.{up_proj,down_proj}.weight`. The first full 30B ingress
artifact now exists at
`/home/sadara/.hipfire/models/nemotron-3-nano-30b-a3b-mq4.hfq` (sha256
`3660ae47c8f7309110d85ee2c013629cb6437b2fdfbb76d41a1a7cb3a49fe2f6`,
25,681.0 MB written).

**Approach.**
1. **Done:** `BlockKind::Moe` ('E'); `parse_block_pattern`; `mixer_profile`
   treating MLP/MoE as FFN-only; `rejects_unknown_block_char` moved off `E`.
2. **Done:** parse MoE config fields (routed/shared expert counts, top-k,
   `moe_intermediate_size`, router grouping and route scaling).
3. **Done for decode correctness:** `MoeRelu2Gpu` block: router GEMV → sigmoid →
   bias-aware top-k/normalize/scale, shared ReLU² expert, selected routed ReLU²
   experts. This reuses existing `MlpRelu2Gpu` blocks and the existing
   DeepSeek/LFM2-style top-k router primitive. The modeling source confirms
   ReLU² MLP experts, not SwiGLU.
4. **Done for decode ingress:** loader/model arms for safetensors and HFQ tensor
   names are wired. `hipfire-quantize` also classifies split Nemotron MoE weights
   as quantizable, keeps `gate.e_score_correction_bias` out of lossy
   quantization, and Q8-protects `*.mixer.gate.weight` routers.
5. **Done for artifact ingress:** rebuilt `hipfire-quantize --format mq4`
   produced the real Nano-30B HFQ artifact. Quantization summary:
   31,577,940,288 total params, mean quant error 0.00094749, max quant error
   0.05048829, 25,681.0 MB written. `test_load_nano30b_hfq` loaded that artifact
   on gfx1151 and decoded two tokens with finite logits.
6. **Remaining:** route the real 30B artifact through daemon/server generation,
   collect coherence evidence, and add a batched expert-sorted MoE prefill path
   if 30B prefill throughput matters. Current `can_batched_prefill()` is false
   when any `E` block is present. The artifact policy is still an ingress
   policy, not a quality-promoted calibration policy; expert promotion needs
   router-hit and quality deltas.

**Reuse.** qwen35/lfm2moe MoE kernels + routing (the big lever); the existing
M/*/- blocks unchanged.

**Risks.** The routing convention has been verified from
`modeling_nemotron_h.py`: sigmoid scores, selection uses
`score + e_score_correction_bias`, selected scores are normalized when
`norm_topk_prob=true`, and then scaled by `routed_scaling_factor`. The current
decode path assumes the observed Nano-30B single-group router (`n_group=1`,
`topk_group=1`). 30B is bigger (needs #3/#4 quantization for comfortable
residency; A3B = 3B active so decode is cheap, but BF16 weights are ~63 GB —
**mq4/q8 HFQ effectively required**, so this pairs with #3/#4).

**Validation.** `cargo run -p hipfire-arch-nemotron --example test_moe_gpu`
passes on gfx1151 with max|Δ|=4.47e-8 against the CPU oracle.
`cargo run -p hipfire-arch-nemotron --example test_load_nano30b_hfq --
/home/sadara/.hipfire/models/nemotron-3-nano-30b-a3b-mq4.hfq` loads the real
30B HFQ artifact and decodes two tokens: final argmax=1044. Remaining
validation: serve Nano-30B → coherent. Also `MEMEM*E…` has runs like `EM` and
`M*` — confirm the flat-block residual handles consecutive same-FFN/mixer blocks
(it does; each char is its own residual block).

---

## Cross-cutting notes

- **Block-type taxonomy** now spans M / * / - / **E** (after #6); keep
  `BlockKind` + `mixer_profile` the single source of truth.
- **Memory:** #3+#4 (mq4) is effectively a prerequisite for #6 (30B) and for any
  multi-model residency; the f32 path stays as the correctness reference floor.
- **Correctness floor:** every new kernel/block (chunked SSD #5, MoE #6) gets a
  CPU oracle + gpu-vs-cpu test *before* integration, as N1–N4 did — that
  discipline is why N5 served coherently first try.
