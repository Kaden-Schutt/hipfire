# Nemotron-H follow-ups + quantized-serving roadmap

Status as of 2026-06-25: **nemotron_h (Nano-4B) serves end-to-end** (arch_id 14,
loads and streams tokens), but coherent chat output remains blocked by the FU1
newline attractor — see `docs/plans/2026-06-24-nemotron-fu5-status.md` for the
current evidence and `docs/plans/2026-06-24-nemotron-h-mamba2.md` for N0–N5.
This doc plans the six follow-ups, each self-contained and grounded in the
current code/checkpoints. Effort tags: **S** ≈ hours, **M** ≈ 1–2 days, **L** ≈
several days.

## Recommended sequencing (dependencies)

```
1. chat-template refinement      (S)  ── done; FU1 coherence still needs a valid reference
2. HF-ref numeric bisect         (M)  ── lock correctness before scaling
3. quantizer compatibility       (M) ─┐ the "real-use memory" track
4. loader compatibility          (L) ─┘ (16GB f32 → ~4GB mq4); 4 depends on 3
5. N6 chunked-prefill + q8 state (L)  ── throughput (independent of 3/4)
6. Nano-30B (MoE 'E' block)      (L)  ── new model/feature (independent)
```
1, 2, 5, 6 are mutually independent; 4 depends on 3. The chat-template sub-work
in 1 is done; the remaining FU1 coherence work needs a valid CUDA/vLLM/NVIDIA
reference before more sampling or prompt-policy tuning is meaningful. Then pick
the track by priority: correctness (2), memory (3+4), speed (5), or scale (6).

---

## 1. Chat-template refinement (S)

**Current state (2026-06-25).** The concrete template items below are done in the
live code: arch 14 resolves EOS from tokenizer `<|im_end|>`, Jinja is the default
when the embedded template exists, and `hipfire chat` forwards thinking-off
controls (`closed_think`, `max_think_tokens = 1`) just like the HTTP path.

This did **not** resolve coherence. Current `target/debug/hipfire-daemon` still
emits newline-only output for Nano-4B, and the local HF pure-Torch fallback with
trained `dt_bias` restored also emits newline-only output for the same
byte-identical prompt. Treat the remaining FU1 task as "obtain a valid
CUDA/vLLM/NVIDIA-runtime reference and compare generation," not as another
chat-template tweak.

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
clean answers") is not met on ROCm today. New validation target: capture a
coherent external reference first, then compare Hipfire prompt IDs, first-token
logits, and generated text against that reference.

---

## 2. HF-reference numeric bisect (M)

**Goal.** Definitively confirm hipfire's nemotron forward matches HF
`NemotronHForCausalLM` per-layer (beyond the coherent-serve + gpu-vs-cpu evidence,
which only validate against *our own* CPU oracle / conventions).

**Current state.** `benchmarks/vision/dump_hf_reference.py` exists but is
vision-scoped. No text/nemotron HF-ref path.

**Approach.**
1. Python (tooling only, not hot path): load HF model with `trust_remote_code`,
   register forward hooks on each `NemotronHBlock` (input + output) and the final
   logits for a fixed token sequence; dump to `.safetensors`/`.npy`.
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
3. **Sensitivity:** the Mamba-2 `in_proj` drives dt/B/C — consider a higher-bit
   group or imatrix calibration (the `astrea` skill) for `in_proj`/`out_proj`
   before trusting mq4 there.

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

**Current state.** `SimpleAr::prefill` is a sequential `forward_gpu` per prompt
token (O(prompt_len) launch-bound), all f32; decode is single-token f32. The SSM
state is f32 `[num_heads·head_dim·state_size]` per Mamba block.

**Approach.**
1. **Chunked-SSD prefill** (the genuinely-new throughput kernel): implement the
   Mamba-2 SSD chunk-scan (chunk_size=256, already in config) — intra-chunk via
   matmuls, inter-chunk via state passing — so a prompt processes in
   `ceil(len/256)` chunked passes instead of `len` single-token steps. Also batch
   the conv1d over the whole prompt and run attention prefill via the existing
   batched/masked flash path instead of per-token.
2. **q8 SSM state:** quantize `h` to q8 between steps (mirror the GDN q8-state
   work noted in CLAUDE.md / `pflash.rs` q8 drafter-state). Cuts state
   memory+bandwidth; pairs with the chunked kernel.

**Reuse.** qwen35 `pflash` chunked prefill structure; GDN q8-state pattern;
`attention_*_batched_masked` for the attention blocks; `conv1d` batched variant.

**Risks.** The chunked inter-chunk recurrence **must be bit-faithful to the decode
recurrence** — the discriminator is gpu-vs-cpu of chunked-prefill logits vs the
validated per-token `forward` over the same tokens. (See the SSD CPU oracle
`ssd.rs` / `block.rs` — extend it with the chunked form as the test oracle.)

**Validation.** (a) chunked-prefill logits == per-token-decode logits
(gpu-vs-cpu, the existing oracle); (b) prefill tok/s benchmark per
`docs/methodology/perf-benchmarking.md` (warm cache, fresh-process probe).

---

## 6. Nano-30B — MoE ('E') block (L)

**Goal.** Serve NVIDIA-Nemotron-3-Nano-30B-A3B (present in `/srv/huggingface`).

**Current state (grounded from its config).** Same `nemotron_h` arch but
hidden=2688, 52 layers, pattern **`MEMEM*EMEM…`** — introduces a **new block
char `E` (MoE FFN)** replacing the dense `-` MLP. MoE: `n_routed_experts=128`,
`num_experts_per_tok=6` (top-6), `moe_intermediate_size=1856`, plus
`n_shared_experts` / `moe_shared_expert_intermediate_size` (shared expert).
**Note:** the N0 parser currently *rejects* `'E'` (it's the negative-test char in
`rejects_unknown_block_char`).

**Approach.**
1. `BlockKind::Moe` ('E'); update `parse_block_pattern`, `mixer_profile`
   (MLP/MoE are FFN-only, no mixer state), and the rejects-test.
2. Parse MoE config fields (routed/shared expert counts, top-k, moe_intermediate).
3. `NemotronMoeGpu` block: router GEMV → top-6 softmax/normalize → expert-indexed
   gate/up/down GEMVs (relu² or swiglu — **verify the MoE activation from the
   modeling source**) + shared expert; reuse the `moe_*` dispatch
   (scalar-indexed expert GEMV, top-k routing) from qwen35 (arch 6) / lfm2moe.
4. Loader: per-expert weight tensors + router + shared-expert names (read the
   30B safetensors header for exact `*.mixer.experts.*` / `*.router.*` naming).
5. Wire into `model.rs` `Block` enum + forward; add an `E` arm.

**Reuse.** qwen35/lfm2moe MoE kernels + routing (the big lever); the existing
M/*/- blocks unchanged.

**Risks.** MoE routing convention (softmax vs sigmoid gate, top-k
renormalization, shared-expert add) — verify from `modeling_nemotron_h.py` MoE
class before trusting. 30B is bigger (needs #3/#4 quantization for comfortable
residency; A3B = 3B active so decode is cheap, but 30B f32 weights are ~60 GB —
**mq4 effectively required**, so this pairs with #3/#4).

**Validation.** gpu-vs-cpu the MoE block (new CPU oracle, like the others); then
serve Nano-30B → coherent. Also `MEMEM*E…` has runs like `EM` and `M*` — confirm
the flat-block residual handles consecutive same-FFN/mixer blocks (it does; each
char is its own residual block).

---

## Cross-cutting notes

- **Block-type taxonomy** now spans M / * / - / **E** (after #6); keep
  `BlockKind` + `mixer_profile` the single source of truth.
- **Memory:** #3+#4 (mq4) is effectively a prerequisite for #6 (30B) and for any
  multi-model residency; the f32 path stays as the correctness reference floor.
- **Correctness floor:** every new kernel/block (chunked SSD #5, MoE #6) gets a
  CPU oracle + gpu-vs-cpu test *before* integration, as N1–N4 did — that
  discipline is why N5 served coherently first try.
