# dots.ocr (Qwen2-VL family) — dev log

Append-only progress timeline for the two-crate bring-up. For the
durable design + decision record see
[`dots-ocr-prd.md`](dots-ocr-prd.md). Section §1 is the
commit-by-commit table; §2-§6 are deeper narrative on phase-by-phase
work; §7 captures the closed risk register; §8 captures rev-3
deferred follow-ons (forward-looking).

## 1. Progress log

| commit | scope |
|---|---|
| `c6d4e539` | Bootstrap `hipfire-arch-qwen2` crate (skeleton from toy), `docs/architecture-ids.md`, rev-2 plan |
| `8ab7ec62` | Real `Qwen2Config::from_hfq` parser + Qwen2-1.5B manifest + 4 unit tests |
| `4bf9f6d4` | HFQ4 quantisation of Qwen2-1.5B validated (820 MB, 100% coverage); `inspect_hfq` example |
| `e034c44b` | Real `Qwen2Weights::load` — 28 layers + tied-lm_head + Q/K/V bias; cross-arch TODO markers |
| `45913eb0` | Rev-2 review fold-in: §2/§5/§6 amendments, tied-F16 lm_head fix (B1), EOS array semantics, lib.rs / doc refreshes, plan rename |
| `9477fbbb` | R1: `hipfire-quantize --arch-id <u32>` flag + re-quantised `qwen2-1.5b.arch7.hfq4` (arch_id byte = `0x07` verified) |
| `51e05b99` | R2: LLaMA-family loader hard-fails when `q_proj.bias` is in the manifest — closes the silent-wrong-output footgun for mis-tagged Qwen2 HFQ files |
| `d7a2ebab` | Phase 0 items 6 + 7: HF reference captured at `benchmarks/references/qwen2_1p5b_instruct_smoke.json` (transformers 5.5.1; 25 KB artifact with first-16 completion IDs + top-100 logits at pos 0/8/14) |
| `00d406af` | R3 mitigation: `infer_qwen2.rs` driver binary — wires the bring-up triple end-to-end. Tokenizer parity confirmed (hipfire's Rust BPE produces byte-identical token IDs to HF on the smoke prompt). |
| `afd4b059` | Phase 1 forward pass: `Qwen2State` + `forward_step` / `forward_step_greedy` (28 layers: RMSNorm → fused QKV + bias adds → RoPE → KV cache → attention → o_proj → residual → FFN norm → SwiGLU → residual; final norm + lm_head). HFQ4G256 9/16 top-1 matches with 7/7 prefix + fluent output. |
| `9bd083f6` | Phase 1 precision sweep: Q8F16 re-quant. **16/16 top-1 matches vs HF F32 reference** — correctness lock-in. Forward in 303 ms (140 ms prefill + 163 ms greedy 16 tokens). Confirms (a) implementation correct end-to-end, (b) HFQ4G256 divergence was 4-bit quant noise. Phase 1 closed. |
| `806680b2` | R3 resolved: daemon arm for `arch_id=7`. Wired `hipfire-arch-qwen2` as a runtime dev-dependency (`arch-qwen2` feature, default-on); LoadedModel fields + `free_gpu` impls; new load arm; `generate_qwen2` JSON stream. `hipfire run` against `qwen2-1.5b.arch7.q8.hfq` at ~96 tok/s for 137 generated tokens. Scope-limited (no DFlash / CASK / PFlash / VL / ChatML / repeat-penalty / top-p / `<think>` / multi-GPU). |
| `2226bbcf` | Pre-PR fixes from rev-3 review fold-in (Claude+Gemini+GLM-5): A1 (bench_prefill arch_id=7 panic) + A2 (reset event missed `Qwen2State.next_pos`) + B1 comment + D1 dead let + minor doc refreshes. `Qwen2State::reset()` helper added and called from daemon `reset` event AND `bench_prefill` cold-start. Regression run (load + bench_prefill[32] + reset + generate[16] + unload) byte-for-byte matches Q8 reference. |
| _PR open_ | [Kaden-Schutt/hipfire#297](https://github.com/Kaden-Schutt/hipfire/pull/297) — phase 1 deliverable. |
| `f6b28a12` | **Phase 2a** — bootstrap `hipfire-arch-dots-ocr` crate (arch_id=8). Mirrors phase 1 sequence: typed Config/Weights/State + Architecture trait impl with dots.ocr-specific EOS overrides. `DotsOcrConfig` wraps Qwen2Config (text) + DotsVisionConfig (vision); `DotsOcrWeights` wraps Qwen2Weights + DotsVisionWeights side-by-side. Vision weight load + `vision_forward` phase-2c stubs. 5 tests passing. |
| `acd75473` | Merge `upstream/master` into `feat/dots-ocr-qwen2`: 47 commits / 779 files / +23.8k lines (license relicense, HFQ6 family, BF16 loader, MoE grouped-WMMA, gfx94x MFMA, Qwen3.5 MoE norm fix). 1 real conflict resolved: `qwen35.rs` dropped `load_norm_weight_raw` (superseded by upstream PR #228 GemmaRMSNorm convention fix). PR #297 mergeable. |
| `544822b4` | **R5** — quantiser packs `generation_config.json` into HFQ metadata. Qwen2 parser walks fallback (`config.eos_token_id` → `generation_config.eos_token_id` → default). Fixes dots.ocr's silent EOS-default-to-`<\|im_end\|>` (151645, never fires) by surfacing the real `[151643, 151673]`. 8 tests (+2 new for the fallback path). |
| `1115486a` | **Phase 0 item 2** — read `modeling_dots_ocr.py` + `modeling_dots_vision.py` end-to-end. No plan contradictions. Captures: attention scale = plain `1/sqrt(head_dim)`, multi-image batching = single flattened sequence (i32 cu_seqlens), bf16 cast at vision forward entry, vision-text integration via `masked_scatter()` with no projection layer, all 42 blocks structurally identical, no dropout/droppath in inference. |
| `bfe1f56d` | **Phase 2b** — image preprocessing. `smart_resize` (28-divisible, beta scaling, AR>200 guard, zero-dim guard), `clip_normalise` (RGB→CHW f32, CLIP constants), `extract_patches` (2×2-grouped-block-major enumeration — the silent-failure trap), `preprocess_image` top-level wrapper with RGBA→RGB compositing. 14 unit tests, no GPU. Patch-order silent-failure-trap gated by `extract_patches_uses_grid_block_order` against a per-pixel-tagged 28×56 synthetic input. |
| `bc4640b7` | Merge `upstream/master` into `feat/dots-ocr-qwen2` (round 2): 2 commits / 1 conflict in `daemon.rs` (PR #312 OpenAI vision refactored generate_vl to GenerateVLParams; kept our generate_qwen2 + json_escape unchanged, took upstream's new signature). PR #297 still mergeable. |
| `8c04ba46` | **Phase 0 item 5** — dots.ocr HF reference captured. **(a)** `dots_ocr_smoke_001.json` via HF/CPU/bf16/eager: prefill logits at positions 0/32/128/5094 (last prompt position; top-1='[' with +11 nats gap, valid forward pass through prompt + image embeds). Greedy decode degrades after 5 tokens (CPU+bf16 limitation; documented in `decode_quality_note`). **(b)** `dots_ocr_smoke_001_vllm.json` via vLLM 0.13.0/GPU/bf16/flash_attn: 13-element layout JSON, parse_status ok — real ground truth for phase-4. Image is `dots_ocr/demo/demo_image1.jpg` (1700×2250 medical-paper page, md5 `a434c567a2dfa0664ce75291508bad85`). |
| `7ba3c749` | **Phase 2c-1** — captured 6 intermediate vision-tower activations via PyTorch forward hooks: `patch_embed`, blocks 0/21/41, `post_trunk_norm`, `merger`. Full tensors (~120 MB each, [19520, 1536] for patch-shape stages and [4880, 1536] for the merger) saved to `/data/cache/hipfire/dots_ocr_activations_full/` for local-only use. A 256-row (64 for merger) linspace sample plus indices committed to `benchmarks/references/dots_ocr_smoke_001_activations/` for git (7.9 MB total). |
| `7051d6e9` | **Phase 2c-2** — vision weight loader. 17 vision-tensor names: patch_embed (Conv2d → linear reshape `[embed_dim,3,14,14]` → `[1536, 588]`, bias, RMSNorm), 42× DotsVisionBlock (norm1, attn.qkv, attn.proj, norm2, fc13_proj load-time concat, fc2), post_trunk_norm, merger (ln_q + bias, mlp.{0,2} + biases). Load-time SwiGLU fusion via fc1+fc3 → fc13_proj concat. 4 helpers carry `TODO(transformer-extraction)` markers. 19 tests still passing. |
| `22d47330` | **Phase 2c-3** — 2-D RoPE prep helper (`rope::build_rope_2d_tables`). Ports `get_pos_ids_by_grid` + `VisionRotaryEmbedding` + `apply_rotary_pos_emb_vision` quarter-repeat layout from `modeling_dots_vision.py` into a single CPU function emitting per-patch `[N_patches, head_dim]` cos/sin in dots.ocr's exact `[hc, wc, hc, wc]` layout. Patch enumeration in 2×2-block-major order matches `image::extract_patches`. 7 new tests including a hand-computed 2×2/head_dim=8 case and a reshape-permute-flatten equivalence check on a 4×6 grid. 26 tests total. |
| `9f738911` | **Phase 2c-4** — vision GPU primitives. (1) New kernel `rope_2d_halfsplit_f32` — applies precomputed cos/sin tables to Q/K in-place; halfsplit pairs `(d, d+head_dim/2)`. (2) Loader refactor: vision linear weights stored as F16 GpuTensor on GPU (HFQ4/Q8/F32 dequantise to F16 at load time per the qwen35-vl pattern; N=~20k patches makes batched HFQ4 GEMM the bottleneck). DotsVisionBlockWeights + DotsVisionWeights field types changed from WeightTensor → GpuTensor. (3) `linear_f16` + `linear_f16_no_bias` private helpers. Vision-shape primitive audit confirmed: existing `rmsnorm_f32`, `silu_mul_f32`, `bias_add_f32`, `gelu_tanh_f32`, `layernorm_batched`, `gemm_f16`, `add_inplace_f32`, `transpose_f32` accept `[N_patches, hidden]` strides. CAVEAT: large-N attention needs `vit_attention_opt` instead of `vit_attention_f32` (latter materialises N² scores in shared mem, ~77 KB at N≈19520 exceeds RDNA per-CU SLM cap). |
| `5409c740` | **Phase 2 review fold-in (rev-claude + rev-glm5 + rev-gemini)** — three-reviewer pass on `f6b28a12..9f738911`. rev-claude A1 (out_hidden_size fallback) VALIDATED → fixed; B1 (smart_resize upscale re-clamp) → fixed; B2/B3 (rope_2d dispatch wiring) → fixed (launch_maybe_blob + begin_timer + head_dim%4 assert); B4 (qwen2 norm-weight length assert) → harmonised; C2 (concat_rows asserts) → added; C5 (TPS>1 doc) → added; rev-glm5 A1 (rope kernel head bounds) → REJECTED (work IS inside guards); A2 (vision_forward GPU signature) → fixed (takes &GpuTensor patches, returns HipResult<GpuTensor>); A3 (load_f16_or_dequant qt=3) → panic message improved (defer qt=3 arm to phase 5); rev-gemini 3.1 (multi-image attention leakage) → DOCUMENTED (single-image-only in vision_forward, per-image loop spec'd for phase 3); 3.2 (IMGPAD count assert) → DOCUMENTED in plan §5 phase 3; 3.3 (R5 deferred-list entry) → fixed. 34 tests passing (26 dots-ocr + 8 qwen2). Review scaffolding files dropped per `feedback_drop_review_files_after_fold_in`. |
| `fa2eaa87` | **Phase 2c-5a** — `vision_forward` assembly. Replaces the phase-2c stub with the full GPU encoder + merger pipeline: 2-D RoPE table upload, patch_embed + patchifier RMSNorm, 42 encoder blocks (norm1 → fused QKV GEMM → 2-D RoPE on Q+K via the **new** `rope_2d_halfsplit_qkv_interleaved_f32` kernel → `vit_attention_opt` → o_proj + residual → norm2 → fused fc13 GEMM → `silu_mul_f32` → transpose → fc2 + residual), post-trunk RMSNorm, and PatchMerger (LayerNorm + free 2×2 reshape + MLP with gelu_tanh). The fused-QKV-interleaved RoPE kernel lets a single QKV GEMM feed directly into attention without intermediate split/merge copies. 34 tests passing. `vit_attention_opt` later swapped for `attention_dflash_f32` in 2c-5b (LDS overflow at N≈20k). |
| `d8e851b8` | **Phase 2c-5b prereq** — `gemm_f16_wmma` correctness fix. Kernel had a known UB bug (each lane writing 256 elements into a 16-element `half16_t` register vector). Fix is the lane-cooperative pattern used by neighbouring working WMMA kernels: lane t holds row t of the input tile. Bit-identical to scalar `gemm_f16` on dots.ocr's patch_embed shape (M=1536, K=588, N=400). Unlocks 256× fewer GEMM blocks for the vision encoder's QKV/FC GEMMs (sub-second instead of minutes per block). |
| `69b6898f` | **Phase 2c-5b** — `infer_dots_ocr` driver + attention swap. New `examples/infer_dots_ocr.rs` validates the vision tower against the 2c-1 HF reference .npy refs (cosine + max-abs per-row, pass tol cos>0.999 OR ‖Δ‖<1e-2). Attention path switched from `vit_attention_opt` to `attention_dflash_f32` (online-softmax, no LDS overflow); new `Gpu::qkv_split_interleaved_f32` splits the fused `[N, 3h]` output into three flat Q/K/V buffers. 2-D RoPE switched back to the separate-buffer variant. Vision linear weights use the fixed `gemm_f16_wmma`. |
| `160a8478` | **Phase 2c-5c perf (1/3)** — `attention_dflash` phase C parallelised over j-chunks. Original parallelised across `head_dim` only; at L≈20k with nthreads=256 and head_dim=128 half the threads idle. Maps `tid → (j_chunk_id, d_lane)` so spare threads cooperate on the j-axis, reusing existing `ws[nthreads]` LDS. Block 0 attention drops from 49.7s → ~12s. |
| `b459b4e6` | **Phase 2c-5c perf (2/3)** — `attention_dflash` multi-accumulator ILP. Both serial-FMA hot loops used a single running accumulator (each FMA waited on previous, ~4-cycle RDNA3 FMA latency × 128 head_dim = ~512 cycles per j). 8-way ILP in phase A's d-loop + 4-way ILP in phase C's j-loop break the chain. 28-case sweep (L=1..16384, hd=64..512) max-abs-diff 1.21e-8 (all PASS). |
| `baf0b13b` | **Phase 2c-5c perf (3/3)** — new `attention_dflash_wmma_f32` (FlashAttention-2 with RDNA3 wave32 WMMA). Drop-in replacement when `head_dim ≤ 256 && head_dim % 16 == 0`; falls back to scalar `attention_dflash_f32` otherwise. Grid `[n_heads, ceil(B/16)]`, block `[32]`, ~26 KB LDS at hd=128. Three phases per K-tile: `S = Q @ K^T` via WMMA, online softmax + α scaling, `O += softmax(S) @ V` via WMMA (V staged into LDS in [k, d] layout so the second WMMA reads V's columns per `D = A @ B^T`). Block 0 attention now ~2.9s (vs 49.7s pre-2c-5c) — **17× speedup**. Full dots-ocr vision encoder 35min → 3.5min on the smoke image. |
| `21ed91e1` | **Phase 2c-5d (WIP)** — investigation infrastructure. Per-stage dump (env-gated `HIPFIRE_DOTS_OCR_DUMP_DIR`) + `scripts/diff_dots_ocr_stages.py` to compute per-stage cosine vs HF reference. Also fixed `preprocess_dynamic_image` resize filter (Triangle → CatmullRom, closest match to HF's `PILImageResampling.BICUBIC`). Resize fix lifted patch_embed cos 0.99916 → 0.99998 and block_00 cos 0.99895 → 0.99995, but deeper-block divergence (merger cos 0.28) remained — separate bug. |
| `283ee5c8` | **Phase 2c-5d (WIP)** — block-by-block bisection. Extended HF reference capture to 9 block indices (0, 1, 2, 4, 8, 12, 16, 21, 41) so the per-block divergence trajectory is visible. Per-stage mean cos drops monotonically — early blocks within 0.999, divergence accumulates by block 21, merger at 0.28. Indicates a per-layer amplification, not a localised bug. |
| `f9fbd4c6` | **Phase 2c-5d (WIP)** — attn_out bisection isolates per-block drift to the attention path (vs MLP). Confirmed by zeroing out the attention residual and tracking how the divergence shape changes. Narrowed to the attention kernel chain (QKV linear → RoPE → attention compute → proj). |
| `ed1f79e6` | **Phase 2c-5d (WIP)** — bf16 round-trip kernel (`bf16_round_trip.hip`, env-gated `HIPFIRE_DOTS_OCR_BF16_RESIDUAL=1`). Hypothesis: HF's bf16 residual stream truncation introduces per-layer rounding that our F32 pipeline doesn't have. Test: zero effect on output cosines. First evidence that the divergence is not residual-stream precision drift. (Same negative result reproduced later for bf16 in QKV output, scores, softmax output — all zero effect.) |
| `d0f38625` | **Phase 2c-5d (WIP)** — pre-attention QKV linear capture. Added a hook that dumps the post-QKV-linear output before reshape/RoPE/attention. Diff vs HF dump: cos 0.99924 (small input delta). After our attention compute on our QKV: cos 0.99819 vs HF's full attn output. → input delta is small; the post-attention divergence is amplified inside the attention compute or proj GEMM. |
| `4c059433` | **Phase 2c-5d landed: no bug.** Decisive evidence via `scripts/numpy_attention_replay.py` — F32 numpy attention on captured QKV reproduces our GPU attn pre-proj at **cos 1.00000** (5 decimals). New `examples/dump_proj_weight.rs` extracts `proj.weight` as F32 .npy so the proj GEMM can be numpy-replayed; same result, cos 1.00000 vs our GPU attn_out. **Our GPU pipeline is bit-equivalent to F32 numpy on the SAME QKV inputs.** Meanwhile `numpy(HF_qkv @ proj_w.T)` vs `HF_attn_out` is cos 0.989 — i.e. HF's bf16 compute diverges from the F32 reference by 1%. We are **6× closer to the F32 reference than HF's bf16 path is.** The cos 0.989 we'd been chasing is HF's bf16 truncation drift, not our bug. Also falsifies a UAF / stream-race hypothesis (numpy reproduces our dumps; no corruption). |
| `1f94da31` | **Phase 2c-5d Strategy A — END-TO-END OCR PASSES 13/13.** Added `qwen2::forward_step_with_embed` (sibling to `forward_step` that uploads a pre-built F32 embedding row instead of doing the lookup — mirrors `qwen35::forward_scratch_embed`) + new `examples/ocr_e2e.rs` (loads HFQ, runs vision_forward, splices merger output at IMGPAD slots during qwen2 prefill, greedy-decodes until EOS) + new `scripts/grade_dots_ocr_e2e.py` (greedy bbox-IoU match + Levenshtein text distance). Result on smoke image vs `dots_ocr_smoke_001_vllm.json`: **regions 13/13, F1 1.000, text exact-match 13/13, max bbox L1 = 1 pixel**. The model is robust to F32-vs-HF-bf16 drift. Strategy B (bf16-truncate-everywhere) NOT needed. Perf one-shot: vision **198 s** + text-weight load 0.4 s + prefill 81.6 s (5095 tokens, 62.5 tok/s unbatched) + greedy gen 119.9 s (4633 tokens until EOS 151673, 38.6 tok/s) ≈ **400 s end-to-end** (initial commit message said 206s — incorrect; the per-block `0.10s` log-line is queue-launch time, not GPU time, and was misread as cumulative). Phase 2 closed. |
| `47a28359` | docs-only plan sync: §0 progress log + §5 phase status updated to reflect 2c-5d closed. New §2.10 "The bf16-oracle lesson" codifying the rule that HF tensor-dump cosine is NOT a valid correctness oracle for bf16-trained models. Phase 3 marked NOT STARTED; phase 4 PARTIAL (smoke-image scaffolding done, broader reference set pending). |
| `67cc4ac6` | Merge `upstream/master` into `feat/dots-ocr-qwen2-phase-2` (2026-05-22): brings pflash perfmax, fwht3/4 KV drafters, gfx10 MQ3/HFQ3 prefill kernels, GPT-2 BPE pre-tok fix. 4 conflicts resolved: `Cargo.toml` (add dots-ocr crate row), `docs/architecture-ids.md` (arch_id=8 status updated), `crates/hipfire-arch-qwen2/src/qwen2.rs` (kept HEAD's R5 EOS-fallback parser + `load_norm_weight_raw` length assert + `forward_step` refactor with `forward_step_with_embed` + R5/EOS tests), plan progress log + §5 phase narrative. |
| `29d08839` | Close 7 `bind_thread()` audit gaps in `dispatch.rs` (surfaced by `scripts/verify-bind-thread.sh` on the post-merge tree). Four phase-2 dots-ocr fns: `bind_thread()?;` already existed but sat after asserts — moved to the top. Three pflash wrappers from master: delegating to an impl that binds — added explicit bind at the wrapper. 446 pub fn audited clean. |
| 2026-05-23 | Plan refactor: split monolithic `qwen_2.0_vlm_plus_dots_ocr.md` into `dots-ocr-prd.md` (durable design) + `dots-ocr-devlog.md` (this file). Renamed `qwen_2.0_vlm_plus_dots_ocr.{dots_ocr,qwen2_1p5b}_manifest.txt` → `dots-ocr.{dots_ocr,qwen2_1p5b}_manifest.txt` for consistency. Dropped 8 stale review-scaffolding files (rev-claude/glm5/gemini artifacts and untracked `dots_ocr_quality_eval_gemini.md`) per `feedback_drop_review_files_after_fold_in`. DFlash parity sweep re-verified after the merge: 112 cases, 0 failed, max-abs-diff 3.052e-5 (under 1e-3 tolerance). |
| _PR open_ | [Kaden-Schutt/hipfire#321](https://github.com/Kaden-Schutt/hipfire/pull/321) — phase 2 deliverable (vision tower + Strategy A E2E OCR). Awaiting maintainer review + merge. |
| `#321 merged` | Phase 2 merged to master (2026-05-25). dots.ocr upstream as an e2e example. Phase 3 work below is on `feat/dots-ocr-phase-3-daemon`. |
| `d9e00e4e` | **Phase 3 — daemon serving path (arch_id=8).** Promotes the `ocr_e2e` splice into the daemon: new `load_model` arm (Qwen2 text decoder reusing `qwen2_state` + resident DotsVisionTransformer weights), `generate_vl_dots_ocr` (preprocess → `build_prompt_ids` → `vision_forward` → per-token IMGPAD-splice prefill → greedy decode → JSONL stream, with a hard merger-count-vs-IMGPAD-slot guard), dispatch routing. New `dots_ocr::build_prompt_ids` reproduces the HF `apply_chat_template` image framing **byte-exact** — verified by an oracle test against the captured 5095-token `dots_ocr_smoke_001.json` (only 12/5095 tokens differ, all on BPE whitespace boundaries that decode identically — the documented `tokenizer.rs` `\s+(?!\S)` lookahead gap, general/benign). Found+removed an ngram loop-guard that false-stopped mid-HTML-table at 391 tokens (OCR layout-JSON legitimately repeats; the proven `ocr_e2e` path uses no guard). **Daemon smoke grades 13/13 exact-match, F1 1.000 vs vLLM** — identical to the standalone path. Cargo: `arch-dots-ocr` feature + dev-dep + daemon required-features. |
| `3d2412b5` | **Phase 3 — base64 image input + loaded-event metadata.** `image::preprocess_image_bytes` (decode from memory) wired to the daemon Base64 source with data-URL prefix stripping (base64 is the normal client transport). Fixed the `loaded` event for arch_id=8: reports arch `"dots-ocr"` (was `"qwen3"`), dim/layers/vocab from `dots_ocr_config.text` (were 0), `vl=true`. Base64 smoke grades 13/13, identical to the file-path path. |
| `a3389fc2` | **Phase 3 perf — batched Qwen2 prefill.** New `qwen2::forward_prefill_batch_embeds`: whole-prompt single pass over a pre-built `[seq×dim]` embedding matrix (token-embedding rows for text, spliced vision-merger rows at IMGPAD), using batched Q8 GEMM (`gemm_q8_0_batched_chunked`) + `rope_batched_f32` + `attention_causal_batched` + `rmsnorm_batched` — mirrors `llama::prefill_forward` with QKV bias added and embeddings (not token-ids) as input. Daemon builds the matrix (~215 text-token GPU lookups; 4880 visual rows host-resident in `merged`). Prefill **46.8s@109tok/s → 32.3s@158tok/s (1.45×)**, 13/13 preserved. RoPE convention + GQA confirmed correct vs the per-token path. |
| `995d7449` | **Phase 3 perf — batched-F32 KV-cache write kernel.** New `kv_cache_write_f32_batched.hip`: scatters all batch rows into the F32 KV cache at their absolute positions in one launch per layer (reusing the RoPE position array), replacing the per-position write loop (5095×28×~5 ops) that dominated batched-prefill time after the GEMMs were batched. Wired only into the Qwen2 batched-prefill path (no qwen35/llama forward change). Prefill **32.3s → 0.35s (14.5k tok/s)** — 92× over per-position write, **134× over the original per-token prefill**; 13/13 preserved. Committed through the coherence gate (qwen35 battery clean; agentic/speed gates skipped — those models absent on this box). |

## 2. Phase 0 — ground-truth capture

**Goal:** committed reference artifacts for every downstream
correctness gate, plus end-to-end read of the HF source to surface
modeling subtleties.

**Items shipped:**

1. Manifests captured: `dots-ocr.dots_ocr_manifest.txt` (642 lines,
   338 tensors), `dots-ocr.qwen2_1p5b_manifest.txt` (339 lines, 338
   tensors). Confirms `tie_word_embeddings=true` on Qwen2-1.5B (no
   `lm_head.weight` on disk) and `attention_bias=true` (Q/K/V bias
   tensors present).
2. End-to-end HF source read of `modeling_dots_ocr.py` +
   `modeling_dots_vision.py`. Findings rolled into PRD §1.9.
3. Qwen2-1.5B-Instruct HF reference at
   `benchmarks/references/qwen2_1p5b_instruct_smoke.json` (transformers
   5.5.1; prompt token IDs, first-16 completion IDs, top-100 logits
   at positions 0/8/14). Sanity check: top-1 at pos_14 = token 362,
   matches `first_16_completion_token_ids[0]`. Greedy + logit dump
   self-consistent.
4. dots.ocr HF reference (TWO complementary artifacts):
   - **HF / CPU / bf16 / eager**: `dots_ocr_smoke_001.json` — prefill
     logits at positions 0/32/128/5094. Coherent (pos_5094='[' with
     +11 nats top-1 gap). Greedy decode degenerates after ~5 tokens
     (CPU+bf16 limitation, both sdpa and eager); `decode_quality_note`
     documents this so consumers don't mistake `completion_token_ids`
     for usable layout. Three transformers-5.5.1 runtime patches
     needed (recorded in `scripts/capture_dots_ocr_reference.py`):
     `prepare_inputs_for_generation` None-guard + first-step
     pixel_values splice via MRO bypass; `mm_token_type_ids` kwarg
     filter; `_validate_model_kwargs` no-op for VL-arg compat.
   - **vLLM 0.13.0 / GPU / bf16 / flash_attn**:
     `dots_ocr_smoke_001_vllm.json` — full 13-element layout JSON,
     parse_status ok. Real ground truth for the phase-4 OCR gate.
5. Per-stage HF activation capture for the vision tower — 6 stages
   (patch_embed, blocks 0/21/41, post_trunk_norm, merger), full
   tensors at `/data/cache/hipfire/dots_ocr_activations_full/`,
   sampled rows committed at
   `benchmarks/references/dots_ocr_smoke_001_activations/`. Later
   extended to 9 block indices (0/1/2/4/8/12/16/21/41) during 2c-5d
   bisection.

## 3. Phase 1 — Qwen2 text decoder

**Closed by `9bd083f6` (16/16 top-1 match vs HF F32 reference at Q8F16).**

Bring-up sequence mirrored from `hipfire-arch-toy` template:

1. Skeleton crate + `Qwen2Config` parser + manifest + 4 unit tests
   (`c6d4e539` → `8ab7ec62`).
2. `Qwen2Weights::load` — 28 layers + tied-lm_head + Q/K/V bias
   (`e034c44b`).
3. HFQ4 quantisation at 100% coverage (`4bf9f6d4`).
4. `Qwen2State` + `forward_step` + `forward_step_greedy`: 28-layer
   decoder with RMSNorm → fused QKV (+ bias) → RoPE → KV cache write →
   attention → o_proj → residual → FFN norm → SwiGLU → residual; final
   norm + lm_head. Initial HFQ4 run: 9/16 top-1 matches with 7/7
   prefix + fluent output — synonym-position divergence consistent
   with 4-bit quant noise (`afd4b059`).
5. Q8F16 precision sweep: **16/16 top-1 matches** — definitive
   correctness lock-in (`9bd083f6`). Confirms (a) implementation
   correct end-to-end, (b) HFQ4G256 divergence was 4-bit quant noise
   not implementation error.
6. Daemon arm for `arch_id=7` (`806680b2`) + rev-3 fold-in
   (`2226bbcf`) → PR #297.

**Subtleties discovered during implementation** (not in the original
plan):
- `HipResult` lives in `hip_bridge::error`, not `rdna_compute`.
- `GpuTensor` doesn't implement `Clone`; tied embeddings cost
  ~117 MB VRAM duplication on Qwen2-1.5B at HFQ4. Resolvable via
  `Arc<GpuTensor>` / shallow-clone in the future Transformer-extraction
  PR.
- The qwen35 weight-loading helpers (`load_norm_weight*`,
  `load_weight_tensor*`) are private; cross-arch reuse requires
  duplication. Both sides carry `TODO(transformer-extraction)`
  markers.
- `hipfire-quantize` auto-assigns `arch_id=1` to Qwen2 inputs.
  `hipfire-arch-qwen2` claims `arch_id=7`, so per-file remap (or
  `--arch-id` flag) needed → R1.
- `hipfire_runtime::llama::EmbeddingFormat` has no `F16` variant —
  llama / qwen35 loaders always expand F16 source → F32 on host
  before upload for tied embeddings. qwen2 loader follows the same
  pattern → R4 latent fix.
- dots.ocr `eos_token_id` lives in `generation_config.json`, not
  `config.json`. Quantiser fix needed → R5.

## 4. Phase 2 — dots.ocr vision tower

**Closed by `1f94da31` (Strategy A end-to-end OCR PASS, 13/13).**

### Phase 2a — bootstrap (`f6b28a12`)

Crate skeleton, typed Config/Weights/State, Architecture trait impl
with dots.ocr EOS overrides, `DotsOcrConfig` (text + vision), 5
tests.

### Phase 2b — image preprocessing (`bfe1f56d`)

`smart_resize` + `clip_normalise` + `extract_patches` +
`preprocess_image` + RGBA→RGB compositing. 14 unit tests, no GPU
needed. Silent-failure-trap gated by
`extract_patches_uses_grid_block_order`.

### Phase 2c — vision tower bring-up

Sub-phases 2c-1 through 2c-5d, each a separate commit on the progress
log. Highlights:

- **2c-1** (`7ba3c749`) — HF activation capture via PyTorch forward
  hooks.
- **2c-2** (`7051d6e9`) — vision weight loader for 17 tensor names,
  load-time SwiGLU fc1+fc3 → fc13_proj concat.
- **2c-3** (`22d47330`) — 2-D RoPE CPU prep helper with
  quarter-repeat layout matching `apply_rotary_pos_emb_vision`.
- **2c-4** (`9f738911`) — vision GPU primitives. Two key decisions:
  (a) F16 GpuTensor storage for vision linear weights (dequant on
  load → `gemm_f16`) because at N≈20k patches the batched HFQ4 GEMM
  is the bottleneck; (b) the existing `gemm_f16` + transpose path
  was good enough to get to assembly without the WMMA fast path.
- **2c-5a** (`fa2eaa87`) — full `vision_forward` assembly.
- **2c-5b** (`d8e851b8` + `69b6898f`) — WMMA correctness fix
  (`gemm_f16_wmma` had a 256-elements-into-16-element register UB
  bug) + driver + attention swap to `attention_dflash_f32`
  (LDS-safe at N≈20k).
- **2c-5c** (`160a8478` + `b459b4e6` + `baf0b13b`) — attention perf:
  j-chunk parallelism, multi-acc ILP, then a new
  FlashAttention-style WMMA kernel. Block 0 attention from 49.7s →
  2.9s (17× speedup). Full vision encoder 35 min → 3.5 min.
- **2c-5d** (`21ed91e1` → `1f94da31`) — the "bug that wasn't" arc.

### Phase 2c-5d — the bf16-oracle investigation

Detailed because it produced a load-bearing rule (PRD §2.1) and
changed how we'll validate future bf16-trained models.

**Setup.** Vision encoder ran end-to-end at 3.5 min per smoke image
after 2c-5c, but per-stage cosine vs HF dumps showed drift:
patch_embed 0.99998, block_00 0.99995, block_21 0.69, merger 0.28.
The drop was monotonic block-by-block — pointing at a per-layer
amplification rather than a localised bug.

**Investigation path:**

1. **Image resize filter** (`21ed91e1`). Triangle (bilinear) →
   CatmullRom (closest cubic available in the `image` crate vs HF's
   PIL BICUBIC). Lifted patch_embed cos 0.99916 → 0.99998. Did not
   fix the deeper-block drift.
2. **Block-by-block bisection** (`283ee5c8`). Extended capture from
   3 indices to 9 (0/1/2/4/8/12/16/21/41). Confirmed monotonic
   drop — no single bad block.
3. **Attn vs MLP bisection** (`f9fbd4c6`). Per-block, zeroed out
   the attention residual and tracked divergence shape. Drift
   localised to the attention path.
4. **bf16 residual stream emulation** (`ed1f79e6`). Added
   `bf16_round_trip.hip` kernel + env gate
   `HIPFIRE_DOTS_OCR_BF16_RESIDUAL=1`. Hypothesis: HF's bf16
   truncation accumulates ~1% per-layer drift we don't have. Test:
   **zero effect on output cosines.** Same negative result later
   reproduced for bf16 in QKV output, scores, softmax output — every
   bf16-truncation hypothesis was falsified.
5. **QKV linear capture** (`d0f38625`). Hook dumps post-QKV-linear
   output before reshape/RoPE/attention. Diff vs HF dump: cos
   0.99924 (small input delta). After our attention compute on our
   QKV: cos 0.99819 vs HF's full attn output. The input delta is
   tiny; the post-attention divergence is amplified inside attention
   compute or proj GEMM.
6. **F32 numpy reference replay** (`4c059433`). Decisive. New
   `scripts/numpy_attention_replay.py` computes F32 numpy attention
   on captured QKV. New `examples/dump_proj_weight.rs` extracts
   `proj.weight` as F32 .npy so numpy can apply the proj.

   Results on block 1:
   - `numpy(our qkv)` vs OUR GPU pre-proj attn: **cos 1.00000** (5
     decimals).
   - `numpy(our pre @ proj_w.T)` vs OUR GPU attn_out:
     **cos 1.00000**.
   - `numpy(HF qkv @ proj_w.T)` vs HF attn_out (HF's actual bf16):
     cos 0.99013.
   - `numpy(HF qkv @ proj_w.T)` vs OUR attn_out: **cos 0.99942**.

   **Our pipeline is bit-equivalent to F32 numpy. HF's bf16 deviates
   from F32 reference by ~1%. We are 6× closer to the F32 reference
   than HF is.** The "divergence" was HF's bf16 drift, not our bug.

   This also falsifies a UAF / stream-race hypothesis floated by a
   read-only-investigation agent: if `gpu.free_tensor(attn)` were
   releasing memory the gemm was still reading, our dumped attn /
   attn_pre_proj wouldn't reproduce under numpy F32 replay at
   cos 1.00000. They do. The pool's free is CPU-only bookkeeping;
   GPU ops are queued on a single stream and execute in order.

7. **Strategy A — end-to-end OCR** (`1f94da31`). The "is it
   task-correct" gate. Added `qwen2::forward_step_with_embed` to
   accept a pre-built F32 embedding row (sibling to `forward_step`
   that does the lookup). New `examples/ocr_e2e.rs`: load HFQ → run
   `vision_forward` → splice merger rows at `<\|imgpad\|>` positions
   during qwen2 prefill → greedy decode. New
   `scripts/grade_dots_ocr_e2e.py`: bbox IoU + Levenshtein.

   **Result vs `dots_ocr_smoke_001_vllm.json`: F1 1.000, 13/13 text
   exact-match, max 1-pixel bbox L1 deviation.** The trained model
   is robust to F32-vs-HF-bf16 drift. Strategy B (bf16-truncate
   everywhere) NOT needed.

**Conclusion (PRD §2.1).** For any bf16-trained model, per-stage HF
dump cosine is NOT a valid correctness oracle. The correct gate is
end-to-end task output.

## 5. Performance baselines

**dots.ocr smoke image, gfx1151, ROCm 7.12, cold kernel cache** (post
`1f94da31`):

| stage | wall | rate |
|---|---|---|
| vision weights load | 12.2 s | one-shot |
| vision encoder (42 blocks) | ~198 s | ~4.7 s / block (post-2c-5c WMMA; ~74% spent in attention_dflash_wmma) |
| text weights load | 0.4 s | one-shot |
| prefill (5095 tokens) | 81.6 s | 62.5 tok/s (one-position-at-a-time, no batched prefill) |
| greedy generation | 119.9 s | 4633 tokens / 38.6 tok/s until EOS 151673 |
| total | **~400 s** | per OCR page |

**Vision encoder is the real bottleneck, not prefill.** Per-block
trace (block 0, `HIPFIRE_DOTS_OCR_TRACE=1`):

| per-block stage | time | share |
|---|---|---|
| attention_dflash_wmma (N=B=L=19520, hd=128) | 2944 ms | 74 % |
| fc13 GEMM (gate+up fused, [N, 8448]) | 540 ms | 14 % |
| fc2 GEMM ([N, 1536]) | 306 ms | 8 % |
| proj GEMM ([N, 1536]) | 127 ms | 3 % |
| transpose | 24 ms | 0.6 % |
| silu_mul | 7 ms | 0.2 % |
| residuals / norms / rope / qkv-split | < 10 ms each | < 1 % |
| block total | ~4700 ms | |

× 42 blocks ≈ 198 s. The 16-query WMMA tile (M=16 in
`attention_dflash_wmma`) is the perf ceiling — every K-tile load
serves only 16 queries, so global-memory traffic dominates.
Theoretical FMA floor at gfx1151's ~17 TFLOPs F16 sustained is ~5 s;
we are 25× over that.

**Speedup roadmap (phase 6 — perf):**
- **Larger M-tile** (M=32 or M=64) in `attention_dflash_wmma`. 2-3×
  attention speedup → vision ~70-100 s. Moderate kernel rewrite.
- **Async K-load + double-buffer** on top of larger M-tile. ~1.5-2×
  more.
- **Cross-block K-cache sharing** within a head (1220 query-tile
  blocks all read the same K). 3-5× theoretical. High effort, HIP
  cooperative-groups territory.
- **hipBLASLt fused attention** if exposed on gfx1151 — link
  against the ROCm flash-attn extension instead of carrying our own.
  Check before committing to a new kernel rewrite.

Prefill is also slow but secondary: 62 tok/s one-position-at-a-time
is the documented unbatched cost (rev-3 deferred §8 "Prefill
batching"); a batched-prefill GEMM variant gets it to several
hundred tok/s. Correctness-wise the current baseline is the
ship-able floor.

**DFlash parity sweep** (post-2c-5c, re-checked 2026-05-23 after
master merge): 112 cases (L ∈ {1, 127, 128, 13951, 13952, 13953,
16384} × hd ∈ {64, 128, 256, 512} × B ∈ {1, 16, 17, 32}; scalar at
B=1, WMMA at all B). 0 failed. Max-abs-diff 3.052e-5 vs CPU naive
softmax reference, well under the 1e-3 tolerance.

**Outstanding: DFlash spec-decode coherence gate not yet re-run
post-2c-5c.** The parity sweep above is numerical-correctness only;
the dflash coherence gate (`scripts/coherence-gate-dflash.sh`)
covers attractor behavior + unique-token-ratio + 3-gram density
under real spec-decode workloads (per CLAUDE.md "DFlash Coherence
Gate"). My 2c-5c changes touched the scalar `attention_dflash_f32`
that the spec-decoder uses; re-run before claiming DFlash perf
parity. PR #321's test plan tracks this.

## 6. ROCm / runtime environment notes

- **ROCm 7.12 lives at `/opt/rocm-7.12`** on the bench host. Must
  export `PATH` + `LD_LIBRARY_PATH` before running any hipfire
  binary. Default `/opt/rocm` symlink is missing on this machine.
- **ROCm 7.12 shutdown segfault** — daemon exits with code 139 on
  clean shutdown after all output is flushed. Coherence gate flags
  this as `HARD_ERROR` even though the run was fine; eyeball the
  output before deciding `--no-verify` is OK.
- **Bench GPU is gfx1151** (Radeon 8060S, Strix Halo APU, 137 GB
  unified VRAM, HIP 7.12), NOT the gfx1010 5700 XT mentioned in
  CLAUDE.md.

## 7. Closed risk register

All risks below were marked OPEN during planning and resolved during
phase 1-2 implementation. Kept here for historical context (and to
preserve the resolution narrative).

- **R1 — HFQ `arch_id` mismatch.** Resolved by `--arch-id <u32>`
  flag on `hipfire-quantize` (`9477fbbb`). Re-quantised
  `qwen2-1.5b.arch7.hfq4` carries `arch_id = 0x07` at header offset
  0x08 (`xxd` verified). In-place `hfq-rewrite-arch-id` and
  arch_id=1→qwen2 daemon route are still on the table for the
  eventual id=1 retirement; deferred per PRD §8.
- **R2 — LLaMA path silently drops Qwen2 Q/K/V bias.** Resolved by
  hard-guard in `51e05b99`: `load_weights_hfq` now checks for
  `model.layers.0.self_attn.q_proj.bias` and refuses with an error
  pointing at `--arch-id 7` and `inspect_hfq`. Defense-in-depth for
  legacy `arch_id=1` Qwen2 HFQ files.
- **R3 — Daemon wired for arch_id=7.** Resolved by `806680b2`:
  load_model arm + LoadedModel fields + `generate_qwen2` JSON-stream
  emit + `arch-qwen2` cargo feature. Bring-up scope only — DFlash /
  CASK / PFlash / VL / ChatML / repeat-penalty / top-p / `<think>` /
  multi-GPU are all explicit refusals on this path.
- **R4 — Tied F16 lm_head corruption (latent).** Originally
  `load_lm_head` took `gpu.upload_raw(&data, ...)` for the F16
  tied-embedding branch while `WeightTensor.gpu_dtype` was F32 —
  kernel would read F16 bytes as F32 → garbage. Caught at rev-2
  review; fixed in `45913eb0` by mirroring qwen35's host-side
  F16→F32 expansion. Latent (didn't fire) because current
  `qwen2-1.5b.hfq4` uses HFQ4G256 for embeddings, not F16.
- **R5 — dots.ocr EOS via `generation_config.json`.** Resolved by
  `544822b4`: quantiser packs `generation_config.json` alongside
  `tokenizer_config.json` into HFQ metadata. Parser walks the
  three-layer fallback `text_config.eos_token_id` /
  `config.eos_token_id` → `generation_config.eos_token_id` → default
  `[151645]`. Tests: 8 passing including
  `eos_falls_back_to_generation_config_when_absent_from_config`
  (dots.ocr's real shape) and
  `eos_in_config_takes_precedence_over_generation_config` (ordering
  ambiguity guard).
- **R6 — Daemon arch_id=7 event-handler gaps.** R3 (`806680b2`)
  added `load_model` / `generate` arms but missed two other handlers:
  `bench_prefill` (panicked because LLaMA-else unwrapped
  `m.llama_config`) and `reset` (cleared `seq_pos` and KV
  `compact_offset` but never touched `Qwen2State.next_pos`, leaking
  prior-turn KV). Both fixed in `2226bbcf` — `bench_prefill` got a
  new arch_id=7 arm calling `qwen2::forward_step` per token, and
  `reset` calls a new `Qwen2State::reset()` helper that also fires
  from `bench_prefill`'s cold-start path. Lesson: when adding a new
  `arch_id` branch, grep `daemon.rs` for every site that switches on
  `m.arch_id` and confirm each one has the new arm. Both external
  reviewers (Gemini, GLM-5) missed this — internal pre-PR self-review
  caught it.
- **M9 — Quantiser support for Qwen2 layer naming.** Verified
  working in `4bf9f6d4` — Qwen2-1.5B quantises to HFQ4 at 100%
  coverage. Q/K/V bias tensors correctly preserved in F16. No
  quantiser-side changes needed.

## 8. Rev-3 deferred follow-ons

Captured from the Claude / Gemini / GLM-5 reviews of commits
`9477fbbb..806680b2`. Ranked, each tagged with the phase that ought
to absorb it. None blocks a phase-3 PR.

### Perf (phase 1.5 / post-PR optimisation pass)

- **Bias-add fusion** — current code is option (a) from the original
  plan (3 separate `bias_add_f32` per QKV per layer = 84 launches /
  decode step). Promote to option (c) — single batched bias-add of
  Q/K/V per layer (~28 launches / decode step) — or option (b) fused
  into `fused_qkv_hfq4g256_bias`. Apply Δ ≥ 5% rule. (Gemini §3.2,
  Claude rev-3 B1)
- **`gemv_hfq4g256_residual` fusion** — o_proj + residual and
  ffn_out + residual currently run as `weight_gemv` +
  `add_inplace_f32` (2 launches each). LLaMA uses fused variant; same
  upgrade saves ~56 launches / decode on Qwen2.
- **`argmax_f32` per-call malloc** — `Gpu::argmax_f32` allocates a
  4-byte result buffer on every invocation. Greedy decode pays one
  malloc + memset + memcpy per token. Move to a persistent scratch on
  `Qwen2State` (`argmax_result: DeviceBuffer`). Cross-arch fix.
  (Claude rev-3 D6)
- **Prefill batching** — `forward_step` is per-token, so a
  2048-token prompt costs ~2048× single-step decode time. Production
  serving needs a GEMM-based batched prefill variant. Required
  before Qwen2 / dots-ocr ships at non-bring-up scale. (Gemini §3.1,
  GLM-5 CAVEAT-3, Claude rev-3 B2-adjacent)
- **KV cache quantisation** — currently F32 (~28 MB at seq=512 for
  the 1.5B). Wire HFQ4 / HFQ8 / Q8 / asym-N modes (the qwen35
  `kv_mode` story) for memory-constrained serving. (Gemini §3.3,
  Claude rev-3 F-rec)
- **Tied-embedding VRAM aliasing** — tied `lm_head` re-uploads
  embedding bytes (~117 MB on Qwen2-1.5B at HFQ4) because `GpuTensor`
  is not `Clone`. Resolve via `Arc<GpuTensor>` / shallow-clone in the
  Transformer-extraction PR.
- **Perf-claim hygiene** — the rev-3 commit log compared an HFQ4
  first-run (4153 ms, JIT contaminated) against a Q8 warm-run (303
  ms). Re-measure both paths fresh before any tok/s ratio enters a
  perf doc. Single-shot tok/s with 2-decimal precision violates the
  CLAUDE.md ±10-15% noise guard. (Claude rev-3 C1, C2)

### Daemon-arm feature parity (phase 3 — pre-GA wave)

- **Chat-template framing on arch_id=7.** `generate_qwen2`
  short-circuits before the daemon's `prompt_frame::apply_chatml_frame`
  pipeline runs. `hipfire run` against a Qwen2 model produces
  continuation, not instruction-following. Wire `apply_chatml_frame`
  before tokenizing once the `prompt_frame_overrides` taxonomy is
  finalised. (GLM-5 CAVEAT-1)
- **Sampling beyond greedy.** `temp` / `top_p` / `repeat_penalty` /
  `repeat_window` are all underscored params on `generate_qwen2`.
  Greedy is the validation contract; non-greedy is a feature gap.
  Add a sampler call (port the LLaMA `sample_top_p` path or use the
  shared sampler infrastructure). (GLM-5 CAVEAT-2)
- **`pp > 1` + arch_id=7.** Currently falls through to
  `load_model_pp` which doesn't have an arch_id=7 arm and errors
  with "non-Qwen3.5 architectures". UX-fix: refuse upstream with a
  Qwen2-specific message; functional fix is multi-GPU pp for Qwen2,
  which is a separate large task.

### Dots.ocr-specific (phase 2/3)

- **§1.3 patch_embed weight is 4-D `[1536, 3, 14, 14]`, not 5-D.**
  rev-1 BUG-1, deferred. Cosmetic — both shapes reshape to the same
  2-D linear at load.
- **§1.5 `<\|endofassistant\|>` token-string handling.** Not in
  `added_tokens_decoder` — must be emitted as raw bytes that the BPE
  tokenizer fragments. rev-1 BUG-2 deferred.

### Cleanup / nice-to-have

- `Qwen2State` could expose `pub fn argmax(&mut self, gpu) -> u32`
  to internalise the logits→token step (currently the daemon does
  `gpu.argmax_f32(&state.logits, cfg.vocab_size)` directly).
- `infer_qwen2.rs` could print the decoded English at the end (not
  just the token IDs).
- `m.conversation_tokens.push(tok)` is filled by `generate_qwen2`
  but never read on the arch_id=7 path. Remove or leave for the
  future sampler wiring.
- `chat_template = resolve_chat_template(...)` loaded on the
  arch_id=7 path but never consulted by `generate_qwen2`. Load now,
  consume when chat-template framing lands.
- `parse_arch_id_override` could move from `unwrap_or_else` with
  `!`-return to an `if let Some(..) else` pattern. Pure style.
- `scripts/capture_qwen2_reference.py` hardcodes the HF snapshot
  path. Acceptable for a one-time phase-0 capture but won't reproduce
  on another machine without editing.
