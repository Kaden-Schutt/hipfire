# Glimmer (arch 14) KLD oracle — MI300X runbook

**Date:** 2026-08-14
**Status:** operational runbook. Not a perf checkpoint, not a claim.
**Purpose:** produce a hipfire-native F32 KLD reference for Muse Glimmer 30B
so MQ4 / AWQ candidates can be ranked on quality, and record why that step
needs rented CDNA3 hardware.

---

## 0 · The F32 oracle has three consumers, not one

**Correction to this document's original framing.** It was written as "build a
KLD reference," which undersold the rental. The F32 oracle is the shared input
to three separate deliverables, and the most valuable one is not the KLD ref:

| consumer | input | produces | why F32 is required |
|---|---|---|---|
| `collect_imatrix_native` (branch `origin/feat/hfim-native-imatrix`) | F32 oracle HFQ | **native HFIM** | captures each linear's RAW input, "pre-rotation, pre-AWQ-scale — the f32 path has neither". A quantized build applies FWHT rotation + AWQ scales and corrupts the stats. |
| `build_kld_ref_native_glimmer` | F32 oracle HFQ | KLD reference | the reference must be the engine's precision ceiling |
| `scripts/collect_hessian.py` | BF16 HF safetensors | HFHS sidecar | GPTQ Hessian; PyTorch-side, independent of hipfire |

### The native HFIM is the quality lever, and it already exists

`origin/feat/hfim-native-imatrix` (unmerged; merge-base with this branch is
`a7a8d89bb`, 2026-04-03) carries a COMPLETE native imatrix stack:

```
3a2d530b2  feat(quantize): HFIM native imatrix container (imatrix_io.rs)
5afbdc384  feat(imatrix): native HFIM collector + AWQ wiring (no GGUF, no remap)
60adf9ba0  fix(imatrix): key HFIM by the resolved full tensor name (== quantizer lookup)
b7507ef51  feat(imatrix): native HFIM blend corpus-prep + agentic-KLD harness
11d3fd245  feat(eval): top-1 agreement in eval_hipfire + chat-model HumanEval harness
```

It exists because borrowing the imatrix from a llama.cpp GGUF "carried two
confounds: (1) the llama.cpp tokenizer disagrees with hipfire on ~46% of token
positions, and (2) the GGUF↔safetensors name remap silently no-op'd AWQ on
27B-3.6 hybrid `linear_attn` names." Glimmer's current AWQ F1/F2 are built from
bartowski's published GGUF imatrix and therefore carry confound (1).

Capture mechanism is arch-generic: `gpu.imatrix_capture = Some(..)` and every
`weight_gemv` — "the single chokepoint" — accumulates `Σ act²` keyed by
`WeightTensor.name`. The per-arch commits (qwen35 +41 lines, qwen2 +9) are
plumbing; Glimmer needs the equivalent, and Glimmer did not exist in June.

### Calibration corpus: bartowski-v5 blended with agentic ChatML

`scripts/build_hfim_corpora.sh` on that branch composes:

- **bartowski-v5** raw multilingual/code/structured — "diversity /
  outlier-channel coverage (what AWQ protects)"
- **agentic ChatML** via `scripts/fetch_calibration_corpus.sh` — "matches
  hipfire's DEPLOYMENT distribution (chat + tool + flow)"

emitting `hfim-blend-calib.txt` plus a DISJOINT held-out `hfim-agentic-eval.txt`
(`SEED=1182`, `EVAL_FRAC=0.12`, md5-pinned), designed so the eval slice "neither
favors the calib corpus nor leaks into it."

### Sequencing consequence

Cherry-pick the five HFIM commits onto this branch rather than merging the whole
April-divergent branch (2803 files). The core is ~500 lines: `imatrix_io.rs`
(211), the collector (170), `dispatch.rs` (+51), arch wiring (+41). Then add the
Glimmer arch wiring. Do this BEFORE the rental so the oracle serves all three
consumers in one window.

---

## 1 · Why this needs a rented box

Glimmer's high-precision oracle must be **111 GB resident**, and no locally
owned GPU can hold it.

Parameter count, from Glimmer's real shapes (52 layers, hidden 6656,
intermediate 19968, vocab 202048, GQA 32q/2kv, head_dim 128 — see
`crates/hipfire-arch-muse-glimmer/src/lib.rs:5-31`):

| component | params | F32 |
|---|---|---|
| 52 × (3 × 27.26M attn + 2 × 1.70M kv + 3 × 132.91M ffn) | 25.16 B | 100.7 GB |
| embed + lm_head (2 × 202048 × 6656, `tie_word_embeddings: false`) | 2.69 B | 10.8 GB |
| **total** | **27.85 B** | **≈111 GB** |

### The file format is irrelevant — VRAM is always F32

`load_wt` (`crates/hipfire-arch-muse-glimmer/src/glimmer.rs:1069-1129`)
resolves every high-precision quant type to `DType::F32` in VRAM:

- `qt=1` (F16) → decoded to `Vec<f32>`, `upload_f32`, `gpu_dtype: DType::F32`
- `qt=2` (F32) → `upload_f32`, `gpu_dtype: DType::F32`
- anything not in {1, 2, 3, 4, 6, 7, 8, 9, 11, 13, 15, 17, 19} → hard refusal,
  `glimmer: unsupported quant_type {qt}`

So a 56 GB BF16 file and a 111 GB F32 file occupy **identical** VRAM. Choosing
a smaller container saves disk and download time, never footprint.

### The gfx1151 escape hatch is closed to Glimmer

`crates/hipfire-quantize/src/main.rs:7845-7848` records the precedent: the
cohere2moe oracle stored "the EXACT downloaded bf16 bytes (~61 GB, fits
gfx1151's GTT; the all-F32 `oracle` passthrough is 122 GB and does NOT fit)."

That trick requires a loader arm that keeps bf16 resident. **Glimmer has no
BF16 arm**, and adding one would not help anyway — mirroring the existing
`qt=1` path widens on upload, landing back at 111 GB. Making 56 GB actually
fit would mean bf16-resident weights plus in-kernel dequant in every Glimmer
GEMV: real kernel work, not a patch.

### No local box fits, and Glimmer cannot be split

| device | capacity | verdict |
|---|---|---|
| MI300X (rented, gfx942) | 192 GB | **fits** |
| hipx gfx1151 (Strix Halo) | 103.1 GB | misses by ~8 GB |
| hiptrx 4 × R9700 (gfx1201) | 32 GB each | Glimmer refuses `pp>1` — cannot pool |
| hipx 7900 XTX (gfx1100) | 24 GB | no |
| local 9070 XT (gfx1201) | 17 GB | no |

hipfire is ROCm-only, so NVIDIA rentals (H100/H200) are not substitutes. The
rental class is specifically **CDNA3**: RunPod, TensorWave, Hot Aisle, Vultr.
A single MI300X is sufficient — no multi-GPU node required.

### Why not a cross-engine reference

llama.cpp's KLD carries a documented cross-engine offset (~+0.3; `41b3efc07`
flags its own numbers as untrustworthy for being cross-engine). The June A3B
work already moved to preferring hipfire's own f32-native HFKLDR for ranking
quants. The oracle must be the engine's own ceiling, which for Glimmer is F32.

### What the rental buys

A **permanent** artifact, and a small one. Every future Glimmer quant scores
against it for free on local hardware — one rental, never repeated for this
model.

Size is exactly predictable from the format:

```
bytes = 32                                  header
      + n_chunk * n_ctx * 4                 token block
      + n_chunk * scored_per_chunk * (8 + 8 * top_k)
where scored_per_chunk = n_ctx - 1 - n_ctx / 2
```

At `top_k=256, n_ctx=512` that is 526,328 B per chunk. Two references on this
box confirm it: `q36a3b-wt2-f32.kldref.bin` is 16,842,528 B, which is
`(16842528 - 32) / 526328 = 32.0` chunks exactly, and
`qwen3.6-27b-MASTER-small.kldref.bin` is 204,813,592 B. Budget **tens to a few
hundred MB** depending on corpus size — trivial to retrieve over a home link.

---

## 2 · Box spec

- 1 × MI300X, gfx942, 192 GB HBM
- **≥ 450 GB disk** if you keep everything resident, ~250 GB if you sequence
  (see below). The consumers:

  | artifact | size |
  |---|---|
  | source safetensors (bf16) | 59.5 GB |
  | F32 oracle HFQ (`--format f32`) | 111 GB |
  | **HFHS Hessian sidecar** | **≈ 142 GB** |
  | candidate for the Step 0 gate | 18 GB |

  HFHS-v1 stores a **full K×K F32** payload per tensor
  (`scripts/collect_hessian.py:36-42`), so Glimmer's Hessian is *larger than
  its oracle*: per layer, six K=6656 tensors at 177.2 MB (q, k, v, attn gate,
  mlp gate, up) + one K=4096 at 67.1 MB (o_proj) + one **K=19968 at 1,595 MB**
  (down_proj) = 2.73 GB, times 52 layers ≈ 142 GB.

  **Sequence to halve the peak:** build the oracle, run the reference pass,
  then delete the 111 GB oracle before starting the Hessian collection. Peak
  becomes ~max(111, 142) + 59.5 + 18 ≈ 220 GB. Provision ~250 GB for the
  sequenced route, ~450 GB to keep both.

  **Never move the HFHS home.** At 142 GB it is worse than the oracle. Run the
  GPTQ quantize *on the droplet* — the quantizer consumes HFHS + source and
  emits a ~16 GB artifact — and retrieve only that plus the `.kldref`.
- ROCm present; `cargo` toolchain
- Good HF bandwidth — the point of building on the droplet is that the 59.5 GB
  download and the 111 GB quantize both happen box-local. **Never upload
  111 GB from home.**

---

## 3 · Step 0 — correctness gate BEFORE anything expensive

Glimmer's arch-14 CDNA3 branches have never executed on any locally owned
device. `crates/hipfire-arch-muse-glimmer/src/forward.rs:2838-2841` and
`3315-3319` both document branches that are "inert on RDNA … live on CDNA3",
and state the failure mode is **"silent wrong activations rather than an
error."** An oracle built on a wrong path is a silently poisoned reference
that would contaminate every number derived from it, permanently.

Qwen3.6 **dense** already runs well on MI300X, so the shared CDNA3 surface
(rocblas eligibility, fp16 shadow cache, MFMA kernels, graph capture, ROCm) is
proven. The residual risk is Glimmer-specific glue only: the pre-`o_proj`
attn-gate ordering and `ensure_fp16_x`.

**Gate:** copy up the existing 18 GB `muse-glimmer-30b.mq4`
(sha256 `87bc776a36e0c4b55edfd1f0da1104c966eb43f59aa6fdd0ee33a14ecc0fc052`)
and require **byte-identical greedy output** against the known gfx1201 result
on the canonical fixture `benchmarks/prompts/merge_sort_thinking_off.txt`
(md5 `253c7ac50857fe6d0e10fb0d2c5e35c0`), temp 0.0.

Cost: ~15 minutes of rental. If it diverges, **stop** — that is a real CDNA3
correctness bug, a much larger job than the eval, and nothing from that box is
trustworthy until it is fixed.

### KV is Q8-only and dual — there is no F32 KV path

Glimmer's KV cache is **Q8, always**, and allocated *inside* the state object:
`GlimmerState::new_with_max_seq` (`glimmer.rs:1721`) builds both caches itself
(`glimmer.rs:1870,1898,1909`), giving `GlimmerState { kv_sliding, kv_full,
kv_slot_for_layer }` — 39 sliding layers (window 2048) plus 13 full layers
(window 0, i.e. NoPE), both `head_dim` 128, `n_kv` 2.

There is no `KvCache::new_gpu` (F32) path for Glimmer, because the forward is
Q8-only: `kv_cache_write_q8_0` / `attention_q8_0_kv_swa`
(`crates/hipfire-arch-muse-glimmer/src/forward.rs:725-814`). Passing F32 KV
would produce garbage, not a higher-precision reference.

Consequently the gemma4 asym3/gfx942 warning **does not apply here** — Glimmer
never uses asym3, so there is no catastrophic-KV mode to avoid.

### The Q8-KV noise floor does NOT cancel — this bounds what can be claimed

It is tempting to argue that because both arms use identical Q8 KV, the
quantization noise cancels and the divergence isolates weight precision. **That
is false, and the gemma4 arm says so.** `build_kld_ref_native_gemma4.rs:150`
states that F32-on-both-sides "removes the shared KV-noise **floor**" — it
treats KV quantization as a floor to be eliminated, not as a term that cancels.

It cannot cancel here, because the quantizer is applied to *different values on
the two sides*: the reference's K/V are computed from F32 weights, the
candidate's from MQ4 weights. Same rounding function, different arguments →
uncorrelated errors that **add**. Identical configuration equalizes the noise
*distribution*; it never equalizes the pointwise error.

So Glimmer's reference is **F32-weights + Q8-KV**, not a full-precision oracle,
and a floor survives in every number it produces. Two consequences:

- **Supported claim:** a *relative ranking* — which candidate sits closest to
  F32 weights under a fixed Q8-KV regime.
- **Unsupported claim:** any absolute "Glimmer MQ4 costs X KLD" figure. The
  reported value contains a KV term of unknown magnitude.

Removing the floor would require an F32-KV attention kernel for Glimmer's
shape, which does not exist. That is a project, not a step in this runbook.

**Failure mode this creates for the rental:** if the Q8-KV floor exceeds AWQ's
weight-side improvement, the eval returns "indistinguishable" for reasons
unrelated to AWQ — reproducing the exact discrimination failure that motivated
the work (AWQ F1's greedy output is already byte-identical to the trunk).

**De-risk locally first, for free.** Before renting, use the Q8-attention
artifact (`87bc776a…`) as a *stand-in* reference and score trunk `.mq4r`, AWQ
F1 (`c67d3066…`), and AWQ F2 (`cf2f012e…`) against it. All are 16–18 GB and fit
a 17 GB card; the code path, harness, and Q8-KV floor are identical to the real
run. If the three separate cleanly, the harness discriminates and the F32
oracle will yield signal. If they collapse to the same KLD, the floor swamps
AWQ — learned at zero cost.

Both arms must still allocate KV identically, for a weaker but real reason: it
keeps the floor *common-mode* across candidates so the ranking stays
meaningful. Both construct state through the same `new_with_max_seq` with
`kv_max = n_ctx + 16`, and both pin `HIPFIRE_GLIMMER_KV_VMM=1` explicitly —
Glimmer chooses between `new_gpu_q8_vmm_capped_filtered` and `new_gpu_q8` on
that variable (`glimmer.rs:1864`), so leaving it implicit in one arm would let
the two diverge silently for anyone who has it set.

**Residual gfx942 risk this creates.** Because Glimmer's attention is Q8-KV
only, the Step 0 gate is also the test of whether `attention_q8_0_kv_swa` and
`kv_cache_write_q8_0` have working CDNA3 paths. If they are gfx11/gfx12-gated
with no gfx942 arm, Glimmer cannot run on MI300X at all and the rental is void
— which is precisely why the gate runs on an 18 GB candidate before any
111 GB oracle is built.

---

## 4 · Steps

### 1. Build

```bash
cargo build --release -p hipfire-runtime \
  --example build_kld_ref_native_glimmer \
  --example eval_hipfire_glimmer \
  --example daemon
```

### 2. Fetch the source at the SAME revision as the candidates

The oracle is only comparable to candidates quantized from identical weights.
The local trunk was built from `meta-models/Muse-Glimmer-30B` revision
`97c77dff…`; pin that exact revision, not `main`.

Upstream is `apache-2.0` and ungated, 165.3k downloads — no access gate.

### 3. Quantize to the F32 oracle

```bash
hipfire-quantize --format f32 <source-dir> muse-glimmer-30b-f32-oracle.hfq
```

`--format f32` stores **every** tensor as `QuantType::F32` (qt=2) — weights,
norms, embeddings — widening the bf16 source losslessly
(`crates/hipfire-quantize/src/main.rs:7793-7798`). Expect ~111 GB.

Do **not** use `--format oracle` or `--format bf16` here: both also set
`use_bf16` (`main.rs:7848`), and Glimmer's loader has no BF16 arm, so the
result risks a hard `unsupported quant_type` refusal at load. `f32` is the
unambiguous spelling.

### 4. Build the reference

```bash
./target/release/examples/build_kld_ref_native_glimmer \
  --model muse-glimmer-30b-f32-oracle.hfq \
  --slice <slice.txt> \
  --output muse-glimmer-30b-f32-native.kldref.bin \
  --top-k 256 --n-ctx 512
```

Sanity-check with `--max-chunks 4` first and confirm the reported `ORACLE mean
NLL` / `PPL` are plausible before committing to the full corpus. A wildly high
PPL here is the same signature as the gemma4 asym3-on-CDNA failure.

### 4b. Collect the GPTQ Hessian in the SAME window — it is the long pole

`scripts/collect_hessian.py` is a working, **arch-agnostic** HFHS-v1 producer
and it is plug-and-play for Glimmer with no code change:

- loads any transformers model via `AutoModelForCausalLM.from_pretrained`
  (`:326`) — no arch table, no `trust_remote_code`
- discovers targets generically: `model.named_modules()` +
  `isinstance(module, torch.nn.Linear)` + `is_gptq_target` (`:353-354`)
- **matches Glimmer's oddball attention gate**: `is_gptq_target` keys on the
  last name component only, so `self_attn.gate_proj` → `gate_proj` → hit. This
  is the same tensor family that needed the `safetensors_to_ggml_name` fix
  (`"self_attn.gate_proj" => "attn_gate"`, `425c5031b`).
- **handles Glimmer's multimodal prefix**: HF drops `language_model.` when
  loading as a CausalLM, so `_translate_to_stored_name` maps in-memory names
  back to stored safetensors keys (`:343-360`) — exactly Glimmer's
  `model.language_model.*` layout — "so the Rust quantizer can look up
  Hessians by the same key as the .hfq tensors."

Output is HFHS-v1, read unchanged by `crates/hipfire-quantize/src/hessian_io.rs`.
The consumer side (`BlockHessianAcc`, `.hblk` E8H1, `load_hessian_blocks`) is
**not** MoE-gated, so **GPTQ-E8 applies to dense Glimmer with no new code** — a
stronger quantization lever than AWQ, which is currently sourced from a
third-party imatrix.

Two scheduling facts make this the right thing to run first:

1. **It is the long pole.** `crates/hipfire-runtime/src/bin/collect_hessian.rs`
   quantifies the Tier-2 Python path at ~8 h for a 27B-class model on MI300X
   (its unbuilt Tier-1 target is 20× faster, ~25 min). The oracle pass is
   hours, not overnight. Start the Hessian job, let it run.
2. **It does not depend on the Step 0 gate.** The collector runs HF
   transformers + PyTorch, not hipfire's forward, so a Glimmer CDNA3 defect
   cannot spoil it. Even in the worst case where Step 0 fails and the oracle is
   abandoned, the rental still returns a usable Hessian.

It needs ~60 GB of BF16 weights resident plus activations — which is why this
only fits on the rented box, and why doing it in the same window avoids a
second rental.

### Why not wait for the native collector

There is no functional hipfire-native generator for either artifact, so there
is nothing to wait for:

- `src/bin/collect_hessian.rs` is a self-described "Foundation scaffold"
  (`#[allow(dead_code)] struct Args`) blocked on unbuilt Task 3/4 pieces.
- `examples/imatrix_collect.rs` is explicitly Tier 2 — a subprocess wrapper
  around llama.cpp's `llama-imatrix`. Its header prices native imatrix capture
  at 6-10 days (dispatch hooks + an on-GPU sum-of-squares reduce kernel).
- `src/calibration.rs` already implements **both** accumulators — per-channel
  `Σact²` (`:236`, shape `[k]`) and the K×K Hessian (`:391`, shape `[k,k]`).
  Only the dispatch taps are missing, which matches the earlier finding that
  `Gpu::hessian_capture` is consumed solely inside `run_moe_decode_cpu_fallback`
  (`hipfire-dispatch/src/pipeline/mod.rs:1319-1398`), a path dense Glimmer
  never reaches.

### 5. Retrieve and tear down

Bring home only:

- the `.kldref` — tens to a few hundred MB, sized per the formula in § 1
- any GPTQ-E8 artifact you quantized on the box (~16 GB)

Leave behind the 111 GB oracle and the ~142 GB HFHS. Both are reproducible from
the pinned source revision and the recipes above, and both are larger than the
products derived from them — which is the whole reason the quantize runs on the
droplet rather than at home. Then destroy it.

### 6. Score locally, free

Candidates are 16–18 GB and run on any local card:

```bash
./target/release/examples/eval_hipfire_glimmer \
  --model <candidate.hfq> --ref muse-glimmer-30b-f32-native.kldref.bin
```

Candidates to rank: MQ4 trunk, AWQ F1 (`c67d3066…`), AWQ F2 (`cf2f012e…`),
and the Q8-attention `.mq4` (`87bc776a…`). Reduce with
`benchmarks/quality-baselines/harness/kld_reduce.py`.

This is the measurement that was previously impossible: AWQ F1's greedy output
is byte-identical to the trunk on the merge_sort fixture, so that fixture
cannot discriminate between them. KLD over a real corpus may — subject to the
Q8-KV floor in § 3, which is why the free stand-in-reference run comes first.

---

## 5 · What this does not settle

**It is not a full-precision oracle.** The reference is F32-*weights* with
**Q8 KV**, because Glimmer has no F32 KV path. The Q8 term does not cancel
between the two arms (see § 3), so every reported figure carries a KV-noise
floor of unmeasured magnitude. What the numbers support is a *relative ranking*
of candidates under a fixed Q8-KV regime; what they do not support is an
absolute "Glimmer MQ4 costs X KLD" claim. Removing the floor needs an F32-KV
attention kernel for Glimmer's shape, which does not exist.

**It is not an HF-parity claim.** The ceiling here is hipfire's own F32 forward,
deliberately — a cross-engine reference carries a ~+0.3 offset. Agreement with
HuggingFace is a separate question this does not touch.

**It says nothing about the vision tower.** Arch 14 is the dense text tower
only, parsed from `text_config`, tensors under `model.language_model.*`, while
upstream's `pipeline_tag` is `image-text-to-text`.

**Wave width is not a gfx942 risk.** `kv_cache_write_q8_0.hip:31-33` reduces
with `__shfl_xor` over `offset = 16,8,4,2,1` — a 32-lane reduction over a
32-element Q8_0 block. XOR offsets ≤ 16 never cross a 32-lane boundary, so on
wave64 each half-wave reduces its own block independently. Correct on wave32
and wave64 alike. The CDNA3 exposure is confined to the two documented
`is_cdna3()` branches in `forward.rs` (2838-2841, 3315-3319), which is what
Step 0 exists to test.
