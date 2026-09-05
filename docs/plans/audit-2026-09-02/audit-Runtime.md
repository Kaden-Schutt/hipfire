<!-- SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Kaden Schutt; hipfire — see LICENSE and NOTICE in the project root. -->

# Audit: Runtime

# Audit — Runtime

Slice: `crates/hipfire-runtime/src/**` (except config/loader_api/multi_gpu ownership). Focus: HFQ/qt registry vs GEMV/GEMM, free_gpu/Drop, reset_core arch keys, spec rollback, sampler/eos emit.

## Broken

### 1. LlamaWeights::free_gpu frees `.buf` only — AWQ/PARO sidecars leak on unload
- **path_line:** `crates/hipfire-runtime/src/llama.rs:683-706` vs `WeightTensor::free_all` at `llama.rs:525-540` and DFlash `dflash.rs:946-975`
- **verified:** true
- **how known:** Read both. `LlamaWeights::free_gpu` does `gpu.free_tensor(l.wq.buf)` (and gate/up/down/output.buf). `WeightTensor::free_all` frees `paro` pairs/theta/scales (unless alias) and `awq_scale` then `buf`. DFlash comments explicitly: "free_all (not .buf) so the awq_scale / paro sidecars are released too — .buf-only freeing leaks one tensor per weight per layer". HFQ `load_weight_tensor` attaches AWQ when `supports_awq_sidecar()` (`hfq.rs:1485-1490`). Qwen35 weights path correctly uses `free_all`.
- **impact:** Unload/reload of llama (or any consumer of `LlamaWeights::free_gpu`) with AWQ/PARO HFQ silently leaks VRAM proportional to weights×layers.

### 2. weight_gemv builds incomplete WeightRef (row_stride=0, awq_scale=None, rotation=None)
- **path_line:** `crates/hipfire-runtime/src/llama.rs:740-748` vs correct adapter `llama.rs:545-560` (`dispatch_ref`)
- **verified:** true
- **how known:** Read `weight_gemv`: hardcodes `row_stride: 0`, `rotation: None`, `awq_scale: None` while `WeightTensor::dispatch_ref` wires `row_stride`, Givens, and `awq_scale`. Paro path re-reads `w.paro` separately; MQ rotation uses `rotate_x_mq_for(gpu, w, …)` so AWQ-on-activation may still apply there — but any kernel path that reads AWQ/row_stride from `WeightRef` (MQ4C padded stride, weight-side AWQ) gets wrong metadata. GL/MQ4C/dense AWQ consumers of this entry point are at risk.
- **impact:** Silent wrong decode or dropped sidecars on formats that need non-zero `row_stride` or weight-ref AWQ; dead `dispatch_ref` helper.

### 3. MQ2G256GL / MQ3G256GL load as passthrough but have no dense GEMV/prerotated key
- **path_line:** `weight_backend.rs:411-422` (RAW_CODECS qt 38/39); `hipfire-dispatch/src/types.rs:120-125` (FwhtG256); `types.rs:730-776` (`for_gemv_prerotated` Err for rotation dtypes not listed); tests `hipfire-dispatch/src/tests.rs:1348-1365` (`gl_dtypes_have_no_dense_gemv_key`)
- **verified:** true
- **how known:** RAW_CODECS admits 38/39. Tests assert plain and prerotated GEMV keys are Err (MoE-indexed only). `weight_gemv` residual arm for FwhtG256 rotates then `GemvVariant::Prerotated` → `for_gemv_prerotated` → UnsupportedVariant. Loader admits tensors the dense engine cannot run (archetype of #683).
- **impact:** Dense GL weights fail at first GEMV; or if any fallthrough mis-routes, garbage. MoE-indexed path is the only supported use.

### 4. HFQ `load_weight_tensor` vs `dequant_weight_raw` F16/F32/BF16 divergence
- **path_line:** `hfq.rs:1456-1483` (qt1→host F32 only; else raw_codec or error); `weight_backend.rs:681-724` (qt1 keep F16; qt2 F32; qt16 BF16→F32)
- **verified:** true
- **how known:** Comments in both files document divergence. HFQ path never special-cases qt 2 or 16 → `raw_codec` None → "unsupported quant_type". Safetensors/backend path keeps F16 on device for qt1 (native GemvF16) while HFQ forces F32 GEMV. Same on-disk qt, different GPU dtype and kernel family.
- **impact:** F32/BF16 weight tensors in HFQ fail load; F16 HFQ models pay 2× weight bandwidth and lose F16 GEMM/GEMV path vs ST path.

### 5. dots_ocr arch_key spelling contradicts reset_core inventory and generate map
- **path_line:** `hipfire-arch-dots-ocr/src/arch_model.rs:24-26` (`"dots_ocr"`); `reset_core.rs:140-152` (`arch: "dots-ocr"`); `hipfire-generate/src/ar.rs:4608-4628` (`8 => "dots-ocr"`); contract comment `arch_model.rs:70-72` ("Matches the key used by reset_core … so the two cannot drift")
- **verified:** true
- **how known:** Cross-read three sites. `reset_coverage_for(bundle.arch_key())` → None for dots_ocr; `reset_coverage_for("dots-ocr")` → Some(Ineligible). Today both yield non-eligible, so retry behavior is coincidentally same; coverage checklist and any future Eligible flip or graph flags are invisible via ArchModel path. Contract comment is violated.

### 6. generate `reset_core_arch_key` drops maple (arch_id 15 → "unknown")
- **path_line:** `hipfire-generate/src/ar.rs:4608-4628`; maple `arch_key` `hipfire-arch-maple/.../bundle.rs:40-42` (`"maple"`); inventory `reset_core.rs:196-220` (MAPLE row present)
- **verified:** true
- **how known:** match arms stop at 14 muse_glimmer; `_ => "unknown"`. Inventory has maple; ArchModel returns maple; arch_id path never sees the row.

### 7. DFlash target_hidden row gap: warn once, still poison session
- **path_line:** `dflash.rs:1083-1100` (GAP_WARNED atomic; eprintln; continues); comments 1083-1088 (NaN hole collapses acceptance for rest of session, survives prompt-cache HIT)
- **verified:** true (code path exists; runtime trigger depends on caller advancing without hidden commit)
- **how known:** Read TargetHiddenLog commit path. Documented class (#462 / forced-advance holes). Mitigation is warn-not-fail; `spec.rs:743-747` documents forced-advance must fill hidden or leave hole.
- **impact:** Silent acceptance collapse until reseed; easy to miss in logs (warn-once).

### 8. Generic DFlash Speculator ignores SpecRequestConfig fields (min_p / ngram / full seed contract)
- **path_line:** `dflash_generic.rs:~1000-1015` (comments: New SpecRequestConfig fields min_p/rng_seed/ngram ignored; only temp/top_p/top_k/rng_state applied)
- **verified:** true
- **how known:** Read configure_request. Documented incomplete vs qwen35 DflashSpeculator.
- **impact:** LlamaCarrier generic DFlash sampling/ngram behavior diverges from request config; user thinks knobs work.

## Missing

### 1. No load-time refusal for passthrough qts without dense GEMV (GL and similar)
- **path_line:** RAW_CODECS + absence of gate in `decode_raw_codec` / model_load
- **verified:** true (gap)
- Loader admits MoE-only formats into WeightTensor used by dense `weight_gemv`/`weight_gemm` without tagging or checking consumer. Missing: disposition `moe-indexed-only` in qt-register + load assert or dtype capability bit.

### 2. MFP2/MFP3 E8 (and other arch-loaded Fwht dtypes) missing from `for_gemv_prerotated` explicit arms
- **path_line:** `hipfire-dispatch/src/types.rs:730-776` (lists MFP4* E8 but not MFP3G32E8/MFP2G32E8); rotation plan marks them FwhtG256 (`types.rs:124-125`)
- **verified:** true for key table; arch crates may use private gemv_auto
- Dense `weight_gemv` prerotated path would Err. Arch-loaded only in register (`qt-register.txt:36-37`) — OK if never routed through llama weight_gemv; missing shared capability matrix so contributors do not wire them into dense path by accident.

### 3. weight_gemm only special-cases a subset; rest is per-row GEMV fallback (incl. formats that then fail)
- **path_line:** `llama.rs:1475-1575` (`_ =>` loop weight_gemv)
- **verified:** true
- Missing batched GEMM for TQ2/BQ1/Lloyd/MFP/GL/etc. Fallback multiplies cost and inherits weight_gemv gaps (GL hard-fail).

### 4. reset_core inventory vs ArchModel arch_key not mechanically enforced for all arches
- **path_line:** `reset_core.rs` tests pin inventory strings; `arch_model.rs` comments require match; dots_ocr proves drift landed
- **verified:** true (test gap)
- Missing compile-time or CI check that every `ArchModel::arch_key` equals inventory / `reset_core_arch_key(arch_id)`.

### 5. EosFilter is solid post-stop; missing unified guarantee that all generate loops never emit after terminal without filter
- **path_line:** `eos_filter.rs:34-49`, `186-189` (stopped → Hold); module docs say daemon loops decode+ship
- **verified:** filter correct; full daemon wire audit not completed in this slice
- EmitAndStop contract (marker not in payload) is implemented. Residual risk is loops that bypass EosFilter or emit raw detok after Stop — hand off to Generate/Cli scouts.

### 6. Maple/Qwen MTP free_gpu still `.buf` on some WeightTensors (cross-slice)
- **path_line:** e.g. arch-qwen35 `mtp_head.rs` frees `eh_proj.buf` / expert `.buf` (seen in grep); Qwen35 main weights use free_all
- **verified:** true for MTP head pattern; primary LlamaWeights issue is in-slice
- Half-migration: free_all adopted in dflash/qwen35 trunk, not universal.

## Would change (ranked)

1. **LlamaWeights::free_gpu → free_all for all WeightTensors** (and audit every `free_tensor(.*\.buf)` on WeightTensor owners in runtime)
   - path: `llama.rs:683-706`
   - cost: **hours**
   - Fix leak class DFlash already documented; add unload smoke that counts device allocs with AWQ model.

2. **weight_gemv/residual/swiglu: use `w.dispatch_ref()` only**
   - path: `llama.rs:740+`
   - cost: **hours**
   - Delete hand-rolled WeightRef; single wire for row_stride/AWQ/Givens. Unit test MQ4C row_stride and AWQ flag propagation.

3. **Refuse or tag MoE-only dtypes at load (GL 38/39)**
   - path: `weight_backend.rs` + `qt-register.txt` + optional `DType::dense_gemv_supported`
   - cost: **hours–1 day**
   - Fail fast with clear error instead of first-token UnsupportedVariant; align register disposition.

4. **Unify HFQ vs dequant host-decode for qt 1/2/16**
   - path: `hfq.rs:1456+`, `weight_backend.rs:681+`
   - cost: **1 day**
   - Prefer keep-F16 for qt1 (match dequant + GemvF16); add qt2/16 arms to HFQ loader or route HFQ through dequant_weight_raw exclusively.

5. **Normalize arch_key spellings (dots_ocr ↔ dots-ocr) + extend reset_core_arch_key for maple**
   - path: dots-ocr arch_model, reset_core, generate ar.rs
   - cost: **hours**
   - Pick one SoT string; CI test: ∀ ArchModel key ∈ inventory keys and ∀ arch_id map value ∈ inventory.

6. **TargetHiddenLog gap: hard error or auto-reseed, not warn-once**
   - path: `dflash.rs:1083-1100`
   - cost: **1 day** (call-site fixes for forced-advance may be more)
   - Prevent silent session poison; pair with Generate scout on forced-suffix paths.

7. **Generic DFlash configure_request parity with SpecRequestConfig**
   - path: `dflash_generic.rs`
   - cost: **hours–1 day**
   - Wire min_p/ngram/seed or document unsupported and refuse non-default.

8. **Capability matrix: QuantType × load path × dense GEMV × GEMM × MoE × embed**
   - path: docs/quant-formats + scripts/check-quant-registry.py extension
   - cost: **days**
   - Prevent #683-class (loader admits, engine cannot run) systematically; include GL, LloydU, TQ2/BQ1, arch-loaded E8.

## Confidence

**Did:** Read qt-register, RAW_CODECS full table, dequant_weight_raw, hfq load_weight_tensor, LlamaWeights free_gpu vs free_all vs dflash free_all, weight_gemv/weight_gemm, for_gemv_prerotated + GL tests, reset_core inventory + generate arch_id map, dots_ocr/maple arch_key, TargetHiddenLog gap, EosFilter Stop/Hold, spec commit_prefix contracts, generic dflash configure_request.

**Did not fully:** Every residual/swiglu WeightRef twin; full dspark_core position rewind line-by-line; every daemon emit loop vs EosFilter (Generate/Cli); KvCache VMM free edge cases beyond free_gpu tests; safetensors_source upload vs mmap pager in depth; open GitHub issue cross-check via `gh` (no shell in this scout); GPU/runtime reproduction.

**Verified vs suspicious:** Items 1–6 and 8 verified by code read/cross-ref. Item 7 (hidden gap) verified as warn-continue behavior; actual production triggers need caller trace. MFP2/3 prerotated gap is real in dispatch table; may be intentional if only arch private paths use them — treat as missing guardrails unless a dense callsite is found.

## JSON summary (for parent merge)

```json
{
  "slice": "Runtime",
  "broken": [
    {"title": "LlamaWeights free_gpu leaks AWQ/PARO sidecars", "path_line": "crates/hipfire-runtime/src/llama.rs:683-706", "verified": true, "summary": "Frees .buf only; free_all exists and dflash uses it; AWQ attach on HFQ load."},
    {"title": "weight_gemv incomplete WeightRef", "path_line": "crates/hipfire-runtime/src/llama.rs:740-748", "verified": true, "summary": "row_stride=0, awq_scale=None, rotation=None; dispatch_ref unused."},
    {"title": "GL qts load without dense GEMV", "path_line": "crates/hipfire-runtime/src/weight_backend.rs:411-422", "verified": true, "summary": "qt 38/39 RAW_CODECS; for_gemv_prerotated Err; MoE-only by design."},
    {"title": "HFQ vs dequant F16/F32/BF16 split", "path_line": "crates/hipfire-runtime/src/hfq.rs:1456-1483", "verified": true, "summary": "HFQ qt1→F32, rejects 2/16; dequant keeps F16 and handles 2/16."},
    {"title": "dots_ocr arch_key drift", "path_line": "crates/hipfire-arch-dots-ocr/src/arch_model.rs:24-26", "verified": true, "summary": "ArchModel dots_ocr vs inventory/generate dots-ocr."},
    {"title": "maple arch_id map missing", "path_line": "crates/hipfire-generate/src/ar.rs:4608-4628", "verified": true, "summary": "arch_id 15 → unknown; inventory has maple."},
    {"title": "DFlash target_hidden gap warn-not-fail", "path_line": "crates/hipfire-runtime/src/dflash.rs:1083-1100", "verified": true, "summary": "Documents NaN poison; warns once and continues."},
    {"title": "Generic DFlash ignores SpecRequestConfig fields", "path_line": "crates/hipfire-runtime/src/dflash_generic.rs:1000-1015", "verified": true, "summary": "min_p/ngram ignored; temp/top_p/top_k only."}
  ],
  "missing": [
    {"title": "Load-time refusal for MoE-only passthrough qts", "path_line": "crates/hipfire-runtime/src/weight_backend.rs:317-470", "verified": true, "summary": "No dense_gemv capability check at decode_raw_codec."},
    {"title": "MFP2/3 not in for_gemv_prerotated", "path_line": "crates/hipfire-dispatch/src/types.rs:730-776", "verified": true, "summary": "FwhtG256 but no prerotated arm; arch-loaded only."},
    {"title": "weight_gemm incomplete batched coverage", "path_line": "crates/hipfire-runtime/src/llama.rs:1475-1575", "verified": true, "summary": "Many dtypes fall back to per-row GEMV."},
    {"title": "CI arch_key ↔ inventory lock", "path_line": "crates/hipfire-runtime/src/reset_core.rs:229-244", "verified": true, "summary": "dots_ocr drift proves comment-only contract."},
    {"title": "Cross-loop post-terminal emit audit", "path_line": "crates/hipfire-runtime/src/eos_filter.rs:186-189", "verified": false, "summary": "Filter OK; daemon bypass paths not fully traced here."},
    {"title": "Universal free_all on WeightTensor owners", "path_line": "crates/hipfire-runtime/src/llama.rs:525-540", "verified": true, "summary": "free_all exists; not all free_gpu call sites use it."}
  ],
  "changes": [
    {"title": "LlamaWeights free_all cutover", "path_line": "crates/hipfire-runtime/src/llama.rs:683", "cost": "hours", "summary": "Match dflash; AWQ unload smoke."},
    {"title": "weight_gemv use dispatch_ref", "path_line": "crates/hipfire-runtime/src/llama.rs:740", "cost": "hours", "summary": "One WeightRef construction path."},
    {"title": "Refuse GL as dense weights", "path_line": "crates/hipfire-runtime/src/weight_backend.rs:411", "cost": "hours", "summary": "Fail at load or tag moe-only."},
    {"title": "Unify qt1/2/16 load paths", "path_line": "crates/hipfire-runtime/src/hfq.rs:1456", "cost": "1 day", "summary": "Keep F16; accept F32/BF16 on HFQ."},
    {"title": "arch_key spelling + maple map", "path_line": "crates/hipfire-runtime/src/reset_core.rs:140", "cost": "hours", "summary": "Single string SoT + CI."},
    {"title": "Hard-fail target_hidden gaps", "path_line": "crates/hipfire-runtime/src/dflash.rs:1083", "cost": "1 day", "summary": "Stop silent poison."},
    {"title": "Generic DFlash SpecRequestConfig parity", "path_line": "crates/hipfire-runtime/src/dflash_generic.rs:1000", "cost": "hours", "summary": "Wire or refuse."},
    {"title": "Quant capability matrix in registry check", "path_line": "docs/quant-formats/qt-register.txt", "cost": "days", "summary": "Loader×engine matrix gate."}
  ],
  "report": "inline-in-architecture-field"
}
```
