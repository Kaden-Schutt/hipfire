# hipfire Architectural Review — 2026-07-03

**Scope:** the full 65-crate Cargo workspace (~500k LOC of Rust; `kernels/`, `scripts/`, `benchmarks/` only where referenced by crate code).
**Method:** 22 subsystem reviewers in two rounds (18 primary scopes + 4 follow-up scopes chosen by a completeness critic), each producing structured findings; **every finding was independently re-checked by an adversarial verifier that re-read the cited code**. Of 151 primary findings, 129 were confirmed as written, 21 adjusted (location/numbers/severity corrected), and 1 refuted and dropped.
**Deliverables:**
- This report (consolidated assessment + roadmap)
- [`architectural-review-2026-07-03-catalog.md`](architectural-review-2026-07-03-catalog.md) — all findings, full text, with per-finding verification verdicts
- [`architectural-review-2026-07-03-file-index.md`](architectural-review-2026-07-03-file-index.md) — source-file index with descriptions, all crates

**Explicitly out of scope:** RDNA2/3/4 portability audit, security penetration testing, and GPU-execution behavioral verification (no GPU runs were performed; testability was judged statically). An input-validation review of the network-facing SD API was included in the follow-up round.

---

## 1. Executive summary

The macro-architecture of hipfire is in better shape than its file-level statistics suggest. Layering is broadly one-directional (leaf crates → compute → runtime → arch crates → serving → binaries), the arch crates avoid cycles with the runtime via a deliberate dev-dependency pattern, the VL crates (`gemma3-vl`, `dots-ocr`) genuinely reuse their base-arch crates instead of forking them, `hipfire-dispatch` vs `hipfire-rdna::dispatch` turns out to be deliberate layering rather than a naming accident, and several crates (`hipfire-detect`, `hipfire-eval`, `hipfire-evidence`, `hipfire-sampler`-adjacent code, the tokenizer) show genuinely strong test discipline. Prior dedup passes clearly happened: the runtime's `gguf.rs`/`tokenizer.rs` are thin re-export shims, not copies.

The risk is concentrated, not diffuse, in four patterns:

1. **A handful of monster compilation units carry the core product.** `qwen35.rs` (32,648 LOC, containing single functions of 5,571 and 3,006 lines), `hipfire-quantize/src/main.rs` (13,875 LOC with a ~5,300-line `main()`), `hipfire-diffusion/src/lib.rs` (10,349 LOC spanning ~9 responsibilities), `deepseek4/forward.rs` (9,202 LOC), `sdapi.rs` (8,249 LOC), `hipfire-eval/src/lib.rs` (8,117 LOC), and `hipfire-daemon/src/main.rs` (6,434 LOC, bin-only crate, ~3,937-line `main()`). These aren't just style problems: they interleave GPU dispatch, env-var policy, and pure math so that the pure logic can't be unit-tested, and they force serial ownership of the hottest files in the repo.

2. **Dual-maintenance surfaces that must be kept in sync by hand.** The most acute is in `hipfire-rdna`: every one of ~207 kernel-launch sites maintains *two* hand-written, order-sensitive copies of its kernel argument list with no check they agree — a silent-corruption hazard, not a compile error. The same shape recurs at every level: decode-vs-prefill parallel function pairs in the arch crates, single-vs-multi-GPU copies of the qwen35 layer loop, a wire protocol typed once in `hipfire-daemon-protocol` but hand-parsed from `serde_json::Value` in the daemon that never depends on it, GGML dequant codecs byte-identical across `hipfire-runtime` and `hipfire-quantize`, and quant block geometry (MQ4=136, OQ4=130) re-hardcoded in 10+ sites across 3+ crates instead of living in `hipfire-quant-format`.

3. **Test coverage is inverted relative to risk.** The best-tested code is leaf/pure utility code; the least-tested is the code most likely to corrupt output silently: quant codec integer math (`codecs.rs`, 2,243 LOC, 0 tests), KV-cache index arithmetic (`kv.rs`, 2,596 LOC, 0 tests), all kernel-selection routing in the dispatch family (0 tests across 7 files), and `hipfire-serving-core` (20,142 LOC, 11 tests). Meanwhile a 42,292-LOC `examples/` tree (148 files) functions as the de-facto QA suite, driven by `hipfire-eval` shelling out to example binaries — and unit tests for `hipfire-generate`'s pure contract logic live, inexplicably, inside `hipfire-daemon`'s binary.

4. **God objects at each layer boundary.** `Gpu` (~50 fields, ~9 responsibilities, ~800 dispatch methods across 24 `impl` files), `LoadedModel` (57 pub fields, 41 `Option`s — the direct cause of the `model.rs ↔ session.rs` import cycle), `DeepseekV4State` (54 lazily-allocated fields read through ~180 `as_ref().unwrap()`s), and `KvCache`'s 9 parallel booleans that metastasized into 49 constructors sharing a 33×-duplicated struct literal.

None of the confirmed findings is an active correctness bug today (the one candidate — an apparent FFI buffer overflow — was refuted under verification; see §7). The debt is structural, and the highest-leverage fixes are mechanical: single-source the kernarg lists, extract pure routing/planning logic behind data-in/data-out functions, adopt the typed protocol in the daemon, and give the codec/KV math the round-trip tests it was clearly designed to support.

**Consolidated severity counts** (after collapsing multi-category findings against the same artifact — raw counts in the catalog are inflated by design since reviewers scored per-dimension): roughly **25 distinct High-severity artifacts/hazards, ~55 Medium, ~40 Low** across 155 verified findings from both rounds.

---

## 1a. Status log — post-review remediation (updated 2026-07-04)

Progress on the §3 High findings since the review was written. Verified against the current tree (line numbers in §3 are as-reviewed and have drifted). Commits are on `chaingun`.

| # | Finding | Status | Notes |
|---|---|---|---|
| 3.1 | Dual kernarg lists (~207 sites) | ✅ Resolved (code) | **Done (2026-07-06)**: single-sourced **all ~400 `launch_maybe_blob` call sites** across all 24 `dispatch/` files onto `launch_kernargs(&kernargs![...])`, which derives both launch ABIs (capture-path `KernargBlob` + `kernelParams` array) from one `&[KernArg]` list — the two-list-must-agree hazard is gone by construction. Shared-across-arm sites use one `let args = kernargs![...]` referenced by each arm. Each site's two old lists were cross-checked before collapse — **0 disagreements found** (no latent corruption was hiding in the parallel lists). The now-unused `fn launch_maybe_blob` helper was deleted. `cargo check -p hipfire-rdna` clean, **0 warnings**; net **−8.8k LOC**. **Remaining before land**: GPU coherence gate (`coherence-gate-dflash.sh`) not yet run — code-complete, not yet GPU-validated. |
| 3.2 | Kernel-selection cascades, 0 tests, dead duplicate | ✅ Resolved | Dead duplicate selector deleted; `kernels.rs` now has 26 tests. |
| 3.3 | `Gpu` god object; arch dispatch in generic compute crate | 🔴 Open | 13 hipfire-rdna files still name specific arch families. |
| 3.4 | `qwen35.rs` 32k-line file, 5.5k-line fn | 🔴 Open | 32,618 LOC; `forward_prefill_chunk` still 5,577 lines. Crate is really **58,036 LOC / 5 files** (`speculative.rs` 12,730 was missed) and **~65–70% generic**, not qwen-specific — see the not-in-original-review addendum below. Multi-session. |
| 3.5 | deepseek4 `forward.rs` 9.2k LOC, 54-field god-state | 🔴 Open | Unchanged. Multi-session. |
| 3.6 | Cross-arch copy-paste | 🔴 Open | No shared-arch crate; partial folding via the capability layer. |
| 3.7 | Serving hand-parsed wire protocol / big main() | 🟡 Partial | **Protocol-drift half done (2026-07-06)**: `hipfire-daemon` now depends on `hipfire-daemon-protocol` and dispatches via an **exhaustive `match DaemonRequest`** instead of a `match msg_type` string switch — adding a protocol variant is now a daemon compile error until it is handled, so the adapter↔daemon drift cannot recur. The typed contract was first *completed* (the daemon accepted ~36 request types but only 24 were typed; the other 14 — batch prefill/decode, prefix-hash preflight, session-state reserve/describe/release, worker status, unload_worker, pflash_labels, train_drafter, diag, bench_prefill, profile — were added to `DaemonRequest`, with the validation-heavy batch/state ops as routing-only markers that keep their authoritative `validate_*`/`parse_*` parsers). The 35 handler bodies are byte-identical (they still read the `msg` Value); only arm patterns changed. Also fixed **two malformed-JSON error emitters** (the invalid-JSON and unknown-type branches raw-interpolated serde/error text into a JSON string literal) by routing them through the serde_json-built `emit_error_with_id`. `cargo check` + clippy clean; protocol round-trip tests green; no-gpu-ci Rust steps green. **Remaining before land**: daemon-runtime (live JSONL) validation not yet run — code-complete, not yet runtime-exercised (see pending-validation note below). **Still open (separate sub-findings)**: `sdapi.rs` big `main()`, `LoadedModel` 57-field Option-soup + `model.rs↔session.rs` cycle, and the half-adopted `SessionServingBackend` `qwen35_*`/`lfm2_*` pairs. Multi-session. |
| 3.8 | diffusion 10k-line grab-bag + coexistence violation | 🟡 Partial | **Part 1 done**: import/pickle/zip tooling moved to new leaf crate `hipfire-diffusion-coexist`, out of the server dependency graph (verified via `cargo tree`). **Part 2 in progress** (2026-07-04): `lib.rs` split into cohesive modules — `metadata` / `config` / `batch` / `cpu_ops` extracted (8,442 → 7,251 LOC; 221 tests green at each step), then the pure CPU-reference ops + `CpuTensor` consolidated *out* of the crate into the `hipfire-cpu` backend crate (see below). **Open**: the ~1,900-line `DiffusionPipeline` god-impl + the remaining runtime-context / CLIP / io clusters; `tests.rs` still one 13.5k-line file. |
| 3.9 | quantize 5.3k-line main(), format geometry, codec copies | 🟡 Partial | **Done**: block geometry single-sourced into `hipfire-quant-format` (WP-3.3, consumed by all arch loaders); GGML/GGUF codec de-duplicated into leaf `hipfire-gguf`; GGUF import pipeline + HFQ writer extracted to the quantize library (11 lib modules); codec round-trip/edge tests added. **Open**: `main()` is still ~5,414 lines (the `parse_args→Recipe`/`run_pipeline` decomposition). |
| 3.10 | `KvCache` 9 booleans → 49 constructors, 0 tests | 🟡 Partial | **Done**: index-math tests (WP-3.2); typed `KvQuantMode` enum + tested pure flag-derivation + `KvCache::quant_mode()`. **Open**: the boolean-fields → enum + 49-constructor → `KvCacheSpec` builder rewrite (~470 hot-path sites; needs GPU coherence validation — defer to a non-LDS-hazard box). |
| 3.11 | Layering inversions & workspace hygiene | 🟡 Partial | **(a) done**: runtime no longer deps `hipfire-eval` — `collect_default_host_profile` moved to leaf `hipfire-sysinfo`. **(b) done**: `GpuTensor`+`DType` extracted to leaf `hipfire-gpu-types` (re-exported from hipfire-rdna; all 25 consumers unchanged). **(c) done**: `[workspace.dependencies]` adopted. **(d) done**: members list deduped, no orphans (83 members = 83 crates). |
| 3.12 | Control-plane arch leakage: scheduler magic-string matching | ✅ Resolved (scheduler) | Scheduler classifies via the canonical `model_arch_family` table (`model_arch_family_from_str` + `ModelArchFamily` match), magic `arch_id` literals removed. The `hipfire-generate` qwen35 saturation (same finding) is still open. |
| 3.13 | Test placement & examples shadow QA suite | 🟡 Partial | **Done**: hsa-bridge 0→5 and hipfire-train 1→16 tests (WP-3.2). **Open**: 130 examples / 5 QA clones / eval-shells-out-to-examples; redline fate. |
| 3.14 | Dead compute in forward paths | ✅ Resolved | qwen2 double-compute dropped; arch-llama's dead 247-line forward deleted (now a thin facade over `runtime::llama`). |

**Legend:** ✅ resolved · 🟡 partial (scoped remainder noted) · 🔴 open. The remaining fully-open items (3.3, 3.4, 3.5, 3.6) and the deferred remainders (3.7-serving-`main()`/`LoadedModel`, 3.9-`main()`, 3.10-fields, and 3.8's `DiffusionPipeline` god-impl) are multi-session refactors and/or require GPU behavioral validation, not single-pass mechanical changes.

**Pending GPU-box validation (code-complete, not yet hardware/runtime-exercised).** Two landed changes are verified only statically (compile + clippy + no-gpu-ci Rust steps) and need a run on an appropriate box before they are considered fully landed:
- **3.1 kernarg single-sourcing** — run `coherence-gate-dflash.sh` (route through the LDS-safe `*_gfx1103` kernels so it can run on nix2) to confirm the collapsed launch ABIs are bit-identical in GPU execution.
- **3.7 daemon typed dispatch** — run a daemon JSONL smoke on a GPU box (load → generate → the extended ops: batch prefill/decode, session reserve/describe/release, worker status, diag/profile) to confirm no wire-shape regression and that the new strict deserialization accepts all real traffic. The intended behavior change is that malformed payloads the old lenient `.ok()`+fallback tolerated are now rejected at the deserialize boundary with a clear error.

**Architectural work beyond the findings (2026-07-04).** Adjacent to the 3.8/3.11 cleanups, three cross-cutting refactors landed:
- **Backend-crate naming symmetry.** Renamed `rdna-compute` → `hipfire-rdna` (package + `rdna_compute::` → `hipfire_rdna::` ident across 27 manifests and 1,538 refs) so the RDNA/HIP kernel crate matches its `hipfire-*` backend siblings (`hipfire-npu`, `hipfire-xdna`, `hipfire-cpu`, `hipfire-rocm`). Still RDNA-specific / HIP-direct — no generic cross-vendor layer (AGENTS.md).
- **CPU compute homed in the CPU backend.** The pure CPU-reference tensor ops + `CpuTensor` moved from `hipfire-diffusion` into `hipfire-cpu::tensor_ops` (with a crate-owned `CpuError`; `CpuTensor::from_hfq` → free fn `cpu_tensor_from_hfq`, ~113 call sites). Each backend crate now owns its own compute + tensor type.
- **CPU-oracle audit.** Swept all `*_cpu`/`*_reference` math: the diffusion ops were the one cohesive misplaced cluster; the rest is correctly arch-coupled (`forward_cpu`), production (`sample_cpu`), or test/example-local. Added a shared `hipfire_cpu::cpu_reference_gemm` and deduped the one real example oracle that used it.

**Monoliths beyond the original review (2026-07-04, not-in-original-review addendum).** A repo-wide monolith re-scan surfaced large files the original §3 findings did not name. Recorded here so they are tracked, not lost:
- **`hipfire-arch-qwen35/src/speculative.rs` — 12,730 LOC, missed entirely.** Finding 3.4 named only `qwen35.rs` (32,618) as the qwen monolith, but the crate is **58,036 LOC across 5 files** (`qwen35.rs` 32,618 + `speculative.rs` 12,730 + `mtp_spec.rs` 3,990 + `mtp_head.rs` 2,683 + `pflash.rs` 2,179). `speculative.rs` is the largest un-reviewed file in the repo.
- **`hipfire-rdna/src/dispatch/` cluster — ~55,548 LOC.** The dispatch mechanics that finding 3.1 (dual kernarg lists) and 3.3 (`Gpu` god object) touch at the symptom level, but the aggregate size of the dispatch tree itself was never scoped as a decomposition target.
- **`hipfire-serving-core` — `generate.rs` / `load.rs` / `generate_arch.rs`.** Finding 3.7 named `sdapi.rs` (the wire protocol) but not the serving-core generate/load drivers, which are their own large multi-responsibility files.

**Qwen-crate genericity (finding 3.4/3.6, quantified 2026-07-04).** Investigated whether the 58,036-LOC qwen35 crate is actually qwen-specific. **It is overwhelmingly not.** Evidence:
- Of 264 top-level fns in `qwen35.rs`, only **15** carry `qwen` in the name; the rest are generic `forward`/`moe`/`load`/`dense`/`gemm`/`ffn`/`prefill` (transformer path), `paro`/`rq`/`mq` (quant-format decode, shared across arches), and `validate`/`kld`/`trace` (eval). The file references generic runtime types (`LlamaConfig`/`WeightTensor`/`KvCache`/`ForwardScratch`) **168×** vs `Qwen35Config` **93×**, and imports its primitives from `hipfire_dispatch::*` and `hipfire_runtime::{hfq,kv,quant,weights,tp_shard,multi_gpu}`.
- **`speculative.rs` (12,730) + `mtp_spec.rs` (3,990) + `mtp_head.rs` (2,683) ≈ 19,400 LOC is a generic draft-verify speculative-decode engine** (dflash/ddtree, MTP, rollback/replay, verify-graph, seed-oracle), coupled to qwen only through a **32-reference** `Qwen35{Config,Weights,Scratch,Model}` seam — i.e. a thin `ModelSlot` boundary, not qwen logic.
- **`pflash.rs` (2,179) + the 21-type `DensePrefillSessionBatch*` cluster** in `qwen35.rs` is generic batched-prefill / pointer-table session machinery (0 qwen-named items in `pflash.rs`).
- The genuinely qwen3.5-specific core is the **DeltaNet hybrid linear-attention** path (`DeltaNet*` weights/state/rule, ~217 mentions), the `LayerType` full-attn/delta interleave schedule, and `Qwen35Config`/`Qwen35Weights` layout — on the order of ~12–18k LOC. **~65–70% of the crate is generic transformer/serving infrastructure misfiled under an arch crate**, confirming the review's premise for 3.6 (no shared-arch crate) and making the spec-decode engine (~19.4k LOC) the single largest extraction candidate.

---

## 2. Architecture map (as-reviewed)

```
                    ┌───────────────────────────────── binaries ─────────────────────────────────┐
                    │ hipfire-cli  hipfire-tui  hipfire-daemon(bin-only)  hipfire-server  quantize │
                    └────────┬───────────────────────────┬───────────────────────┬───────────────┘
                             │                           │                       │
             ┌───────────────▼──────────┐   ┌────────────▼──────────┐   offline tooling:
             │ hipfire-serving-core     │   │ daemon-adapter        │   hipfire-quantize, -train,
             │ (load/generate/session)  │   │ daemon-protocol       │   -coexistence, -kld, -atlas
             └───────────────┬──────────┘   └───────────────────────┘
                             │
   ┌─────────────────────────▼───────────────────────────┐
   │ arch crates: qwen35, deepseek4, lfm2moe, minimax,   │◄─ hipfire-dispatch (family/quant resolver)
   │ zaya, nemotron, qwen2, gemma3(+vl), llama, dots-ocr │
   └─────────────────────────┬───────────────────────────┘
                             │
             ┌───────────────▼──────────┐    ┌──────────────────────────────┐
             │ hipfire-runtime (hot     │    │ control-plane libs: model,   │
             │ path: hfq/kv/llama/arch) │    │ generate, prompt, state,     │
             └───────────────┬──────────┘    │ scheduler, config, detect    │
                             │               └──────────────────────────────┘
             ┌───────────────▼──────────┐
             │ hipfire-rdna (Gpu, ~800  │
             │ dispatch fns, kernels)   │
             └───────────────┬──────────┘
                             │
                  hip-bridge / hsa-bridge (dlopen FFI floor)
```

Layering violations found (details in §3): `hipfire-runtime → hipfire-eval → hipfire-daemon-adapter/tokio` (hot path pulls the eval harness for one function); arch-specific dispatch (`deepseek4.rs`, `zaya_cca.rs`, `mamba2.rs`) living inside generic `hipfire-rdna`; qwen35 batch protocol saturating generic `hipfire-generate` (218 references); offline import tooling inside `hipfire-diffusion`, which `hipfire-server` links.

---

## 3. Top findings (consolidated per artifact)

Each item below consolidates all confirmed findings against one artifact; full per-finding text, evidence, and verification notes are in the catalog.

### 3.1 [High] `hipfire-rdna` dispatch: dual kernarg lists at ~207 launch sites
`crates/hipfire-rdna/src/dispatch/gemm_hfq.rs:1891-1935` (representative)
Every kernel launch builds a `Vec<*mut c_void>` **and** a duplicate `KernargBlob` closure that re-pushes the identical arguments. Both must match each other and the HIP kernel signature exactly; nothing enforces it, and a drift produces silent kernarg corruption. ~207 `launch_maybe_blob` sites, 2,454 `as *mut c_void` casts across 7 files; `gemv.rs` alone has 476 `push_*` lines.
**Fix:** one declaration that emits both representations — a `kernargs![a_ptr, x_ptr, m:i32, k:i32]` macro, or fold blob-building into `launch_maybe_blob(&[KernArg])` and derive the pointer vec internally. Mechanical, ~20 lines saved per dispatch fn, eliminates the hazard class.

### 3.2 [High] Kernel-selection cascades: zero tests, one dead duplicate selector
`crates/hipfire-rdna/src/dispatch/` — 7 files, 0 `#[test]`; `gemm_qkv.rs:844-914`
The routing logic ("gfx1103 + batch 32 + hfq4 → which kernel?") is the highest-consequence pure-ish logic in the repo and is untested; the leaf predicates in `arch_caps.rs` are pure and well-tested (19 tests), but the composite cascades call `bind_thread_or_warn()`/`mmq_screen_weight()` mid-decision so they can't run without a GPU. A dead `pub fn gemm_qkvza_hfq4g256_route_label` (zero callers) duplicates one real cascade and must be manually kept in sync.
**Fix:** extract each cascade into a pure `fn choose_*_kernel(caps, flags, shapes, screen: ScreenOutcome) -> KernelChoice` taking GPU-derived facts as values; table-test the full arch×shape×format matrix on CI (this is also the cheapest way to make the RDNA2/3/4 portability invariant testable). Delete the dead selector.

### 3.3 [High] `Gpu` god object; model-specific dispatch inside the generic compute crate
`crates/hipfire-rdna/src/dispatch/mod.rs:532-769`; `dispatch/deepseek4.rs` (3,536 LOC, 65 methods)
`Gpu` carries ~50 fields spanning ≥9 responsibilities (runtime+caps, JIT caches, memory pool, calibration hooks, quant scratch, conversion scratch, MMQ screening cache, three hipGraph capture subsystems), with ~800 dispatch methods across 24 `impl Gpu` files. DeepSeek-V4-private concepts (`hc_sinkhorn_4x4`, `indexer_top_k`, NSA compressor), plus `zaya_cca.rs` and `mamba2.rs`, live unconditionally inside the "generic RDNA dispatch" crate despite dedicated arch crates existing.
**Fix:** split `Gpu` state into owned sub-structs (`JitCache`, `QuantScratch`, `CaptureState`, …) held by a slim `Gpu`; move arch-private dispatch either into the arch crates (via an extension-trait over `Gpu`) or behind feature gates as an interim step.

### 3.4 [High] `qwen35.rs` — 32,648-line file with 5,571-line functions
`crates/hipfire-arch-qwen35/src/qwen35.rs`
70 pub fns, 53 structs, 18 impl blocks, ≥8 responsibilities in one flat file. God functions: `forward_prefill_chunk` (~5,571 LOC), `forward_scratch_layers` (~3,006), `forward_scratch_layers_multi` (~1,730), `prefill_moe_ffn_body_batched` (~1,592); in `speculative.rs`, `spec_step_dflash` (~3,278). The single- and multi-GPU decode loops are parallel copies (fix-in-two-places; the multi copy already carries hazards the single copy doesn't). ~60 scattered `HIPFIRE_*` env toggles read inline; 29 `too_many_arguments` allows. Bright spot: the pure plan/contract logic is well-factored and well-tested.
**Fix:** promote to `src/qwen35/` module tree (config / weights / state / prefill / decode / moe / env-policy); extract per-layer bodies shared by the single/multi loops into one parameterized implementation; centralize env reads into one `Q35Policy` struct constructed once.

### 3.5 [High] deepseek4: 9,202-line `forward.rs`, 54-field lazy god-state, decode/prefill copy-pairs
`crates/hipfire-arch-deepseek4/src/forward.rs`; `deepseek4.rs::DeepseekV4State`
85 functions mixing decode, batched prefill, MTP heads, expert-parallel forward, MoE routing math, and RoPE/YaRN math. The state struct's 54 lazily-allocated fields are read through ~180 `as_ref().unwrap()` calls — ordering bugs panic at runtime instead of failing to compile. Per-layer algorithms are duplicated between decode and prefill variants (`q_lora` vs `q_lora_batched`: same 6-step sequence).
**Fix:** split by responsibility; replace lazy-Option state with typed phases (e.g. `State<Uninit> → State<Ready>` or grouped sub-structs allocated together); unify decode/prefill bodies over a `TokenView` (single vs batched) parameter.

### 3.6 [High] Cross-arch copy-paste the codebase itself has already flagged
16 `TODO(transformer-extraction)` markers + "Mirrors …"/"Replicated from …" comments
Per-arch config structs, HFQ-metadata parsers, and low-level weight/quant helpers (`sext4`, `dequant_hfq4`, config parsing) are copy-pasted across 4+ arch crates with cosmetic drift; the promised shared home (`hipfire_runtime::transformer`) exists but holds predicates, not the helpers. The `Architecture` trait (`hipfire-runtime/src/arch.rs:85`) is never used polymorphically (no `T: Architecture` or `dyn Architecture` anywhere) — it costs boilerplate without buying dispatch. A fix to one copy of a dequant helper silently diverges from the other three; near-zero unit tests exist over these pure functions.
**Fix:** actually populate the extraction target (either `hipfire_runtime::transformer` or a new `hipfire-arch-common` leaf crate) with the shared parsers/helpers + their tests; reconsider whether `Architecture` should remain a trait or become a convention.

### 3.7 [High] Serving: hand-parsed wire protocol, 3,937-line `main()`, `LoadedModel` Option-soup, duplicated per-arch session surface
`hipfire-daemon/src/main.rs`; `hipfire-serving-core/src/{model.rs,session.rs}`
The protocol is typed once (`hipfire-daemon-protocol`, serde enums, used by adapter/eval/steer-harness) — but the daemon itself never depends on it and hand-parses `serde_json::Value` with string-literal matching: protocol drift is a live correctness risk, and the daemon is a bin-only crate so none of its logic is unit-testable. `LoadedModel` has 57 pub fields (41 `Option`) with per-arch parallel slots, directly causing the `model.rs ↔ session.rs` cycle; session ops are ~13 hand-duplicated `qwen35_*`/`lfm2_*` function pairs although a `ServingBackend`/`SessionServingBackend` abstraction already exists, half-adopted. 63 tests for *other crates'* functions live inline in the daemon bin; the serving hot path itself has zero in-file tests.
**Fix:** make the daemon depend on `daemon-protocol` and deserialize into the typed enums (one-way door, mechanical); split daemon into lib+bin so handlers are testable; finish `SessionServingBackend` adoption to collapse the `qwen35_*`/`lfm2_*` pairs; break `LoadedModel` into per-arch resident structs behind one enum or trait object; relocate the misplaced tests next to their subjects (`hipfire-generate`, `hipfire-state`).

### 3.8 [High] `hipfire-diffusion/src/lib.rs`: 10,349-line grab-bag + coexistence-invariant violation
`crates/hipfire-diffusion/src/lib.rs` (import block at L8444-10349)
One file holds HFQ metadata, primitive CPU tensor ops, the CLIP encoder, a 27-method / ~1,890-line pipeline god-impl, JSON/shape helpers, and — the boundary violation — ~1,900 lines of offline import tooling: `import_diffusers_to_hfq`, single-file checkpoint import, safetensors state-dict parsing, a hand-rolled PyTorch **pickle interpreter** and a from-scratch **zip reader**. AGENTS.md mandates conversion tooling live in `hipfire-coexistence` (or a dedicated tooling crate), *not* in a crate the server links — and `hipfire-server` links this one, so untrusted-format parsing ships in the serving binary. Also: a 213-copy error-map closure, ~445 hand-written CPU/GPU dispatch wrappers, and a single 13,491-line `tests.rs`. (Credit where due: the past channels_last/stride loader bug is now pinned by two pure-CPU regression tests, and kernel dispatch correctly reuses `hipfire-rdna` rather than reinventing it. Note `hipfire-quantize`'s GGUF import is *compliant* — quantize is a dedicated offline tooling crate.)
**Fix:** move the import/pickle/zip block to `hipfire-coexistence` (it already owns LoRA export/merge — this is its charter); split lib.rs into `metadata / clip / pipeline / cpu_ops / io` modules; generate the dispatch wrappers with a macro; split tests.rs per module.

### 3.9 [High] Quantization: 5,300-line `main()`, format geometry owned by no one, cross-crate codec copies
`hipfire-quantize/src/main.rs:5444-10743`; `hipfire-quant-format` (193 LOC)
`main()` inlines CLI parsing, format-recipe normalization, calibration loading, and the per-tensor pipeline. `hipfire-quant-format` — the crate whose whole purpose is format identity — owns only the 1-byte QuantType code, while block geometry is re-hardcoded across ≥10 sites in 3+ crates (MQ4 block=136: codecs.rs ×4, gptq.rs, three bins, runtime examples; OQ4 block=130 similarly). GGML dequant codecs are byte-identical copies between `hipfire-runtime/src/quant.rs` and `hipfire-quantize/src/gguf_input.rs`. The pure codecs (`codecs.rs`, 2,243 LOC, 40 pub fns) have **zero** direct tests despite encode/decode pairs sitting in the same module (round-trips are trivial to assert); a golden byte-stability battery exists but doesn't cover edge cases. The "pure" codecs also read a process-global toggle.
**Fix:** move block geometry constants + layout math into `hipfire-quant-format` and make everyone consume it; extract `main()` into lib functions (`parse_args → Recipe`, `run_pipeline(Recipe)`); single-source the GGML dequant (runtime already re-exports other loaders from hipfire-model — same pattern applies); add codec round-trip + edge-case tests (degenerate groups, saturation, NaN scale).

### 3.10 [High] `KvCache`: 9 boolean modes → 49 constructors → 33 copies of the struct literal; 0 tests
`crates/hipfire-runtime/src/kv.rs`
Quant mode is 9 parallel booleans instead of an enum, expanded combinatorially into 49 `new_gpu*` constructors each ending in a near-identical ~25-field literal (33 occurrences). 2,596 LOC of hot-path index/size arithmetic (`kv_dim`, packed-element math `(x+3)/4`, block-table offsets) with zero tests, though several helpers are pure.
**Fix:** `enum KvQuantMode { Q8, Int8, Hfq4, Asym4, Asym3, Asym2, Fwht, Kvarn }` + a `KvCacheSpec` builder; one constructor; unit tests over the index math (pure, no GPU needed).

### 3.11 [High] Layering inversions & workspace hygiene
`hipfire-runtime/Cargo.toml:61`; root `Cargo.toml`
(a) The inference hot path depends on `hipfire-eval` (21k-LOC evidence harness, pulling tokio-process via daemon-adapter) for exactly one symbol (`collect_default_host_profile`). (b) `GpuTensor`/`HipResult` (1,323/1,527 graph edges) live inside the 63.5k-LOC `hipfire-rdna`, so ~20 crates depend on the whole compute crate to name a tensor type. (c) No `[workspace.dependencies]`: serde declared in 33 crates, serde_json in 42, with already-diverging feature sets. (d) `crates/hipfire-primitives` appears twice in `members`; one orphan crate dir isn't listed.
**Fix:** move `host_profile` collection into `hipfire-sysinfo` (or invert: eval depends on runtime); extract `GpuTensor`/`HipResult`/`DiffusionResult` into a leaf `hipfire-gpu-types` crate (or into the existing zero-dep `hipfire-primitives`); adopt `[workspace.dependencies]`; dedupe the members list.

### 3.12 [High] Control-plane arch leakage: `hipfire-generate` (218 qwen35 refs) and stringly-typed scheduler
`hipfire-generate/src/lib.rs`; `hipfire-scheduler/src/lib.rs:575-671`
The generic generation-protocol crate is saturated with one architecture's batch protocol (43 qwen35-named fn/type declarations). The scheduler classifies arch families by matching `arch_id` against magic strings (`"5" | "6"`, `"10" | "11" | "14"`) and substring-matching artifact paths — bypassing the canonical arch-family table and silently misclassifying anything new.
**Fix:** move qwen35 batch contracts into the qwen35 crate (or an arch-protocols crate); give the scheduler a typed `ArchFamily` enum sourced from the canonical table, with an exhaustive match.

### 3.13 [High] Test placement & the examples/ shadow QA suite
`crates/hipfire-runtime/examples/` (148 files, 42,292 LOC); `hipfire-eval/src/executor_examples.rs`
31 `test_*`, 16 `bench_*`, 11 `profile_*` example binaries, including 6 `*QA.rs` clones of their non-QA siblings; `hipfire-eval` drives them by shelling out to example binaries — contradicting the AGENTS.md rule that admission evidence lives in eval batteries, and leaving the "tests" unbuildable by `cargo test`. Worst LOC-per-test crates: `hipfire-serving-core` (20,142/11), `hipfire-train` (11,314 LOC, exactly 1 `#[test]`, with NaN-panic-prone pure fns), `redline` (5,660/0, unsafe KMD driver, unconsumed — its own docs recommend abandoning), `hsa-bridge` (2,988/0 incl. byte-layout-critical pure AQL packet-building).
**Fix:** triage examples into (i) real usage samples (keep), (ii) eval batteries (move into `hipfire-eval` executors), (iii) integration tests (`tests/` dirs, no GPU where possible); delete the QA clones after migration. Add `#[test]`s for hsa-bridge's pure packet/header builders and train's math. Decide redline's fate explicitly (archive to `third_party/` or delete; its docs already recommend abandonment).

### 3.14 [Medium→High cluster] Dead compute and dead code in forward paths
`hipfire-arch-qwen2/src/qwen2.rs:1010-1234`; `hipfire-arch-llama/src/arch.rs:143-390`
qwen2's hand forward path computes QKV and FFN gate/up **twice** — Block A's results are overwritten unread by the lowered steps (100% dead GPU compute on that path); arch-llama ships a 247-line dead reimplementation of the runtime forward. Both are leftovers of half-finished pipeline migrations.
**Fix:** delete Block A after a GPU parity check; delete the dead llama forward (the crate is a thin facade over `hipfire-runtime/src/llama.rs`, which is the real implementation — rename/document accordingly).

---

## 4. Test coverage snapshot

| Crate | LOC | `#[cfg(test)]` files | Assessment |
|---|---|---|---|
| hipfire-rdna | 92,859 | 6/176 | Routing cascades 0 tests; arch_caps predicates well-tested (19) |
| hipfire-runtime | 73,209 | 18/189 | sampler (17), tool_call (14), eos_filter (12) good; **kv.rs 0, hfq.rs 1** |
| hipfire-arch-qwen35 | 59,509 | 6/17 | Pure contract logic well-tested; forward paths GPU-bound |
| hipfire-diffusion | 38,226 | 4/13 | 13.5k-line tests.rs monolith; stride regression pinned ✔ |
| hipfire-quantize | 27,589 | 7/20 | gptq 24 / qtip 11 / hessian_io 7; **codecs.rs 0, gguf_input 0** |
| hipfire-serving-core | 20,142 | 3/18 | **Worst large-crate ratio: 11 tests**; hot path untested in-file |
| hipfire-eval | 21,195 | 2/16 | 104 tests, zero unwraps in prod code — healthy |
| hipfire-train | 11,314 | 1/71 | **1 test total**; NaN-panic-prone pure fns |
| redline | 5,660 | 0/24 | Unsafe, unconsumed, untested |
| hsa-bridge | 2,988 | 0/6 | Byte-layout AQL builders untested |
| hipfire-model | 6,238 | 2/4 | tokenizer 58 tests; artifact-name grammar parser tested ✔ |
| hipfire-detect | 3,262 | 13/14 | Exemplary: one module per detector, ~64 tests |

Well-covered reference points worth imitating: `hipfire-detect` (module-per-detector, tests co-located), the `.hfq` artifact-name parser, the sampler's NaN edge-case tests, `hipfire-eval`'s zero-unwrap production code.

~7,600 `unwrap()` calls workspace-wide (307 in hipfire-server), but sampled parsers bounds-check before their `try_into().unwrap()` — the acute gap is missing tests around codec/KV/index math, not raw unwrap density.

---

## 5. Refactoring roadmap

Ordered by leverage-per-risk; each phase is independently shippable.

**Phase 0 — mechanical quick wins (days)**
1. Adopt `[workspace.dependencies]`; dedupe `hipfire-primitives` member entry. (§3.11)
2. Delete dead code: `gemm_qkvza_hfq4g256_route_label` (§3.2), qwen2 Block A after GPU parity check, arch-llama dead forward (§3.14).
3. Make `hipfire-daemon` depend on `hipfire-daemon-protocol` and deserialize typed requests (§3.7) — highest correctness-risk reduction per line changed in the repo.
4. Move `host_profile` off the runtime→eval dependency (§3.11).

**Phase 1 — kill the silent-corruption hazards (1-2 weeks)**
5. `kernargs!` single-source macro across all ~207 launch sites (§3.1).
6. Extract pure kernel-selection functions + arch×shape×format table tests (§3.2).
7. Codec round-trip/edge tests for `codecs.rs`; KV index-math tests; hsa-bridge packet-builder tests (§3.9, §3.10, §3.13).
8. Move quant block geometry into `hipfire-quant-format`; single-source GGML dequant (§3.9).

**Phase 2 — break the monoliths along verified seams (weeks, incremental)**
9. `qwen35.rs` → module tree; unify single/multi-GPU layer loops (§3.4).
10. `hipfire-quantize` `main()` → lib pipeline (§3.9); daemon `main()` → lib handlers (§3.7).
11. Diffusion lib.rs split + move importer/pickle/zip to `hipfire-coexistence` (§3.8).
12. `KvCache` mode enum + spec builder (§3.10). `LoadedModel` per-arch residents; finish `SessionServingBackend` adoption, dissolving the model↔session cycle (§3.7).
13. deepseek4 `forward.rs` split + typed state phases; move deepseek4/zaya/mamba2 dispatch out of `hipfire-rdna` (§3.3, §3.5).

**Phase 3 — consolidation (ongoing)**
14. Populate the transformer-extraction target; collapse the 16 documented TODO copies (§3.6).
15. Extract `GpuTensor`/`HipResult` leaf crate; begin `Gpu` state decomposition (§3.3, §3.11).
16. Examples-tree triage into eval batteries / integration tests / true examples (§3.13).
17. De-qwen35 `hipfire-generate`; typed `ArchFamily` in scheduler (§3.12).

---

## 6. Follow-up round findings

The completeness critic flagged 19 crates and several lenses the primary round under-covered. Four were re-reviewed with the same reviewer→verifier method. Highlights:

**Two confirmed High input-validation issues on the network-facing SD API** (`hipfire-server/src/routes/sdapi.rs`). These are real robustness holes, mitigated today only by the default `127.0.0.1` bind (host is configurable to `0.0.0.0`):
- **Unbounded request geometry → memory/compute DoS.** `width`/`height`/`steps`/`batch_size`/`n_iter` are accepted with only `unwrap_or` defaults and `.max(1)` — no upper bound anywhere (`sdapi.rs:1711-1765`). A sub-2 MB body `{"width":100000,"height":100000}` flows straight into `Vec::with_capacity(batch*channels*height*width)` (`hipfire-diffusion/src/lib.rs:1145`), a hundreds-of-GB allocation; `100000 % 8 == 0` so it passes the only validation (positivity + VAE-scale divisibility). **Fix:** a `TryFrom<SdGenerationRequest>` validated newtype that rejects geometry above portability-safe caps (≤4096, multiple-of-8; steps ≤200) with a 400 at the extractor boundary.
- **Client-controlled output directory → arbitrary directory creation + write.** `sdapi_output_dir` (`sdapi.rs:1651-1685`) turns `override_settings.outdir_*` into a `PathBuf` with no traversal check; `save_sdapi_images_with_kind` then `create_dir_all`s it and writes PNG bytes there. Opt-in (`save_images:true`), content is PNG-magic-constrained, but the destination is arbitrary. **Fix:** treat `outdir_*` as admin-only config; for network requests, canonicalize and require the path to stay within a server-owned output root.
- Two Mediums: model-name lookup (`find_model_in`, `hipfire-model/src/lib.rs:1125`) accepts absolute paths and `../` traversal (bounded by a subsequent `inspect_hfq().is_ok()` gate); and `/sdapi/*` is entirely unauthenticated (only `/admin/*` is gated) with no rate limiting. Credit noted: numeric fields are typed `Option<u32>`/`Option<i64>` so malformed numbers yield 422 not panic, errors funnel through proper status codes, and several `checked_mul` guards exist.

**Independent confirmation of the daemon protocol-drift risk (§3.7), with the direction corrected.** The critic suspected `hipfire-daemon-adapter` was a *third* hand-maintained copy of the wire protocol; verification **refuted** that — the adapter correctly consumes the typed `hipfire-daemon-protocol` enums for both serialize and deserialize. The real second copy is the **daemon server**: `hipfire-daemon/src/main.rs` never references the typed enums (zero mentions), hand-parsing `serde_json::Value` across ~34 `"type"`-string match arms and hand-building responses with `json!`. Adding a protocol variant updates the adapter automatically but silently leaves the daemon parser stale. There is also a latent bug: the raw-string error emitter produces malformed JSON when the serde error text contains a quote or newline. The adapter itself carries Medium-grade wrapper duplication (byte-identical `expect_steer_ok`/`expect_lora_ok`; 8 methods sharing one drain skeleton), untested `steer_*`/`lora_*` families despite a ready `MockTransport`, and mixes client + daemon-startup-lease responsibilities in one 1,387-line file. Its lock code is **compliant** with the `hipfire-lock` invariant (uses `FlockGuard`, no sentinels).

**Lock-discipline audit surfaced a stale legacy path, not a live break.** Two shell gates reference `/tmp/hipfire-gpu.lock` (`scripts/serve-restart.sh:15` deletes it; `tests/pp-gate.sh:503` tests its existence as a held/free signal — sentinel semantics). Verification found the current GPU flock lives at `~/.hipfire/locks/hip-gpu-0.lock`; **no Rust code flocks `/tmp/hipfire-gpu.lock`** — it survives only in stale docs (`hipfire-lock/AGENTS.md`, `hipfire-daemon/AGENTS.md`) and these scripts. So no mutual-exclusion break occurs, but `pp-gate.sh`'s parent-lock-detection guard is **effectively dead** (probes a path nothing writes), and the stale references violate AGENTS.md's "remove legacy-name fallback / update stale references" rule. **Fix:** point the gates at `hipfire lock status` (backed by `probe()->LockState`) and purge the `/tmp/hipfire-gpu.lock` references from scripts and docs.

**Accelerator/fallback crates are mostly healthy, with one mislabel.** `hipfire-xdna`, `hipfire-hneurons`, `hipfire-vision-cache` each have a clear single responsibility and good tests; `hipfire-npu` is a clean admission-policy layer delegating device access to xdna. The weak spot: `hipfire-cpu` is described as "deterministic CPU oracle backends" but ~95% of its 894 LOC is a Qwen3.5-specific backend-selection/module-evidence policy DSL (`'qwen35.layers.{}.mlp.swiglu_down'` string-building) — same arch-leakage pattern as §3.12, and its one real CPU compute path is nearly untested. `hipfire-kvquant` is healthier than the runtime KV baseline: it does **not** re-declare `kv.rs`'s 9-boolean mode soup and correctly delegates f16 conversion to `hipfire-primitives`; its issues are Low/Medium (untied FWHT seed literals that must agree across encode/decode, a `KVARN_GROUP=128` const duplicated in `hipfire-rdna` because the dependency direction blocks sharing, an unchecked pack-loop bound).

---

## 7. Review coverage, refuted findings, and limitations

**Adversarial verification results:** 151 primary + follow-up findings were each re-checked by an independent agent instructed to refute them. 1 was refuted and dropped: the claimed heap overflow in `hip-bridge`'s `get_arch` (`ffi.rs:1385-1411`, 1024-byte buffer vs 1472-byte `hipDeviceProp_t`) — the dlsym-loaded bare symbol resolves to the ELF versioned-default `hipGetDeviceProperties@@hip_4.2`, whose legacy ABI layout fits the buffer, so no overflow occurs. It remains a fragility worth a comment or an explicit `hipDeviceProp_tR0000`-sized buffer, but it is not a live bug.

**Known inflation:** raw severity counts multi-count monster files (diffusion lib.rs drew 4 High findings under different categories; qwen35.rs drew 3). §3 consolidates per artifact; use those numbers, not raw counts, for prioritization.

**Dimensions intentionally not covered** (flagged by the completeness critic, out of this review's structural charter): RDNA2/3/4 portability correctness of kernel selection (partially mitigated by §3.2's table-test recommendation), deep FFI soundness proofs, penetration-style security testing. A workspace-wide lock-discipline audit and an input-validation pass on `sdapi.rs` were added in the follow-up round (§6).

**Crates declared clean after spot-checks:** `hipfire-hash` (93 LOC), `hipfire-build-info`, `hipfire-arch-toy` (template), `hipfire-detect`, `hipfire-kld`, `hipfire-vision-cache`, `hipfire-cpu` (7 tests), `hipfire-lock`.

---

## Appendices

- **A. Findings catalog** — [`architectural-review-2026-07-03-catalog.md`](architectural-review-2026-07-03-catalog.md): all findings with full observation/recommendation/evidence text and per-finding verification verdicts, grouped by subsystem with per-subsystem assessments.
- **B. Source file index** — [`architectural-review-2026-07-03-file-index.md`](architectural-review-2026-07-03-file-index.md): every source file with LOC and a one-line responsibility description (large example/kernel-variant trees clustered).
