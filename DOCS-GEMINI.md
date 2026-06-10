# Hipfire Documentation Alignment Audit Report

This document compiles the findings of a comprehensive audit comparing the project's documentation files (located in the root directory and under `docs/`) with the actual implementation in the source code (primarily Rust and TypeScript under `crates/` and `cli/`).

Overall, the documentation is incredibly rich, detailed, and technically precise. However, as the codebase has rapidly evolved through recent versions (v0.1.9-alpha and v0.2.0-era branches), several significant discrepancies have emerged. These are detailed below, categorized by area.

---

## 1. High-Impact Discrepancies

### A. TriAttention Sidecar Filename Extension Conflict (`.triattn.hfq` vs `.triattn.bin`)
There is a structural naming convention conflict regarding the preferred file extension for TriAttention sidecar files.

*   **What the Documentation Says:**
    *   `docs/CLI.md` states: *"Generate a `.triattn.bin` sidecar for the given model. ... `my-finetune.mq4` → `my-finetune.mq4.triattn.bin`."*
    *   `docs/CONFIG.md` states: *"For models from HuggingFace, a published `.triattn.bin` ships alongside the weights ... For custom or quantized models, you must generate one... for example `my-finetune.mq4.triattn.bin`."*
    *   `docs/GETTING_STARTED.md` states: *"The daemon automatically discovers the shipped `.triattn.bin` beside the weights."*
*   **What the Code & Guidelines Say:**
    *   The project's official naming guidelines in `AGENTS.md` explicitly mandate: **"Use `.triattn.hfq` for TriAttention sidecars even though they are not weight tensors; do not introduce `.triattn.bin` for new files."**
    *   `cli/index.ts` implements this guideline exactly:
        *   The function `defaultRoleSidecarPath` (line 6944) returns files ending with `.triattn.hfq` (i.e., appending `.triattn.hfq` or replacing `.hfq` with `.triattn.hfq`).
        *   The `sidecar-gen` sub-command (line 9813, 9880) defaults the output sidecar path to `<model-stem>.triattn.hfq`.
*   **Code-Level Status:**
    *   The daemon's auto-discovery logic (in `cli/index.ts` lines 891–902) scans for entries starting with `<basename>.triattn` and ending in either `.hfq` or `.bin`.
    *   The model registry (`cli/registry.json`) references a published sidecar file using the `.bin` suffix: `"qwen3.6-27b.mq4.triattn.blended_v3.bin"`.
*   **Impact:** Although both formats load successfully due to the daemon's fallback logic, the user-facing documentation teaches users to use/expect `.triattn.bin` filenames, conflicting with both the new `AGENTS.md` naming standards and the CLI's default `sidecar-gen` output (`.triattn.hfq`).

---

### B. Outdated "No WMMA Prefill Path for MQ3" Statement
The documentation incorrectly states that the sub-4-bit Magnum Quant 3 (MQ3) format has no fast prefill path.

*   **What the Documentation Says:**
    *   `docs/QUANTIZATION.md` states: **"There is no WMMA prefill path for MQ3 or MQ2 yet, so prefill falls back to per-row GEMV until the kernel lands in a follow-up PR."**
*   **What the Code & Testing Playbook Say:**
    *   `AGENTS.md` (§ "What v0.1.9-alpha added") states: **"WMMA prefill family (`gemm_qkvza/qkv/gate_up/residual hfq3`) closing the 17× prefill gap that gated ship. Arch-gated to gfx11 wave32 WMMA. gfx12 K4 variant ships in the same release."**
    *   The GPU dispatch implementation (`crates/rdna-compute/src/dispatch.rs`) fully contains active WMMA prefill paths for `hfq3`/`mq3`:
        *   `gemm_qkvza_hfq3g256_wmma` (line 15107) and its variants:
            *   `gemm_qkvza_hfq3g256_wmma_mb4` (line 15231)
            *   `gemm_qkvza_hfq3g256_wmma_gfx12` (line 15377)
        *   `gemm_qkv_hfq3g256_wmma` (line 15682)
        *   `gemm_gate_up_hfq3g256_wmma` (line 16856)
        *   `gemm_hfq3g256_moe_grouped_wmma` (line 21286)
        *   `gemm_hfq3g256_residual_wmma` (line 23715)
*   **Impact:** The math/design documentation in `docs/QUANTIZATION.md` is outdated and underrepresents the actual capabilities of the engine, which fully supports WMMA prefill for MQ3 in production on RDNA3/4 (`gfx11` and `gfx12`) architectures.

---

### C. Massive Environment Variable Reference Drift (`docs/env-vars.md`)
Running the project's automated verification script (`./scripts/regen-env-vars-doc.sh`) reveals a significant drift between the source code references and the "canonical" table in `docs/env-vars.md`.

*   **Overall Metrics:**
    *   **Source Code:** 307 unique environment variables currently extracted.
    *   **Documented in `docs/env-vars.md`:** 141 environment variables.
*   **Undocumented Variables (166 total missing from the table):**
    *   Major new features are completely silent in the table. Examples include:
        *   **MTP Speculative Decode:** `HIPFIRE_MTP_MODE`, `HIPFIRE_MTP_K`, `HIPFIRE_MTP_GPU_ACCEPT`, `HIPFIRE_MTP_PROPOSAL_GRAPH`.
        *   **CASK/TriAttention Override:** `HIPFIRE_CASK_OFF`.
        *   **PFlash Prefill Compression:** `HIPFIRE_PFLASH_DEBUG`, `HIPFIRE_PFLASH_DRAFTER_KV`, `HIPFIRE_PFLASH_DRAFTER_STATE`.
        *   **Model-Specific Diagnostic Overrides:** `HIPFIRE_DEEPSEEK4_*` cluster (dozens of variables covering attention tuning, routing scale, MTP, and MoE grouped gates) and `HIPFIRE_LFM2_*` / `HIPFIRE_MINIMAX_*` clusters.
        *   **Arch/GPU Tuning Switches:** `HIPFIRE_GPU_SLAB_LOAD`, `HIPFIRE_GPU_SLAB_MIB`, `HIPFIRE_LOAD_TRANSPORT`, `HIPFIRE_HIP_WAIT`, `HIPFIRE_GFX942_*` (MI300X specific tweaks).
*   **Stale Variables (27 total documented but deleted from source):**
    *   Several deprecated or refactored variables remain in the doc table. Examples include:
        *   `BENCH_BATCH`, `BENCH_K`, `BENCH_M` (un-prefixed benchmarking configs).
        *   `DDTREE_TIMING`, `DEBUG_LAYERS`, `DFLASH_LIVE_TAU`, `FP32_STATE`.
        *   `HIPFIRE_GCN5_WAVE64_HYBRID`, `HIPFIRE_GEMV_DP4A`, `HIPFIRE_GEMV_PREFETCH`, `HIPFIRE_MMQ_MIN_BATCH`, `HIPFIRE_ROCBLAS_MIN_BATCH`.
        *   `MAX_TOKENS`, `NO_NGRAM`, `USE_SAMPLE`.
*   **Impact:** Contributors and operators cannot rely on `docs/env-vars.md` as a source of truth for all environment controls, and might waste effort tuning deprecated variables.

---

## 2. Medium & Minor Discrepancies

### A. Missing `/set` Slash Command in `docs/CHAT.md`
The interactive TUI chat application supports an extremely useful runtime slash command that is completely omitted from the documentation.

*   **What the Code Implements:**
    *   In `cli/chat.ts` (lines 276+), the `/set <key> <val>` command is defined:
        ```typescript
        case "set":
          // Adjusts session-specific parameters (temperature, top_p, max_tokens, etc.)
        ```
*   **What the Documentation Says:**
    *   The "Slash Commands" table in `docs/CHAT.md` lists `/help`, `/?`, `/clear`, `/stats`, `/trim`, `/exit`, `/quit`, but completely omits the `/set` command.
*   **Impact:** Users running `hipfire chat` miss out on the ability to adjust their parameters (like temperature or max tokens) on the fly during a live chat session without quitting and editing their config file.

### B. Mismatched Hugging Face User Endpoint in `AGENTS.md`
`AGENTS.md` § "2.D" lists Hugging Face draft model endpoints under the account `schuttdev` (e.g., `schuttdev/hipfire-qwen3.5-9b`).
*   **Alignment check:** This matches the actual Hugging Face repositories queried inside `cli/registry.json`.
*   *Note on DeepSeek V4 Flash:* The main DeepSeek repository is owned by `nwoolmer/hipfire-deepseek-v4-flash`, which is correctly documented in `docs/MODELS.md` and configured in `cli/registry.json`.

---

## 3. High-Fidelity Document Alignment (Areas of Perfect Match)

The audit also revealed several core areas where the documentation and the codebase are perfectly aligned, demonstrating highly structured maintenance.

### A. Configuration Ranges and Defaults
The global config keys, ranges, and validation checks documented in `docs/CONFIG.md` are exactly identical to the validations in `cli/index.ts`.
*   `temperature` range of `0.0–2.0` matches the validation `value >= 0 && value <= 2`.
*   `top_p` range of `0.0–1.0` matches `value > 0 && value <= 1`.
*   `max_tokens` range of `1–131072` matches exactly.
*   `max_seq` range of `512–524288` matches exactly.
*   The default `repeat_penalty` value of `1.05` matches perfectly.

### B. Architecture Registry IDs
The `docs/architecture-ids.md` registry perfectly mirrors the actual `Architecture::arch_id()` markers and the load paths dispatched in the daemon:
*   `arch_id = 0` $\rightarrow$ LLaMA / Mistral (`hipfire-arch-llama`)
*   `arch_id = 1` $\rightarrow$ plain Qwen3 / Qwen2 (`hipfire-arch-llama`)
*   `arch_id = 5` $\rightarrow$ Qwen3.5 dense (`hipfire-arch-qwen35`)
*   `arch_id = 6` $\rightarrow$ Qwen3.5 / 3.6 MoE / A3B (`hipfire-arch-qwen35`)
*   `arch_id = 7` $\rightarrow$ Qwen2 dense standalone (`hipfire-arch-qwen2`)
*   `arch_id = 8` $\rightarrow$ Qwen2-VL family (`hipfire-arch-dots-ocr`)
*   `arch_id = 9` $\rightarrow$ DeepSeek V4 Flash (`hipfire-arch-deepseek4`)
*   `arch_id = 0xFF` $\rightarrow$ toy / template (`hipfire-arch-toy`)

### C. CLI Sub-commands and Arguments
`docs/CLI.md` has excellent parity with `cli/index.ts` argument parsing, including:
*   All major commands (`pull`, `list`, `ps`, `rm`, `run`, `chat`, `serve`, `stop`, `bench`, `config`, `quantize`, `sidecar-gen`, `diag`, `update`).
*   Advanced parameters of `sidecar-gen` (e.g., `--corpus`, `--max-tokens`, `--chunk-len`, `--gpu-calib`, `--cpu-calib`, `-o`, `--skip-validation`) map 1:1 with their parsed values in `cli/index.ts`.

---

## Summary of Recommended Action Items

To bring the documentation back into a perfectly synchronized state with the codebase, the following surgical edits are recommended (no files have been modified in this run per user request):

1.  **Standardize TriAttention Extensions:**
    *   Update `docs/CLI.md`, `docs/CONFIG.md`, and `docs/GETTING_STARTED.md` to reference `.triattn.hfq` as the canonical extension (the output of `sidecar-gen`).
    *   Retain mentions of `.bin` strictly as a fallback discovery pattern for legacy/published files.
2.  **Update MQ3 Prefill Details:**
    *   Rewrite the MQ3 section in `docs/QUANTIZATION.md` to confirm that WMMA/MMQ prefill paths are fully implemented on RDNA3/4 (`gfx11` / `gfx12`).
3.  **Sync Environment Variables:**
    *   Perform a baseline check-in using `./scripts/regen-env-vars-doc.sh` to update `docs/env-vars.md`.
    *   Introduce missing production controls and prune stale entries.
4.  **Add `/set` Command to `docs/CHAT.md`:**
    *   Add `/set` to the "Slash Commands" reference table under `docs/CHAT.md` to help users discover session parameter tuning.
