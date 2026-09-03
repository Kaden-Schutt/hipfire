<!-- SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Kaden Schutt; hipfire — see LICENSE and NOTICE in the project root. -->

# hipfire codebase audit — 2026-09-02

_Lifecycle: planned intent (docs/INDEX.md § plans). A point-in-time audit of master `8cd15a62b` with a ranked remediation plan; findings are cited to `path:line` at that commit and go stale as those lines move._

Master `8cd15a62b`. Twelve read-only scouts, one slice each, same rubric (broken / missing / would change, every finding cited to `path:line`). Their full reports are alongside this file as `audit-<Slice>.md`. A follow-up kernel-family audit of MQ4G256V2 (qt=44), the production 4-bit format, is in [`audit-Mq4v2Kernels.md`](audit-Mq4v2Kernels.md) — verdict: sound end to end, one runtime admit out of lockstep, one non-discriminating parity fixture. I spot-checked the five findings below marked ★ by reading the cited lines myself; the rest are the scouts' verified claims and should be read as "cited, plausible, not re-derived by me."

## The shape of it

The codebase is in better shape than the clobber week suggested, and the failure modes are consistent. Almost nothing here is a wrong computation. Almost everything is one of three things:

1. **A door the loader opens that the engine can't walk through.** The loader admits a (model, topology, quant) combination, spends the VRAM, and then the generate side has no arm for it — or has an arm that refuses. #683 (Qwen3.5-MoE EP) was the first one found; the audit found five more of the same species.
2. **Two halves of a migration both alive.** A "canonical" path was introduced (`reset_lifecycle`, `production_fail_closed_rollback`, `ProcessConfig`, `ArchModel`) and most routes moved to it, but a handful of routes — usually the bring-up archs and the vision loops — still carry the old hand-rolled version, which has since drifted.
3. **Comments and docs that describe the design, not the code.** `SpecLoadCfg` says it falls back to ambient env; it doesn't. `gemm_table` says V2 is gfx12-only; it isn't. `env-vars.md` says production never reads ambient `HIPFIRE_*`; the daemon reads six. These aren't cosmetic — two of them (the vmm-on-EP comment in #682, the "mirrors" claim) are exactly how the clobber got approved.

Nothing in the audit is a security problem. Nothing is data loss. The worst class is "a user configures a supported-looking thing, waits for a 19 GB load, and gets an error" — and the second-worst is "a reload leaks VRAM until the process dies."

## Broken — the ones that matter, ranked

**1. ★ Loader admits, engine refuses (the #683 family).** Six instances, all verified:
- Qwen3.5-MoE EP: `hipfire-generate/src/qwen.rs:240` routes arch 6 to the dense server (#683, already filed).
- **Gemma4 lowered/MoE**: the carrier loads `Gemma4Lowered` (`hipfire-arch-gemma4/src/carrier.rs`), and `ar.rs:1160-1170` hard-refuses generate on it — "not yet wired on this build". The load succeeds first.
- **Any EP arch other than 9/10 that reaches `generate_ep`** falls through to `ep_serve_ds4` (`ar.rs` `generate_ep` `_ =>` arm) — wrong server, not a refusal. LFM2/Cohere2 have no `EpArch` at all, so a forced EP load lands here.
- **LFM2 continuous batching**: `batch.rs:166-181` runs four eligibility checks and then `return false` unconditionally. `caps.supports_continuous_batch` is `true` for LFM; the batch state gets allocated and never used.
- **Muse-Glimmer continuous batching**: same pattern — `GlimmerDecodeBatchState` exists (`hipfire-arch-muse-glimmer/src/batch.rs`), `continuous_batch_route` only admits `5|6|11`.
- **GL quant types 38/39** load via `RAW_CODECS` (`weight_backend.rs`) and have no dense GEMV dispatch (`hipfire-dispatch/src/types.rs`), so a GL-quantized dense model loads and cannot decode.

Fix shape is the same for all six: refuse at admission with the reason, before allocation. That's a one-arm change each and the cheapest correctness win in the codebase. Wiring the features is separate work.

**2. ★ MQ8 GEMV family path panics.** `hipfire-dispatch/src/families/gemv.rs:140-148` maps `RotationPlan::Mq8Internal` to a plain FWHT rotation, which never fills `scratch.mq_x_q8`; `rdna-compute/src/gemv.rs:6163` then `unwrap()`s it. The pipeline path (`steps.rs:940`) special-cases MQ8 correctly and is what production uses today, so this is a landmine on the `run_auto` path rather than a live crash — but the plain `KernelKey::GemvMq8G256` is registered and falls through to `MissingImpl` too, so MQ8 has exactly one working route and two broken ones. Hours to fix; add a unit test that runs every registered gemv key once.

**3. ★ VRAM leaks on reload.** `LlamaWeights::free_gpu` (`hipfire-runtime/src/llama.rs:683`) frees `.buf` per weight and skips the PARO rotation and AWQ scale sidecars; the leak repeats every reload of an AWQ/PARO llama/qwen3. `DeviceBuffer` has no `Drop` (`hip-bridge/src/lib.rs`), so every owner that forgets `free` leaks silently — the audit found the `dspark` weights, the DFlash layer staging on partial failure, and the `Qwen3DsparkScratch` intermediates all in that state. (Several of these were what the reverted G4 series *did* fix; those small pieces are worth re-landing individually.)

**4. Reset/rollback is canonical on the main routes and hand-rolled everywhere else.** `production_fail_closed_rollback` covers AR, dense-TP, and DS4. Verified gaps: EP GPU-error exits skip `ep_reset_after_abort` (multi-rank state left dirty — MiniMax LCP especially); the vision cancel path emits a wire terminal without a GPU/DeltaNet rollback (`vision.rs`); vision context-full still does a manual DN memset; the bring-up dense archs (qwen2, minimax, maple, cohere) have an `Abort => {}` arm — no rollback at all. `reset_core_arch_key` maps arch 15 (maple) to `"unknown"`, and `dots_ocr` vs `dots-ocr` hyphen drift means dots never gets reset-core coverage. Any of these is a next-turn corruption bug on a cancel or an error.

**5. EP loads build a Tp mesh.** ★ Every `load_model_ep_*` calls `Gpus::init_tp` (`hipfire-loader/src/lib.rs:2994,3222,3361`), which records `DeviceMesh::rect(Tp, n)` (`multi_gpu.rs:300`). So after an EP load, `mesh.size_of(Ep) == 1`. `#681` just landed this type with no readers, so it's harmless today — but the first consumer of the mesh will get the wrong topology for every MoE load. `Gpus::init_ep` is an hour of work and should land before anyone reads `mesh`. Also: `Gpus::single` records an empty mesh while `from_parts` records `Pp:N` for N=1 — two representations of the same thing.

**6. Config bypasses.** `docs/env-vars.md:51-69` says production never reads ambient `HIPFIRE_*` after `ProcessConfig` is installed. The daemon reads `LOG_FORMAT`, `DFLASH_DRAFT`, `PP_DFLASH`, `PFLASH`, `DPM_WARMUP` directly (`main.rs:497,1199,1575-1589,1801`), the loader reads `PAGE_EVICTION` (`carriers.rs:409`), and `hipfire bench` sets `continuous_batch` via `set_var` as a side channel to itself (`cli/main.rs:3925`). `check-env-docs.py` only catches literal `HIPFIRE_*` string reads and reports success on this tree. The PP experimental gates are snapshotted into config at startup *and* re-read live — two sources of truth for the same flag.

**7. Docs contradict the build.** `CONTRIBUTING.md:58`, `GETTING_STARTED.md:129`, `VALIDATION.md:84-97` still say `cargo run --example daemon -p hipfire-runtime` and reference a `test_kernels` example that doesn't exist; the daemon has been a crate since the saddle. `CONTRIBUTING.md:33-44` links `.skills/`; the real path is `.agents/skills/`. `ARCHITECTURE.md:107` says `--json`/`--no-stream` force local; `main.rs:1911-1918` shows they don't. `MODELS.md` says deepseek-v4-flash defaults to `mq2lloyd`; the registry says `mq2r`. README's model count is three different numbers. `CONTRIBUTING.md:286-295` asks contributors to help with issues #57 and #58, both closed in April/May.

**8. Smaller verified items** (each an hour): `gemm_table.rs:364` comment says V2 is gfx12-only while the code gates on `HasWmma` and gfx11 sources exist; MQ6 residual `ensure_kernel` uses the `_mq5v2` module suffix (`gemm.rs:32097`, no collision, wrong name); Cohere2-MoE docs say LayerNorm, forward does RMSNorm; `SpecLoadCfg` doc claims an env fallback that `spec_build.rs:201` doesn't implement; `verify-bind-thread.sh` only audits `dispatch.rs`, while `impl Gpu` is split across ten files; `hsa-bridge` has ~30 `unsafe` blocks with no SAFETY comments; the registry silently drops aliases whose target doesn't exist (`hipfire-registry/src/lib.rs:328`).

## Missing

- **`Gpus::init_ep`** and EP-mesh admission (see 5).
- **A `DeviceBuffer` Drop or a `#[must_use]` free** (see 3) — the one structural change that would retire a whole bug class; days, because VMM-backed buffers have different owners.
- **Fail-closed `generate_ep`** and EP admission refusal for arch 11/12/13.
- **Docs-drift CI** beyond `check-env-docs.py` — the entrypoint/path/count drift above would all be caught by a script that greps docs for `cargo run --example`, `.skills/`, and compares README counts to `registry/models.json`.
- **Windows device selection** (issue #669): `hardware.devices` → `ROCR_VISIBLE_DEVICES` doesn't work there; no docs, no fallback. HUSRCF's workaround is `CUDA_VISIBLE_DEVICES`, which nothing documents.
- **LFM2 multi-turn**: always cold-resets conv+KV (`dense.rs:6105`), so every turn re-prefills. Correct, but it's why LFM feels slow in chat; a conv-state snapshot is a few days.
- **DS4 `route_scale`** is forced to 2.2/1.8 instead of the checkpoint's 1.5 (`config_cache.rs:450-510`) to compensate for a MoE routed-branch shortfall nobody has root-caused. It works; it's also a hidden numeric fudge that will bite the next quant.
- `hipfire registry` subcommand is absent from `docs/CLI.md`; `init_vram_weighted` is a stub that errors; `ds4-parent` is an offline oracle and should stay quarantined (it is).

## Would change — what I'd actually do, in order

1. **Admission refusals for the six #683-class combinations** (hours, one PR, ships behind hw-gate). Biggest user-facing win per hour in the codebase; every one currently costs a user a full load before the error.
2. **`Gpus::init_ep` + unify `single`'s mesh** (hours). Land before anything reads `mesh`.
3. **Re-land the G4 leak fixes as individual PRs**: `LlamaWeights` sidecar frees, DFlash/DSpark staging on failure, `Qwen3DsparkScratch` (days total, each one a small PR with a fault scenario, gated).
4. **Close the reset gaps**: EP error exits → `ep_reset_after_abort`; vision cancel → rollback; bring-up `Abort => {}` arms → the canonical rollback; fix `reset_core_arch_key` for maple and the dots hyphen (days). This is the G4 work that was actually right, done at a size that can be reviewed.
5. **MQ8 rotation family** (hours) + a test that launches every registered gemv key once.
6. **Route the residual ambient env reads through `ProcessConfig`** and make `check-env-docs.py` fail on them (1–2 days). Until then `env-vars.md`'s promise is false.
7. **Docs pass** (a day): entrypoints, `.agents/skills`, force-local, model counts, MODELS default, ARCHITECTURE carrier table, INDEX pin, closed-issue asks. Then a drift script in CI.
8. **`DeviceBuffer` RAII** (days, coordinate VMM). The structural fix for class 3.
9. **Issue triage** (a day): 88 open; the scout's table in `audit-DocsIssues.md` clusters them. #675/#677/#527 are historical after the revert; #593/#595/#563 are on a stale beta base; ~20 have had no response in 30+ days.

Things I would explicitly *not* change now: the DS4 route-scale compensation (works; root-cause first), PFlash (retired, quarantined, not reachable from the daemon), `ds4-parent`.

## Confidence

Twelve slices, ~7 minutes each, read-only. The scouts were told to prefer 10–20 findings that matter over exhaustiveness, so the long tail of the arch crates (deepseek4's 36k lines, rdna-compute's 178k) got a survey, not a read. Kernels (`kernels/`, 146k lines HIP) were sampled for dispatch coverage only; nothing in this audit says the math is right — that's what the redline/parity routes are for. I re-derived five findings by reading the lines; the rest carry the scouts' citations. No finding contradicts another scout's, and the three that overlap (#683 family, reset gaps, VRAM leaks) were found independently by two or three slices each, which is the best signal that they're real.
