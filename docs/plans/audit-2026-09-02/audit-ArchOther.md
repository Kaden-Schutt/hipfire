<!-- SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Kaden Schutt; hipfire — see LICENSE and NOTICE in the project root. -->

# Audit: ArchOther

# Audit ArchOther

## Broken

1. **Gemma4 lowered/MoE loads; generate hard-refuses (verified)** — `hipfire-generate/src/ar.rs:1160-1170`. If `gemma4_lowered_mut().is_some()`, emit *"lowered/MoE generate not yet wired (eager dense only)"*. Carrier dual Eager/Lowered (`hipfire-arch-gemma4/src/carrier.rs`). Load-OK/generate-fail archetype. Open #678 describes lowered generate on a branch; **master still refuses**.

2. **Gemma4 EAGLE full code, not product path (verified)** — `hipfire-arch-gemma4/src/speculative.rs:1-50` `spec_step_gemma4_eagle`; drafter + `infer_gemma4_spec`; `dense.rs` ~2226 EAGLE arm when arch-22 drafter + greedy. Gate-off class a72279cb1 (greedy parity). Dual maintenance.

3. **Optional bias panics (verified)** — `tools/change_gate/routes.py` documents `tensor not found: layers.0.self_attn.q_proj.bias`. `WeightBackend::bias` non-optional. 31B-class optional bias; 12B dense OK. Distinct from #614 hipGraph and #678 ThoughtRouter.

4. **Muse-glimmer continuous batch implemented, never staged (verified)** — `muse-glimmer/src/batch.rs` `GlimmerDecodeBatchState` + `batch_weight_formats_supported`. `hipfire-loader/src/lib.rs:186-198` `continuous_batch_route` only 5|6|11. `carriers.rs:2123-2124` `supports_continuous_batch: false`.

5. **MuseGlimmerBundle/ArchModel orphaned in loader (verified)** — no arch-crate `carrier.rs`; `MuseGlimmerCarrier`+bundle in `hipfire-loader/src/carriers.rs` arch 14. Contradicts `hipfire-runtime/src/arch.rs:1-30` (ArchModel in arch crate).

6. **dots-ocr docs say load unwired; code wired (verified)** — `lib.rs`/`map.md` vs `carrier.rs`+Architecture+ArchModel+SpecTarget arch 8.

7. **`Architecture::eos_filter_overrides` dead for product generate (verified)** — trait `arch.rs:191-194`; dots/gemma4 overrides; generate never calls them (Architecture used for cohere2moe/deepseek4/minimax load). `dense.rs:7678-7682` Maple no EosFilter. Dots tests only `arch.rs:138-142`.

8. **DotsOcr SpecTarget vision lifetime contradiction (verified)** — `spec_impl.rs:58-68` comment vision NOT included/freed after prefill; struct holds full `DotsOcrWeights`.

## Missing

1. Gemma4 vision stubs (`gemma4_vision.rs`).
2. Maple: no Architecture/SpecTarget/batch; no EOS/think filter on generate.
3. Glimmer spec/MTP: drafter exists; caps MTP false; emitter unwired.
4. Toy unshippable (`0xFF`, load always Err) still sole “template” teaching Architecture-centric world.
5. saddle-core: caps real; Architecture naming split confuses; spec helpers thin.
6. No load-time refuse when generate cannot serve lowered gemma4.

## Would change (ranked)

1. Fail-closed gemma4 lowered at load **or** wire generate_lowered — carrier+ar.rs — **hours**.
2. Optional bias → Option/skip-missing — weight_backend+gemma4 — **hours–days**.
3. Fence/delete EAGLE product surface — speculative/drafter/dense — **days**.
4. Move MuseGlimmer Carrier+Bundle+ArchModel into arch crate — **days**.
5. Wire or delete glimmer continuous batch — batch+route+caps — **days–week+**.
6. dots-ocr doc cutover + vision lifetime + generate-side EOS — **hours**.
7. Maple EOS/stop on generate without full Architecture — **hours**.
8. Retire toy as sole template; point at maple+dots — **hours**.

## Confidence

Read-only master `/home/kaden/ClaudeCode/warpfront/hipfire`. No GPU/tests/builds. Did not exhaust every forward kernel body. saddle-quant → AuditQuantize; MoE/Qwen generate → peers. GitHub open: #614, #678, #672 not re-filed; findings additive. Re-check ar.rs if lowered generate merges.

## Contract JSON (for Main persist)

```json
{
  "slice": "ArchOther",
  "broken": [
    {"title": "Gemma4 lowered/MoE loads but generate hard-refuses", "path_line": "hipfire-generate/src/ar.rs:1160-1170", "verified": true, "summary": "Loader publishes Gemma4Lowered; ar.rs refuses generate eager-only."},
    {"title": "Gemma4 EAGLE full code production-gated", "path_line": "hipfire-arch-gemma4/src/speculative.rs:1-50", "verified": true, "summary": "spec_step_gemma4_eagle+drafter live; product path gated."},
    {"title": "WeightBackend bias panics optional missing", "path_line": "tools/change_gate/routes.py", "verified": true, "summary": "q_proj.bias tensor not found panic; 31B-class."},
    {"title": "Glimmer CB implemented never staged", "path_line": "hipfire-loader/src/lib.rs:186-198", "verified": true, "summary": "batch.rs exists; route only 5|6|11; caps false."},
    {"title": "MuseGlimmer Bundle/ArchModel in loader", "path_line": "hipfire-loader/src/carriers.rs", "verified": true, "summary": "Orphan rule; no arch carrier.rs."},
    {"title": "dots-ocr docs claim load unwired", "path_line": "hipfire-arch-dots-ocr/src/lib.rs", "verified": true, "summary": "Carrier wired; docs stale."},
    {"title": "eos_filter_overrides not used by generate", "path_line": "hipfire-runtime/src/arch.rs:191-194", "verified": true, "summary": "Trait docs lie relative to generate."},
    {"title": "DotsOcr vision doc vs bundle", "path_line": "hipfire-arch-dots-ocr/src/spec_impl.rs:58-68", "verified": true, "summary": "Comment no vision; struct has DotsOcrWeights."}
  ],
  "missing": [
    {"title": "Gemma4 vision", "path_line": "hipfire-arch-gemma4/src/gemma4_vision.rs", "verified": true, "summary": "Stubs only."},
    {"title": "Maple Architecture/SpecTarget/EOS", "path_line": "hipfire-arch-maple/src/carrier.rs", "verified": true, "summary": "Thin surface; generate EOS gap."},
    {"title": "Glimmer spec/MTP", "path_line": "hipfire-loader/src/carriers.rs caps", "verified": true, "summary": "Drafter present; MTP false."},
    {"title": "Toy unshippable template", "path_line": "hipfire-arch-toy", "verified": true, "summary": "0xFF always-Err load."},
    {"title": "saddle-core Architecture naming split", "path_line": "saddle-core/src/caps.rs", "verified": true, "summary": "caps real; naming confuses."},
    {"title": "Load-time refuse lowered gemma4", "path_line": "carrier.rs + ar.rs", "verified": true, "summary": "Only generate-time error."}
  ],
  "changes": [
    {"title": "Fail-closed or wire lowered generate", "path_line": "carrier.rs + ar.rs", "cost": "hours", "summary": "Stop load-OK/run-fail."},
    {"title": "Optional bias Option/skip", "path_line": "weight_backend", "cost": "hours-days", "summary": "Unblock 31B-class."},
    {"title": "Fence/delete EAGLE product", "path_line": "speculative.rs", "cost": "days", "summary": "Reduce dual path."},
    {"title": "Move glimmer Carrier into arch crate", "path_line": "carriers.rs", "cost": "days", "summary": "Match maple/dots."},
    {"title": "Wire or delete glimmer CB", "path_line": "batch.rs + continuous_batch_route", "cost": "days-week+", "summary": "End half-migration."},
    {"title": "dots doc+vision+EOS", "path_line": "lib.rs spec_impl generate", "cost": "hours", "summary": "Cutover stale contract."},
    {"title": "Maple generate EOS", "path_line": "dense.rs maple", "cost": "hours", "summary": "Stop-token without Architecture."},
    {"title": "Retire toy as sole template", "path_line": "hipfire-arch-toy", "cost": "hours", "summary": "Point at maple+dots."}
  ],
  "report": "local://audit-ArchOther.md"
}
```
