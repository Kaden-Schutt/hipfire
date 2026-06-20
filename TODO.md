# TODO

## Evaluation Branch

### Active

- Extend startup device inventory into daemon-owned placement and scheduler
  routing:
  - `hipfire-model` now owns a shared accelerator inventory/device contract,
    `/health.runtime_workers` reports the shared inventory payload, and the
    daemon exposes a typed JSONL `inventory` command;
  - server health polls daemon inventory when a daemon is already running and
    otherwise reports `source=not_probed`, while worker identity carries
    accelerator kind/device id;
  - priority prefill scheduler admission can now consume accelerator inventory
    and reject sessions targeting missing or unavailable worker devices;
  - daemon inventory now includes explicitly configured XDNA1/NPU rows from the
    NPU module runtime/artifact contract;
  - Rust server now has a daemon-inventory-backed `PriorityPrefillScheduler`
    construction seam, and `/health` shares the same inventory probe helper;
  - replace the XDNA1 env/artifact sentinel with a real XRT hardware probe when
    NPU runtime dispatch owns that boundary;
  - wire live server request scheduling to pass daemon inventory into scheduler
    admission at the request queue site instead of using a single
    server-selected HIP placement.
- Regenerate quality KLD references as first-party HFQM `.kldref.hfq`
  packages. Do not trust previously downloaded raw `.kldref.bin` files for
  baseline claims. Regeneration must use Hipfire reference execution with
  `--kv-mode fp32` and FP32 DeltaNet state, and metadata must record the source
  model hash, slice hash, KV mode, state precision, `top_k`, context length,
  and producer command.
- Unblock full KLD regeneration throughput. The current first-party producer
  emits correct metadata and can produce smoke `.kldref.hfq` packages, but the
  0.8B BF16 `top_k=256` path still runs at roughly 20 scored tokens/s on the
  full 2048-token slice shape. Add a GPU-side `top_k=256` reducer or a faster
  BF16/F32 `lm_head` path before replacing the legacy raw refs.
- Keep model-backed profile collection in the eval harness. The `profile`
  battery should run a real Hipfire model-backed anchor and ingest runtime
  evidence artifacts, especially `moe_router_histogram.json` for MoE/A3B
  models.
  - The profile row now declares the expected runtime evidence kinds
    (`performance`, `memory`, `launch_counts`, `moe_router_histogram`) so
    artifact collection regressions stay visible in row metadata.
  - Generic runtime oneshot evidence writers and sparse router-histogram
    artifact rendering now live in `hipfire-evidence`; model-specific gatherers
    adapt their native counters into the shared histogram contract.
  - `GenerateTextRequest.evidence_dir` lets daemon-backed AR text generation
    write standard runtime oneshot artifacts directly via `hipfire-evidence`;
    DFlash/MTP/VL evidence emission remains a path-specific follow-up.
  - The profile battery now has a daemon-backed model anchor that reuses the
    shared daemon load path and requests runtime evidence artifacts.
  - Daemon Qwen3.5 MoE AR requests now adapt native router counters into the
    generic `hipfire-evidence` histogram contract when `evidence_dir` is set;
    DFlash/MTP/VL evidence emission remains a path-specific follow-up.
- Run the full no-GPU handoff gate before committing the branch.
- Extend Qwen35 paged expert execution coverage beyond the current indexed MQ
  routed path: grouped Path 2, Paro, full-precision, and CPU-fallback paged
  paths still need explicit tests before broad admission.
- Extend routed-only Qwen3 MoE forward validation beyond the current one-token
  modular/lazy expert smoke: multi-token prefill, decode continuation, eager vs
  paged parity on a smaller artifact, and grouped routed batches still need
  explicit coverage before removing the guarded execution flag.
- Convert eval execution from “examples sub-process fan-out” toward daemon-backed,
  reusable model processes before the next modularization step:
  - The user-visible symptom is repeated model loading in `--tier fast` runs:
    same model path is reloaded in separate subprocesses for smoke/coherence/
    speed/pflash/dflash rows because examples executor launches one process per row.
  - Evidence from latest run:
    - command line `hipfire-eval --model <hfq> --tier fast` emitted multiple
      `loading weights: ... HFQ payload` blocks in logs for the same model.
    - per-row commands in results showed duplicated `examples/run` plus speed cases.
  - Root cause hypothesis:
    - evaluator still routes key batteries to `examples` binaries (new subprocesses)
      not a resident daemon/service boundary.
    - no shared model lifecycle exists across battery rows; every row independently
      executes `run`/`run --session-reset-smoke`/`bench_qwen35_speed` etc.
  - What to finish to align with daemon/lib+IPC direction:
    - finish `EvalClient` transport in eval harness with request batching for same
      model across batteries where semantics allow;
    - daemon-backed smoke model-load, finite greedy decode, repeated greedy
      reset/recall, and explicit `--executor daemon --battery speed` timing
      anchors now use the shared JSONL daemon adapter; examples-backed pp32/pp128
      speed-gate rows remain the benchmark-grade default;
    - default `--executor auto` now prefers daemon-backed smoke/speed rows and
      shares one daemon load across those rows and the profile anchor when they
      run together;
    - add process and handle ownership so one model-load maps to one resident daemon
      process per loaded quant/placement;
    - make battery rows consume a shared model cache key and avoid repeated
      `hfq` opens.
  - Success criteria:
      - `--tier fast` on a single model produces a single model-load event per
        daemon-backed model per run.
      - row metrics still preserve current throughput correctness and hard-fail
        semantics.
      - no functional regression in `--battery speed`, while retaining the
        command-level `--prefill-list` optimization already in `bench_qwen35_speed`.

- [Documentation debt] Refresh docs drift where active behavior has changed:
  - Keep TriAttention sidecar examples and naming in active docs on
    `.triattn.hfq` as canonical; mention `.triattn.bin` only as explicit legacy
    compatibility.
  - Regenerate `docs/env-vars.md` from `./scripts/regen-env-vars-doc.sh` after
    source/env surface changes.

### Deferred

- `crates/hipfire-runtime/src/kv_adaptive.rs` — adaptive KV precision downshift
  module, written but not yet wired in. Missing pieces before use:
  - `KvCache::transcode_v_step` and `transcode_k_step` methods (called by
    `maybe_downshift`) are not yet implemented.
  - Module is not declared in `crates/hipfire-runtime/src/lib.rs`.
  - `Conservative` and `Aggressive` presets share the same floor values
    (both Fwht2/Lloyd2) — likely a placeholder; floors should differ.
  - No integration site in the decode loop (`maybe_downshift` must be called
    after each committed token write alongside `maybe_evict`).

- Adapt the build system to autodiscover GPUs the same way NPU detection now
  works: query HIP/ROCm at `cargo build --features npu-kernels` time (or a
  new `gpu-kernels` feature) to identify installed GPU arch(s) and select
  kernel tuning parameters automatically, rather than requiring the user to
  pass an explicit arch flag. The NPU precedent is `detect_npu()` in
  `tools/npu/build_qwen35_swiglu.py` + `HIPFIRE_NPU_TARGETS` env override in
  `crates/hipfire-arch-qwen35/build.rs`.
- Finish full daemon-backed `hipfire bench` replacement after eval-backed speed
  rows match the current public output shape.
- Promote long-context, vision, CASK/TriAttention, DFlash resident, cold-process
  distribution, and Kernel Atlas artifact ingestion from explicit skipped or
  external-evidence rows into native model-backed eval batteries.
- Extend host capability profiling beyond the current GPU/storage/memory report
  to measure NPU bandwidth paths when an NPU is present, and store measured
  bandwidth alongside static hardware metadata in eval reports.
- Migrate imatrix, CASK/TriAttention, DFlash sidecars, and other non-weight
  analysis packages into metadata-rich HFQM containers after the KLD reference
  package format is settled.

## PFLASH Review Debt (migrated from MANUAL_REVIEW)

- Investigate and close the remaining long-context pflash score kernel regression:
  - `qwen3.5-4b` + drafter paths currently exhibit `ScoringDegenerate { non-finite
    scores: 337 NaN, 0 inf }` at 32K NIAH source (`21551` tokens) despite 8K/16K pass.
  - Work to isolate root cause (drafter forward numerical blow-up vs score kernel)
    and add targeted diagnostics in `pflash::compute_scores_batched_gpu`.
- Keep the historical full-coherence/speed hang trace in the historical docs index
  as an execution quirk note, and re-run the gate from a clean environment when
  the next session reset is available.

## FWHT Residual QJL Transform

Status: deferred.

- Implement a Johnson-Lindenstrauss / QJL transformation on the residual in the FWHT path. The current FWHT path applies a signed-FWHT rotation to Q/K for attention and leaves the residual stream without a separate QJL transform.



## Check all hot paths for graph safety
>>> One issue surfaced before verification: gemm_f16_x_f32_wmma currently launches with raw stack kernargs rather than the graph-safe blob helper. I’m tightening the env gate so this experimental route only runs outside hipGraph capture; captured paths will keep using the scalar default until the dispatcher wrapper is made graph-safe.


## further investigate using packed 4 bit operations on gfx1151/RDNA3/RDNA3.5

## DeltaNet state precision (follow-ups to Phase A gate, 2026-06-15)

Phase A made the DeltaNet recurrent state default to **FP32** for all
current models, gated on redundancy (`head_dim × n_heads`) via
`qwen35::default_state_quant` (env: `HIPFIRE_DN_STATE_QUANT`,
`HIPFIRE_DN_STATE_FP32_BELOW`). Two kernels are missing to generalize it:

1. **Real FP16 DeltaNet state kernel.** `StateQuant` is only
   `{FP32, Q8, Q4}`; there is no `gated_delta_net_f16`. The plan wants
   FP16 for high-redundancy models (cheaper than FP32, safer than Q8),
   but it doesn't exist, so the gate currently selects FP32 for the whole
   upper tier (threshold `usize::MAX`). To finish:
   - add `StateQuant::FP16` + alloc path in `DeltaNetState::new_with_quant`;
   - add `gpu.gated_delta_net_f16(...)` + `_batch_seq` / `_routed_batch_seq`
     dispatch (mirror the FP32 kernels);
   - coherence-gate the f16 state numerics, then lower the default
     `HIPFIRE_DN_STATE_FP32_BELOW` so big models use FP16.

2. **Higher-precision tree DeltaNet replay (FP32 *and* FP16).** Tree-mode
   spec-decode hard-errors on FP32 state (`qwen35.rs`: *"FP32-state batched
   prefill does not support tree DeltaNet replay yet"*); only
   `gated_delta_net_q8_tree_batch_seq` exists. Consequence: with the FP32
   default, tree-based spec-decode (DDTree, DFlash-tree,
   `spec_step_dflash_mtp_tree`) cannot run — Phase B MTP drafting must use
   the **non-tree** `spec_step_mtp` path. Future work, both precisions:
   - `gated_delta_net_f32_tree_batch_seq` — FP32 S-tape replay, unblocks
     tree drafting at full anchor precision (serves the FP32-forced
     low-redundancy models like 0.8B directly).
   - `gated_delta_net_f16_tree_batch_seq` — FP16 S-tape replay, pairs with
     the FP16 state kernel (item 1) so high-redundancy models can run tree
     drafting at the cheaper FP16 tier once FP16 coherence is established.
   Until either lands, tree drafting needs `state_quant=q8` (only safe on
   high-redundancy models, never 0.8B/2B).

## Native Rust/GPU Hessian collector (finish the collect_hessian.rs scaffold)

The QTIP LDLQ path (Phase C1e) needs per-layer input Hessians. The native
`crates/hipfire-runtime/src/bin/collect_hessian.rs` is currently a SCAFFOLD
that panics ("not implemented"); we fall back to the Tier-2 Python collector
`scripts/collect_hessian.py` (HF transformers + ROCm/CPU torch). The Python
path WORKS (validated 2026-06-16: 0.8B → 2.21 GB HFHS file, read by
`hessian_io.rs`), so the HFHS-v1 format + approach are proven.

Implement the native Tier-1 collector (the scaffold's own TODO list) so we
get GPU speed and drop the torch dependency:
- BF16 safetensors → device tensors loader.
- On-GPU K×K outer-product Hessian rank-1-update kernel (accumulate
  H += xᵀx over calibration tokens per GPTQ-target linear).
- `ActivationCapture` wiring at the GPTQ-target dispatch sites in the
  qwen35/generic forward (capture each linear's input activations).
- HFHS-v1 binary writer matching `scripts/collect_hessian.py` (so
  `hessian_io.rs` reads it unchanged).

Benefit: CPU torch is slow (0.8B calibration is minutes; bigger models
hours); the GPU forward + on-GPU accumulation would be far faster, and it
removes the Python/torch tooling dependency from the quant pipeline.
Reference: the validated Tier-2 `scripts/collect_hessian.py` + the existing
scaffold's documented deliverables.

## GPU (HIP) trellis-encode kernel for QTIP quantization

The QTIP encode (`qtip::beam_encode_group_bits` + the LDLQ block loop in
`ldlq.rs`) is the slow stage: per 256-weight group it runs 256 sequential
Viterbi steps, each generating `beam_width × 2^bits` candidates
(128 × 8 = 1024 at 3-bit) and doing a sort/dedup-by-state + top-`beam_width`
select. CPU rayon parallelizes only across groups/rows, capping throughput at
core count (0.8B qtip3-sim ≈ 10 min wall on 24 cores; LDLQ is slower still).
This is pure *offline* cost (Rule 1 does not apply — quantize is tooling), so
spend GPU compute once to improve the many. **Verdict: HIP/RDNA is the right
target; the XDNA NPU is not** (beam search is sort/branch/backtrack-heavy —
the opposite of the AIE dataflow-MAC array; only FWHT + the LDLQ Cholesky/
residual GEMMs are NPU-shaped, and those aren't the bottleneck).

Build a HIP encode kernel mirroring the structure of the decode kernel
(`kernels/src/gemv_qtip3g256.hip`), single backend:
- **One group per workgroup/wavefront.** Cross-group parallelism (millions of
  256-groups) saturates every CU; the 256-step recurrence stays serial inside
  the workgroup (don't parallelize the time axis).
- **Codebook in LDS.** 4096 × f32 = 16 KB fits the LDS budget; the per-step
  cost eval becomes an LDS read + FMA. Or recompute the 1MAD hash per lane
  (the same few-ALU-op hash the decode kernel already computes — bit-identical
  to the PPL-validated path).
- **Beam + candidates in LDS.** beam_width 128 × (state u32 + cost f32) is
  tiny; the 1024 per-step candidates are an LDS scratch.
- **Per-step top-k is the only hard op** — a bitonic top-k in LDS (~10 stages)
  or a segmented min-reduction keyed on next-state. This is the bulk of the
  kernel engineering.
- **LDLQ arm:** keep full row-parallelism (thousands of m rows encode
  concurrently), serialize only the ~k/256 column-blocks (block residual feeds
  the next block). The per-tensor inverse-Cholesky (O(k³), k ≤ 4096) is a
  one-time dense-LA op (rocSOLVER/rocBLAS), not the bottleneck.

Expected payoff: order-of-magnitude+ over the 24-core path (thousands of
groups in flight vs ~24). Precedent: QTIP's own reference runs trellis encode
+ LDLQ on GPU. Cheap CPU-side stopgap meanwhile: lower `beam_width` (trellis
quality is flat above ~16–32). Priority rises when encode moves to 4B/9B,
where the offline cost actually starts to hurt.

## hipfire finetune tool (QTIP blocked finetune / quant-error recovery)

QTIP's headline sub-4-bit numbers depend on **blocked fine-tuning** to recover
quality after quantization; hipfire has no training stack, which is why pure
PTQ qtip3-sim lands at +8.3% PPL (15.20 vs MQ4 14.03) on the 0.8B worst case
and 2-bit collapses there even with LDLQ. A native finetune tool would close
that gap (and is the realistic path to *usable* 2-bit on bigger models, where
the paper says blocked FT is what makes 2-bit work). Now unblocked: halo
(Strix Halo gfx1151, 124 GB unified RAM) runs ROCm torch on the 8060S GPU
(C1h note in NEXT-STEPS), so GPU finetune of the 0.8B is feasible (slow at
~3 TFLOP/s, but it's a one-time offline cost — acceptable per the "improve the
many" principle).

Scope (offline tooling — Rule 1 does not apply):
- **Blocked / layer-wise finetune** of the QTIP-quantized model against the
  fp16 teacher: freeze the trellis symbol assignment, learn the per-group
  `scale` (and optionally the codebook affine) to minimize per-block output
  error — the cheap, decode-compatible recovery that keeps the kernel
  unchanged. Then optionally end-to-end finetune of the dequantized weights
  with straight-through trellis re-encode (the paper's full recipe).
- **Teacher/student wiring:** reuse the calibration corpus + the Hessian
  collector's `ActivationCapture` sites; the loss is block-output MSE (or KL on
  logits for the final stage).
- **Backend decision:** Tier-1 native HIP training is a large lift; Tier-2 is a
  gated Python/torch tool on halo (matches the Hessian collector's two-tier
  precedent — Python is allowed for offline tooling). Start Tier-2 to get a
  quality verdict, promote to native only if it pays off.
- **Acceptance:** qtip3 (and a 7B+ qtip2) PPL closes meaningfully toward MQ4 /
  fp16 after FT; coherence-gate clean; the finetuned artifact still decodes
  bit-identically on the `gemv_qtip3g256` kernel (scale-only FT) or re-packs
  cleanly (weight FT). Validate on a 7B+ model on halo where 2-bit QTIP is
  expected to become usable.

### Legible golden for the tiny fixtures (first customer of the finetune tool)

Upgrade the tiny-fixture golden (see "Tiny random-init fixtures" below) from an
opaque `logit_hash` + random-token argmax to a **self-documenting generated
sequence**, once the finetune tool can memorize a short prefix. Memorizing one
fixed prefix is the *simplest* training objective (pure overfit, seconds on
CPU) — so this doubles as the finetune tool's own smoke test.

Design (force a fixed-length, e.g. 256-token, greedy generation):
- **Trained legible preamble** — e.g. `"The model is working, what follows is
  deliberately random:"`. Human-readable "is it alive / catastrophic
  regression" signal; a CI failure is instantly interpretable.
- **Untrained random tail** — the rest. NOT trained → high-entropy. "Random"
  = varied *content*, still fully **deterministic** (fixed weights + greedy +
  deterministic kernels). This is the **sensitive** tier: near-tie tokens sit
  on decision boundaries, so they flip under subtle drift that the confident
  (large-margin) memorized preamble would mask — recovering the sensitivity
  pure memorization throws away. Keep a hash of the tail as the byte-exact
  assertion.
- **Bonus coverage:** 256 generated tokens exercise the full autoregressive
  decode loop + KV-cache growth (the current single-position prefill golden
  doesn't).
- **Vocab:** byte-level (256-vocab, English as raw UTF-8, embed ≈ 65K params,
  no tokenizer file) keeps it tiny; avoids the 248K-vocab embed blowup.
- **MoE caveat:** the near-tie tail tokens are exactly where MoE-down atomicAdd
  ULP noise can flip run-to-run → on the MoE fixture pin the deterministic
  combine (or keep the run-twice determinism check). Dense path: a tail flip =
  a real change.
Complements, does not replace: `logit_hash` stays the sensitive tier today; the
35B agentic-gate stays the *behavioral* arbiter (this is still memorized, not
Q&A capability).

## Tiny random-init fixtures + golden-output tripwire (fast kernel/MoE plumbing)

The coherence/agentic gates load `qwen3.6-35b-a3b` because the agentic gate is
*behavioral* (JSON.parse + tool-call schema match under 780–1300 token agent
prompts, guarding the #87 long-prompt MMQ-corruption class). But the
*regression-cover* half of that — "did a kernel start corrupting long-prompt
output" — is really a **golden-output (characterization) test**: it needs
determinism + coverage of the same kernel-selection branch, NOT a model that
emits valid JSON. So a sub-10M random-init model is a valid **fast tripwire**;
the 35B stays as the behavioral arbiter.

Coverage holds because auto-MMQ selects on `batch_size >= min_batch`
(`arch_caps.rs:229`) — token/batch count, not model dims — so a tiny model on
the same long prompt crosses the *same* MMQ gate (`HIPFIRE_MMQ_MIN_BATCH` can
force it lower for a cheap deterministic test). Residual: the specific MMQ
*tile variant* can differ by K/N, so it's same-selection-branch coverage, not
identical-tile — hence tripwire + 35B backstop, not standalone.

Build TWO fixtures (must be hipfire's supported archs, NOT upstream
`qwen3_moe` — unsupported; see `main.rs:6210`):
- **Tiny dense (arch 5, `model_type: qwen3_5`).** Dense GEMV/attention writes
  unique outputs, **no atomicAdd → deterministic on a fixed binary**. So a
  **byte/token-exact golden is stable run-to-run**, and any drift is a real
  signal — escalate to the 35B golden. This is the clean primary tripwire.
- **Tiny MoE (arch 6, `model_type: qwen3_5_moe`,** DeltaNet LA+FA hybrid +
  stacked-3D experts `mlp.experts.gate_up_proj/down_proj [E,…]`**).** Covers
  router/expert-gather/grouped-MQ-GEMM. CAVEAT: MoE-down combine uses
  `atomicAdd` with **documented non-deterministic final bits** (`kernels.rs:
  3751`, `gemv_hfq4g256_moe_down.hip:19–23`) → a raw byte-exact MoE golden
  diffs run-to-run with no code change. FIX: pin the golden to the in-tree
  **deterministic no-atomicAdd combine** (expanded per-expert outputs +
  ordered `moe_down_combine_k8_batched`, `kernels.rs:3748`). Bonus: that makes
  the harness double as a **determinism gate** (catches anything re-routing
  MoE-down through the atomicAdd path).

Two-tier policy: tiny golden runs always (seconds, CPU/no-GPU-friendly);
on drift, run the 35B *golden* (not the coarse JSON-valid check — it can pass
while tokens shift). Only-tiny-moved ⇒ tiny-specific, rebaseline; 35B-also-
moved ⇒ real change, rebaseline both deliberately (never auto on a coarse
pass). Greedy decode, fixed long agent-shape prompt, forced MMQ.

Generator = a **tiny-fixture emitter built into `hipfire-quantize`** (NOT a
one-off script). Rationale: the quantizer already owns each arch's tensor
manifest + the HF→internal name mapping (that's what its ingest path does), so
reusing it as the single source of truth keeps fixtures from drifting; a
standalone Python generator would re-derive every arch's tensor list and rot
when a layout changes. Native Rust also drops the torch/transformers dep.

Design:
- **Emit HF safetensors + `config.json`, then run the normal `--input` quantize
  path** (don't synthesize `.hfq` internals directly) — this exercises the
  arch-specific **name-mapper** too, a common break point, so the fixture flow
  gives full-pipeline coverage for free.
- **Seeded random init** (byte-reproducible across machines → stable golden) +
  a per-arch **"tiny preset"**: dims <10M params while preserving the structural
  features gating needs — for Qwen3.5: ≥1 of EACH layer type (DeltaNet +
  FullAttn, dense + MoE), enough experts for top-k, batch large enough to cross
  the MMQ threshold. (Router-margin knob for MoE fixtures is DEFERRED until the
  hipfire finetune tool exists — see that section; until then the MoE fixture
  golden is stabilized by pinning the deterministic combine.)
- CLI shape e.g. `hipfire-quantize --emit-fixture <arch> --tiny --seed N
  --out <dir>` → then quantize to mq4/qtip3/etc.
- **Cost to budget:** each arch's manifest is today *implicit* in the ingest/
  mapping code; the emitter needs it *explicit/enumerable* ("arch → [(name,
  shape-formula)]"). Modest refactor, but healthy — the same table a fixture
  emitter wants is also what a manifest-validator and per-arch docs want, and
  it's a forcing function to document tensor layout when adding an arch.
- **Generalizes:** the dense (arch 5) + MoE (arch 6) goldens are just the first
  two consumers; deepseek4 / minimax / lfm2moe / dots-ocr each get a tiny
  gating fixture from the same mechanism as support lands. <10M is trivial
  (hidden ~256, 2–4 layers, 8 experts top-2, small `moe_intermediate`).

Wire the golden runner into `no-gpu-ci.sh` (CPU reference) + a GPU dispatch
channel-test. Build order: dense arch-5 first (isolates the shared DeltaNet
LA+FA hybrid manifest, deterministic golden), then MoE arch-6 is additive
(router + experts + the combine-path stabilization above).

### TODO: extend tiny-golden coverage to the other model families

**Status:** P1–P3 of `docs/plans/2026-06-20-tiny-golden-tripwire.md` shipped —
the two-tier gate is live, but it only covers **qwen3.5/3.6 dense (arch 5) +
MoE (arch 6)**. We now ship many more families; each is a forward-pass path the
tripwire is blind to, so a regression there falls back on the qualitative 35B
battery (or nothing, if the heavy gate is skipped on a tiny-pass).

**Families with no tiny fixture yet** (`crates/hipfire-arch-*`, excl. the
synthetic `toy`): `llama`, `qwen2`, `gemma3`, `gemma3-vl`, `deepseek4`,
`minimax`, `lfm2moe`, `qwen35-vl`, `dots-ocr`. The VL ones (`gemma3-vl`,
`qwen35-vl`, `dots-ocr`) also need a tiny image-feature stub to exercise the
vision/projector path, or a text-only mode.

**Each family needs two pieces** (today both hard-code qwen35):
1. **`emit_fixture` preset** (`hipfire-quantize/src/fixture.rs`) — the `match
   arch_norm` only has `qwen3_5` / `qwen3_5_moe`; its own error already says
   "Add a tiny preset per arch". Each preset = a <10M config + explicit tensor
   manifest (the manifest-must-be-enumerable refactor noted above).
2. **`fixture_golden.rs` arch dispatch** — it hard-codes
   `qwen35::{config_from_hfq, load_weights, forward_scratch}`. Generalize to
   dispatch on `arch_id` so each family's forward runs (per-arch shim; the
   runner only needs argmax + KV growth per arch).

Then add the family to `ARCHS` in `tests/fixture-golden-gate.sh` and `--record`
its baselines (×format axis × gpu-arch fleet — see P4). Order by blast radius:
`llama`/`qwen2` (simplest dense, shared kernels) → `gemma3` → `deepseek4`/
`minimax`/`lfm2moe` (MoE/hybrid, reuse the arch-6 combine-pin pattern) → the VL
trio last (need the image stub).

## Deterministic MoE-down reduction (reconsider the atomicAdd default)

The fast MoE-down combine accumulates expert contributions via fp32
`atomicAdd` into the residual (`gemv_hfq4g256_moe_down.hip:154`,
`kernels.rs:3751`). FP add is non-associative, so undefined atomic ordering
makes the result **non-deterministic to ~ULP run-to-run on a fixed binary**
(documented in-kernel). Usually benign — but the residual feeds the *next
layer's MoE router top-k*, a discrete selection. On near-tied routing logits a
last-bit wobble can **flip expert selection and diverge macroscopically** from
that token on. So determinism here is a correctness/repro concern, not just a
test-harness annoyance.

Decision to make: should the deterministic path be the **default**, not just a
test variant? Options, best first:
- **Fixed-point / integer atomicAdd accumulation.** Integer add is exact and
  order-independent ⇒ deterministic regardless of atomic order, at ~atomics
  speed; round once on the final convert-back. This is the only scheme that
  *guarantees* determinism while keeping atomics. **Fidelity caveat:** the
  fixed-point accumulator must cover the dynamic range — when contribution
  exponents are far apart, a flat fixed-point grid loses the small terms, so
  size the integer width for the worst-case range or use a per-row/block
  shared-exponent (the absolute LSB position is well-defined in fixed-point,
  unlike float — which is exactly why post-fp32-sum "trim the bottom LSB" is
  unsound: the LSB's absolute position moves with the magnitude, and a reorder
  can straddle any trim boundary, so trimming only lowers the flip *rate*, it
  is not a guarantee).
- **Fixed summation order** — the in-tree `moe_down_combine_k8_batched`
  (expanded per-expert outputs + ordered sum). Guaranteed deterministic; costs
  an expanded scratch buffer + a combine pass. Already exists; make it the
  default for repro/test and any router-feeding site.
- Keep fast atomicAdd only where downstream is provably linear/tolerant.

Bench the fixed-point variant; if it's near-zero perf cost it should likely be
the global default. Pairs with the tiny-fixture golden/determinism gate above.

### Router-margin tuning of the tiny MoE *fixture* (DEFERRED → needs hipfire-finetune)

NOTE: this is deferred until the **hipfire finetune tool** lands — margin
tuning is an optimization step the finetune tool is the natural home for.
Until then, stabilize the MoE fixture golden by **pinning the deterministic
combine** (the first option below). Recorded here so the approach isn't lost.

A fixture-construction trick (this is about the tiny golden model, NOT a
production model technique): when generating the tiny MoE fixture, nudge the
random-init router weights so that — for the fixed test prompt — every token's
top-k expert selection lands with a **comfortable margin** (selected vs.
dropped experts well separated, "near the middle" of the decision region). Then
no routing decision is near a flip boundary, so the MoE-down atomicAdd ULP
noise cannot cascade into an expert swap, and the fixture's *token* output is
stable run-to-run **on the production fast path** — no need to pin the
deterministic combine for the fixture's golden.

This gives two independent ways to make the MoE fixture golden stable; pick per
goal:
- **Pin the deterministic combine** → byte-exact *logits* golden; tests the
  deterministic kernel specifically.
- **Router-margin-tune the fixture** (this) → token-exact golden that exercises
  the **default atomicAdd fast path** (only benign sub-flip ULP noise remains).
  For full token stability also give the fixture's final lm_head argmax a
  margin on the test prompt. Cheaper to run and tests what production uses.

(A production router-margin regularizer — hardening real routing against quant
error — is a *separate* idea; park it under the finetune tool only if it earns
its own motivation, not as part of this fixture work.)

---

## PFlash drafter training — progress reporting + checkpoint/resume (2026-06-18)

The drafter trainer (`crates/hipfire-train/examples/pflash_drafter_train.rs`,
plan `docs/plans/2026-06-18-pflash-qat-drafter.md`) currently runs as a
fire-and-forget loop. Two ergonomics gaps surfaced while running it:

- **Progress reporting.** Label capture (16 × 3B target forwards) and the long
  epoch loop are silent for minutes. Shipped a per-chunk `captured i/N` line; still
  want a per-epoch (or every-K-step) line with loss + a wall-clock / ETA estimate,
  and a `--quiet`/`--verbose` knob. Consider a tiny shared progress helper so other
  training examples (`overfit_supra50m`, `recovery_ft_supra50m`) can reuse it.

- **TUI dashboard (eventual).** Build a proper `ratatui` dashboard for the trainer
  (deps already added: `ratatui` 0.29 + `crossterm` 0.28 in hipfire-train, matching
  hipfire-tui). Target panels: capture progress + ETA; live loss + eval-Spearman
  sparkline vs the shallow bar; per-epoch/step timing + overall ETA; current
  hyperparams (τ/lr/epochs) and checkpoint status. Should degrade gracefully to the
  plain-line output when stdout isn't a TTY (CI / piped logs). Sits on top of the
  progress-helper above. Not urgent — land after the P3 result and the
  checkpoint/resume round-trip are solid.
- **Checkpoint / resume.** Long runs can't currently be stopped and continued.
  Add: periodic checkpointing of the drafter weights + AdamW moment buffers
  (m/v/t) + RNG/epoch position to a `.hfq`-style or simple binary artifact, and a
  `--resume <path>` that reloads them and continues. Also cache the captured
  mid-layer labels (they're deterministic per corpus+target) so a resume skips
  the expensive recapture. Graceful SIGINT → checkpoint-then-exit would be ideal.

Both are training-harness ergonomics, not blockers for the P3 result, so they're
parked here. Revisit once P3 shows the drafter beats the shallow bar and we move
to longer real-target (qwen) runs where these actually bite.

## Circle back: QTIP recovery export — norms-only quality vs Path B (2026-06-17)

Phase 3 export bridge (`hipfire-train` → daemon-servable qtip3 `.hfq`) is being
built on **Path A** (layernorm-only recovery → patch tuned norms into a qtip3
`.hfq`; lossless, codes untouched). Validated: norms-only recovery rescues
coherence (2-bit Supra-50M: corpus KL 1.58 → **0.144**, degenerate loop →
grammatical prose, from only ~12.8k trainable params).

**The tradeoff to revisit:** norms-only floors ~6× higher KL than **LoRA+norms**
(0.144 vs **0.025**). Path A ships that lossless-but-lower-capacity result. If
the served quality isn't good enough, evaluate **Path B** (merge LoRA into the
base, re-quantize to qtip3) and **measure how much of the 0.025 survives
re-quantization** — Path B trades a re-quant loss for LoRA's extra capacity.
Also possible: **Path C** (LoRA fp sidecar, daemon applies it — needs new daemon
support, no re-quant loss).

Decision deferred; Path A v1 ships first. See
`docs/plans/2026-06-17-hipfire-train-phase3-hfq-export.md`.
