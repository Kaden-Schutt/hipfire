# STEP-005 production SuperOp inventory

Date: 2026-08-27
Authority: `.agent-progress/device-mesh-refactor-tracker.md` STEP-005

| Family | Mode | Production entry | SuperOp route | Default | Hand/state oracle | Replacement owner | Status |
|---|---|---|---|---|---|---|---|
| Gemma4 | Single | `forward_scratch_inner` | `forward_scratch_inner_lowered` -> `run_layer_program` | on | `sliding_layer_decode` / `full_layer_decode` | Gemma4 increment | open — fixture-ready |
| LFM2 | Single | `decode_step_layers_and_head` | `decode_step_layers_and_head_lowered` -> `run_layer_program` | on | direct layer loop with capture | LFM2 increment | open |
| MiniMax | Single | `decode_step_body` | `decode_step_body_lowered` -> `run_layer_program` | on | direct attention + sealed MoE loop | MiniMax increment | open |
| Qwen35 | Single | `forward_scratch_layers` | `forward_scratch_layers_lowered` -> `run_layer_program` | on when no hidden ring or mRoPE | direct hybrid/DeltaNet loop | Qwen35 Single increment | open |
| Qwen35 | EP | `qwen35::ep_batch::forward_ep` | `run_layer_program_ep` | on | emulated EP oracle | Qwen35 EP increment | open |
| DeepSeek4 | Single | `decode_step_body` | `decode_step_body_lowered` -> `run_layer_program` | on | direct MLA + sealed MoE loop | DeepSeek4 increment | open |
| DeepSeek4 | EP | `deepseek4::ep::forward_ep` | `run_layer_program_ep` | on — admitted gfx1201 MQ2R TP3/TP4 graph route only; else direct non-SuperOp | emulated EP oracle (`ep_deepseek4` / pinned `DS4_EP2_FNV`) | DeepSeek4 EP increment | open |

## Row verification against current source (2026-08-27, base `2743acf2`)

Every row below was verified in the current worktree source before being
retained. Line numbers are exact at this base commit.

### Gemma4 / Single — VERIFIED

- Production entry `forward_scratch_inner` — `crates/hipfire-arch-gemma4/src/lowered.rs:2710`; the
  gate `if forward_lowered_enabled() { return forward_scratch_inner_lowered(...) }` sits at `:2721`.
- Lowered route `forward_scratch_inner_lowered` — `:5818`; calls
  `superop::run_layer_program(gpu, &ctx, &program, &mut bind)` at `:5846`.
- Default ON — `forward_lowered_enabled()` at `:5806-5813`:
  `std::env::var("HIPFIRE_FORWARD_LOWERED").ok().as_deref() != Some("0")` with the comment
  "Default ON (byte-parity validated 2026-06-08)". (The module header comment at `:5048-5051`
  still says "default OFF" — stale text; the code is the authority.)
- Hand/state oracle — hand arms in `forward_scratch_inner` call `sliding_layer_decode`
  (`:2742`, defined `:2858`) and `full_layer_decode` (`:2746`, defined `:3301`).
- Fixture state: READY (see "Gemma4 fixture state" below) → `fixture-ready` recorded in the
  row; pre-change hand-route baseline md5s are recorded below.

### LFM2 / Single — VERIFIED

- Production entry `decode_step_layers_and_head` — `crates/hipfire-arch-lfm2moe/src/forward.rs:322`;
  gate at `:343`: `if lfm2_forward_lowered_enabled() && capture.is_none()` — the oracle-dumper
  capture path forces the hand loop.
- Lowered route `decode_step_layers_and_head_lowered` — `:1128`; calls
  `superop::run_layer_program(...)` at `:1149`.
- Default ON — `lfm2_forward_lowered_enabled()` at `:1114-1123`: env `!= Some("0")`; comment
  "DEFAULT ON as of 2026-06-07 — fleet byte-parity validated (k9lin gfx1100 / hiptrx gfx1201 /
  hipx gfx1151, lowered == hand token-text md5 754a38b5…)".
- Hand/state oracle — the direct layer loop (mixer/FFN + final norm + lm_head) inside
  `decode_step_layers_and_head` (`:347`+), reachable with `HIPFIRE_FORWARD_LOWERED=0` or capture.

### MiniMax / Single — VERIFIED

- Production entry `decode_step_body` — `crates/hipfire-arch-minimax/src/forward.rs:227`; gate at
  `:258`: `if minimax_forward_lowered_enabled() && capture.is_none()`.
- Lowered route `decode_step_body_lowered` — `:1063`; calls `superop::run_layer_program(...)` at
  `:1083`.
- Default ON — `minimax_forward_lowered_enabled()` at `:1049-1057`: env `!= Some("0")`; comment
  "DEFAULT ON as of 2026-06-07 — hipx/gfx1151 byte-parity validated (lowered == hand token-text
  md5 2a46c35e…)".
- Hand/state oracle — the direct attention + sealed MoE loop inside `decode_step_body` (`:262`+);
  the MoE arm runs the SAME manifest-derived sealed program
  (`minimax_moe_single_step` → `execute_lowered_moe` Single) as the super-op `Moe` handler.

### Qwen35 / Single — VERIFIED

- Production entry `forward_scratch_layers` — `crates/hipfire-arch-qwen35/src/qwen35/forward.rs:2787`;
  gate at `:2809`: `if forward_lowered_enabled() && hidden_rb.is_none() && mrope.is_none()` — i.e.
  "on when no hidden ring or mRoPE"; VL (3D mrope) and hidden-ring spec capture always take the
  hand arms (`:2803-2808` rationale comment).
- Lowered route `forward_scratch_layers_lowered` — `:5627`; calls `superop::run_layer_program(...)`
  at `:5697`.
- Default ON — `forward_lowered_enabled()` at `:5283-5289`: env `!= Some("0")`; comment "DEFAULT ON
  as of 2026-06-07 — validated byte-identical via fleet decode byte-parity (RDNA3 k9lin / RDNA4
  hiptrx / RDNA3.5 hipx, dense + MoE) + full coherence battery (13 cases)".
- Hand/state oracle — the direct hybrid/DeltaNet loop (hand arms incl. mrope branch) inside
  `forward_scratch_layers` (`:2813`+).

### Qwen35 / EP — VERIFIED

- Production entry `qwen35::ep_batch::forward_ep` — `crates/hipfire-arch-qwen35/src/qwen35/ep_batch.rs:2178`
  (re-exported at `crates/hipfire-arch-qwen35/src/qwen35.rs:29-32`); per-layer
  `hipfire_runtime::ep::run_layer_program_ep(gpus, binds, partials, &program, dim)` at `:2295`.
- Default: the SuperOp EP executor is used unconditionally by this entry (no toggle) → "on".
- Hand/state oracle — the emulated EP oracle: `crates/hipfire-runtime/examples/ep_decode_parity.rs`
  (built only under the non-default `emulated-ep2-harness` feature — runtime `Cargo.toml:146-148`),
  which drives the test-only emulated EP2 harness (`src/ep2_harness.rs`, `src/store/store_ep2.rs`,
  feature `emulated-ep2-harness`, never default): two logical expert-ownership ranks over one GPU
  (stride-2 `EmulatedExpertPartitionPlan`), comparing baseline vs EP2 logits/tokens.
- Precision note: the daemon's live Qwen35 EP continuous-batch path
  (`Qwen35DecodeBatchEpState::forward_tick`, ep_batch.rs:1670, driven by
  `drive_qwen35_ep_continuous_batch` in `crates/hipfire-generate/src/batch.rs:3316`) runs each
  layer through `forward_batch_chunk_impl` + `all_reduce_sum_f32_peer_rooted_leased` — it does NOT
  call `run_layer_program_ep`. It is a separate EP batch route (peer-rooted leased reduce), not a
  SuperOp consumer, and is not an inventory row.

### DeepSeek4 / Single — VERIFIED

- Production entry `decode_step_body` — `crates/hipfire-arch-deepseek4/src/forward.rs:6314`; gate at
  `:6329`: `if ds4_forward_lowered_enabled() { return decode_step_body_lowered(...) }`.
- Lowered route `decode_step_body_lowered` — `:7233`; calls `superop::run_layer_program(...)` at
  `:7256`.
- Default ON — `ds4_forward_lowered_enabled()` at `:7219-7227`: env `!= Some("0")`; comment
  "default ON, matching qwen35/lfm2/minimax; set =0 to fall back to the hand loop. Flipped on after
  hipx byte-parity in both plain AR and MTP spec-decode modes."
- Hand/state oracle — the direct MLA + sealed MoE loop inside `decode_step_body` (`:6333`+).

### DeepSeek4 / EP — VERIFIED

- Production entry `deepseek4::ep::forward_ep` — `crates/hipfire-arch-deepseek4/src/ep.rs:44`
  (re-exported at `crates/hipfire-arch-deepseek4/src/forward.rs:7265`; daemon caller
  `crates/hipfire-generate/src/qwen.rs:637,791`). When `tp_graph_admitted` (`:93-110`: n∈{3,4},
  MQ2R, gfx1201, peer access, TP-size==n, graph signals ready) it routes to `forward_ep_tp_graph`
  (`:114`) → `forward_ep_tp_graph_body` (`:146`), which calls
  `hipfire_runtime::ep::run_layer_program_ep(...)` at `:172`.
- Default: on the admitted gfx1201 MQ2R TP3/TP4 graph route the SuperOp EP executor is the default
  execution; the non-admitted arm `forward_ep_direct` (`:361`) runs the sealed parallel executor
  (`execute_lowered_moe`, `MoeExecutionTarget::Parallel`) and does NOT touch `run_layer_program_ep`.
- Hand/state oracle — emulated EP oracle: `crates/hipfire-arch-deepseek4/examples/ep_deepseek4.rs`
  with the source-pinned `DS4_EP2_FNV = 0x26a13602bedf9926` (`ep_deepseek4.rs:398`:
  `assert_eq!(fnv, DS4_EP2_FNV, "output drifted from pinned D2a hash")`), run over emulated ranks
  (`HIPFIRE_EMULATE_GPUS=2`) with the peer all-reduce path (`HIPFIRE_EP_PEER_ALLREDUCE_DECODE=1`,
  RCCL-free on boxes without librccl) under `HIPFIRE_DETERMINISTIC=1`.
- Cross-task warning (unverified): tracker HW-001 names the `ep_decode_parity` committed-token
  hash as the DeepSeek4 EP oracle, but `ep_decode_parity` is a Qwen35-only runtime example
  (feature `emulated-ep2-harness` wiring `hipfire-arch-qwen35/emulated-ep2-harness`, default
  prompt `benchmarks/prompts/qwen35_moe_ep_parity.txt`; runtime `Cargo.toml:146-148`). That
  reference cannot be a DeepSeek4 oracle; it is flagged here for the tracker owner and no
  DeepSeek4 claim in this inventory derives from it.

## Shared definitions (not production callers)

- `crates/hipfire-dispatch/src/pipeline/superop.rs` — the substrate: `SuperOp` (`:134`),
  `SuperOpKind` (`:140`), `LayerProgram` (`:178`), `LoweredForward` (`:186`), `lower_walk`
  (`:209`), `lower_layer` (`:255`), `ForwardBindings` trait (`:287`, incl. `run_moe_ep` `:350`,
  `ep_add_into_residual` `:369`, `supports_tp_peer_hc4` `:428`, `supports_tp_peer_hc3` `:434`),
  `dispatch_super_op` (`:495`), and the executor `run_layer_program` (`:523`).
- `crates/hipfire-dispatch/src/pipeline/mod.rs:23` — `pub mod superop;` (the export).
- `crates/hipfire-runtime/src/ep.rs` — the EP executor `run_layer_program_ep` (`:112`) +
  `ensure_rank_streams` (doc `:6-32` describes the zero → owned-experts → all-reduce → add-back
  contract; attention-TP hooks `tp_peer_hc3/hc4_admitted` `:79-97`).

## Mechanical sweep (harness Grep, not shell grep)

Pattern: `run_layer_program_ep|run_layer_program|ForwardBindings|LayerProgram|SuperOpKind|HIPFIRE_FORWARD_LOWERED`
over the whole repo. Every production result maps to exactly one inventory row:

| # | Production result | Row |
|---|---|---|
| 1 | `gemma4/lowered.rs` `forward_scratch_inner` → `forward_scratch_inner_lowered` → `run_layer_program` (`:5846`) | Gemma4 Single |
| 2 | `lfm2moe/forward.rs` `decode_step_layers_and_head` → `decode_step_layers_and_head_lowered` → `run_layer_program` (`:1149`) | LFM2 Single |
| 3 | `minimax/forward.rs` `decode_step_body` → `decode_step_body_lowered` → `run_layer_program` (`:1083`) | MiniMax Single |
| 4 | `qwen35/forward.rs` `forward_scratch_layers` → `forward_scratch_layers_lowered` → `run_layer_program` (`:5697`) | Qwen35 Single |
| 5 | `qwen35/ep_batch.rs` `forward_ep` → `run_layer_program_ep` (`:2295`) | Qwen35 EP |
| 6 | `deepseek4/forward.rs` `decode_step_body` → `decode_step_body_lowered` → `run_layer_program` (`:7256`) | DeepSeek4 Single |
| 7 | `deepseek4/ep.rs` `forward_ep` → `forward_ep_tp_graph_body` → `run_layer_program_ep` (`:172`) | DeepSeek4 EP |

Additional results are classified below; none is a production SuperOp caller and none gets a row.

### Related production routes that are NOT SuperOp consumers (no row)

- `crates/hipfire-runtime/src/llama.rs` (`llama_forward_lowered_enabled` `:4406-4418`, default ON)
  and `crates/hipfire-arch-qwen2/src/qwen2.rs` (`qwen2_forward_lowered_enabled` `:2054-2067`,
  default ON) share the `HIPFIRE_FORWARD_LOWERED` env name but implement their own lowered decode
  (`llama_kv_write_attend` / `dense_forward`); neither calls `run_layer_program` nor uses the
  SuperOp substrate (STEP-004 inventory rows 1 and 7 classify both as Step-complete). Out of
  STEP-005 scope.
- Sealed-parallel EP/TP routes (Step-backed `execute_lowered_moe`, not `run_layer_program_ep`):
  `minimax/forward.rs` `forward_ep`/`forward_tp` (`:2162`/`:2332`, via `minimax_ep_moe_step`),
  `deepseek4/ep.rs` `forward_ep_direct` (`:361`) and `deepseek4/mtp.rs` `mtp_forward_ep` (`:558`),
  and the Qwen35 EP batch `forward_tick` (see precision note above). MiniMax EP has no SuperOp
  route and therefore no inventory row.

### Tests (SuperOp symbols only in test code)

- `hipfire-dispatch/src/pipeline/superop.rs` `mod tests` (`:535+`): `lower_walk_*` CPU-pure unit
  tests (collapses/all-unfused/single-cluster/zero-span).
- `hipfire-arch-qwen35/src/qwen35/forward.rs` `mod tests` (`:5769+`): `lowered_fullattn_program_shape`
  and shape tests asserting `LayerProgram` mirrors the hand-arm op sequence.
- `hipfire-arch-minimax/src/forward.rs` tests (`:2495+`): `SuperOpKind::{Attend, Moe}` shape test;
  `minimax_single_old_vs_lowered_program_shape` (`:3102`): genuine old-vs-lowered differential
  against the test-only legacy oracle.
- `hipfire-arch-lfm2moe/src/forward.rs` tests (`:1262+`): `lfm2_variant_shapes`.
- `hipfire-arch-deepseek4/src/forward.rs` tests (`:18020+`): program-shape tests using
  `SuperOpKind::{Attend, Moe}`.

### Examples (never production routes)

- `crates/hipfire-runtime/examples/ep_decode_parity.rs` — **Qwen35-only** emulated-EP2 parity
  driver; built only with the non-default `emulated-ep2-harness` feature (which wires
  `hipfire-arch-qwen35/emulated-ep2-harness`; runtime `Cargo.toml:146-148`). Not a DeepSeek4
  fixture.
- `crates/hipfire-arch-deepseek4/examples/ep_deepseek4.rs`, `ep_dspark_topology_probe.rs`,
  `tp_deepseek4.rs`, `ds4_tp_longctx_capacity.rs` (runtime example), `ds4_longctx_probe.rs`
  (sets `HIPFIRE_FORWARD_LOWERED=0`), `ds4_prod_vs_parent_trace.rs` (sets
  `HIPFIRE_FORWARD_LOWERED=0`).
- `crates/hipfire-arch-minimax/examples/ep_minimax.rs`.

### Historical documentation and validation scripts (not source of truth)

- `.agent-memory/notes/device-mesh-pivot-execute-steps-spine.md` (records the 2026-07-07 SuperOp
  substrate deletion and its 2026-08-26 re-absorption context), `device-mesh-next-followups.md`,
  `device-mesh-review-findings-2026-07-10.md`, `godstruct-collapse-handover-2026-07-11.md`,
  `pd-decompose-*.md`, `ep-minimax-stopseq-kv-overcount.md`.
- `.agent-progress/step-004-inventory.md` (STEP-004 predecessor inventory; superseded for the
  execution spine by this file per the tracker's 2026-08-26 reconciliation),
  `.agent-progress/device-mesh-phase0.md`, `.agent-progress/device-mesh-status.md`,
  `.slim/deepwork/*`.
- `docs/design/2026-06-13-greenfield-engine-architecture.md` (pre-merge EP shape; its "qwen35 EP
  is substrate-only, not reachable from the daemon" claim predates the EP batch admission),
  `docs/design/lfm2moe-gfx1201-{decode,prefill}-architecture.md`, `docs/plans/gemma4_forward_as_pipeline*.md`,
  `docs/plans/qwen2-dots-ocr-forward-lowering.md`, `docs/plans/ship6-{substrate-ep,deepseek4-ep}.md`,
  `docs/plans/daemon-ep-wiring.md`.
- `docs/REDLINE.md:879-880`, `docs/env-vars.md:151-152,524-526`, `docs/admissions.yml:50-51`
  (LFM admission row pins `HIPFIRE_FORWARD_LOWERED=0`).
- `scripts/forward-lowered-parity.sh` — committed parity gate: runs the daemon twice
  (`HIPFIRE_FORWARD_LOWERED=0` vs default) and hard-fails on committed-token-stream divergence.
- Doc comments only: `crates/hipfire-hardware/src/mesh.rs:146-147` (DeviceMesh::single), 
  `crates/hipfire-dispatch/src/pipeline/steps.rs:2029-2032` (historical `run_layer_program_mesh`
  reference).

## Gemma4 fixture state (Step 3 follow-up)

The following user-provided facts are authoritative and are recorded verbatim. Per the brief, no
artifact size, metadata, SHA-256, or MD5 checks were rerun or recomputed.

- `~/.hipfire/models/gemma4-12b.mq4`
  - size: `8,914,591,328` bytes
  - arch: `13 / gemma4_unified`
  - tensors: `666`
  - SHA-256: `4ceb57b558275776680b9acd78fa4e058abefa994a901eb5253654c51e9981c3`
  - MD5: `a1419f8a5ddbbe70ad5fa7e6a3c2b73a`
- `~/.hipfire/models/gemma4-26b-a4b.mq4`
  - size: `15,242,780,732` bytes
  - arch: `13 / gemma4`
  - tensors: `8,277`
  - SHA-256: `6f83d448d4bc089aa18debd6601d34c6fd3ce0bab96ee8519d08f6d65121df63`
  - MD5: `182eafae7b25386ac9f9b73ce77b1a88`

The committed prompt digest is preserved:

- `benchmarks/prompts/merge_sort_thinking_off.txt` — present, git-tracked (commit `f38918e56`),
  SHA-256 `d671894964cb957643fcb961151f3d1b407cb5c206766eaed60e9c593e6ed9d0` (the committed
  digest).

Both canonical model paths were used directly for the baselines below; no substitute artifacts or
prompts were used. Fixture state is `fixture-ready`.

## Gemma4 pre-change hand-route baselines (Steps 4–5)

### Oracle build

Exact command:

```bash
cargo build --release --locked -p hipfire-arch-gemma4 --example infer_gemma4
```

Result: exit status `0`; the existing release `infer_gemma4` oracle built successfully. The build
emitted compiler warnings only; no build error occurred.

### Exact dense hand-route baseline

Command:

```bash
export GEMMA4_DENSE="$HOME/.hipfire/models/gemma4-12b.mq4"
export GEMMA4_MOE="$HOME/.hipfire/models/gemma4-26b-a4b.mq4"
HIPFIRE_FORWARD_LOWERED=0 HIPFIRE_GEMMA4_GRAPH=0 HIPFIRE_GEMMA4_EAGLE=0 \
  target/release/examples/infer_gemma4 --model "$GEMMA4_DENSE" \
  --token-ids 2,9259,236888,575,106 --max 32 --rep-pen 1.0 \
  >"$HOME/hipfire-step005/gemma4/baseline/dense-hand.log" 2>&1
```

Result: exit status `0`; log persisted at
`$HOME/hipfire-step005/gemma4/baseline/dense-hand.log`. The run reported `decoded 32 tok in
1.45s (22.1 tok/s)` and emitted these 32 continuation IDs:

```text
[45518, 107, 101, 1509, 5724, 1133, 611, 2473, 735, 3265, 496, 11409, 3618, 653, 496, 116896, 167043, 236775, 575, 236775, 1018, 769, 108, 3910, 740, 564, 1601, 611, 3124, 236881, 1637, 611]
```

### Exact MoE hand-route baseline

Command:

```bash
export GEMMA4_DENSE="$HOME/.hipfire/models/gemma4-12b.mq4"
export GEMMA4_MOE="$HOME/.hipfire/models/gemma4-26b-a4b.mq4"
HIPFIRE_FORWARD_LOWERED=0 HIPFIRE_GEMMA4_GRAPH=0 HIPFIRE_GEMMA4_EAGLE=0 \
  target/release/examples/infer_gemma4 --model "$GEMMA4_MOE" \
  --token-ids 2,9259,236888,575,106 --max 32 --rep-pen 1.0 \
  >"$HOME/hipfire-step005/gemma4/baseline/moe-hand.log" 2>&1
```

Result: exit status `0`; log persisted at
`$HOME/hipfire-step005/gemma4/baseline/moe-hand.log`. The run reported `decoded 32 tok in
0.43s (74.1 tok/s)` and emitted these 32 continuation IDs:

```text
[236772, 79770, 11542, 237323, 236772, 3643, 569, 68179, 569, 569, 174759, 236811, 121511, 242467, 8946, 1082, 239858, 16314, 498, 239858, 569, 236772, 237122, 1092, 236772, 8155, 231216, 236772, 236772, 236804, 236772, 36283]
```

### Baseline log hashes and runtime verdict

Exact command:

```bash
md5sum "$HOME/hipfire-step005/gemma4/baseline/"*.log
```

Exact result:

```text
9a90ac8344eeb4024822bde3fbda5096  /home/bjoern/hipfire-step005/gemma4/baseline/dense-hand.log
4e7a48dd1ef5272324d8314ffa30f0e2  /home/bjoern/hipfire-step005/gemma4/baseline/moe-hand.log
```

Inspection verdict: PASS — both exact canonical models loaded and ran in separate processes with
exit status `0`, each produced 32 continuation IDs, and neither log contains a panic or invalid
access. The dense log notes on-demand recompilation for pre-compiled blobs without hash files;
that did not prevent a successful baseline.

The prior fixture blocker is fully resolved for Gemma4; the remaining inventory rows and source
verification are unchanged.

## Out of scope (later tasks)

- Muse/Glimmer bespoke decoder — STEP-006 (no SuperOp/Step route; CAP refuses PP/TP/EP).
- Cohere2-MoE, dots.ocr, Qwen35-VL — no SuperOp consumers; STEP-006/VL-001 scope per tracker.
- Gemma4 prefill (`forward_prefill_chunk`) and the `sliding_layer_decode_impl` /
  `full_layer_decode_impl` `stop_before_moe` arms are hand-path prefill helpers, not decode
  SuperOp consumers.
