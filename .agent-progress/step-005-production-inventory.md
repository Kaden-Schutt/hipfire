# STEP-005 production SuperOp inventory

Date: 2026-08-27
Authority: `.agent-progress/device-mesh-refactor-tracker.md` STEP-005

| Family | Mode | Production entry | SuperOp route | Default | Hand/state oracle | Replacement owner | Status |
|---|---|---|---|---|---|---|---|
| Gemma4 | Single | `forward_scratch_inner` | `forward_scratch_inner_lowered` -> `run_layer_program` | on | `sliding_layer_decode` / `full_layer_decode` | Gemma4 increment | open — fixture-ready; Task 7 serve/lifecycle blocked |
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

## Task 7 lifecycle and user-facing evidence (2026-08-27)

Task 7 is **BLOCKED**; this row is not `complete`. Evidence is local at
`/home/bjoern/hipfire-step005/gemma4/final/`. No implementation files were changed.
- Serving prerequisite command `cargo build --release --locked -p hipfire-cli -p hipfire-daemon`
  exited `0` with compiler warnings only; it produced `target/release/daemon` used by
  the isolated runs. No formatter, linter, project-wide suite, or unrelated build was run.

- Task 2–6 commits: `97a9ac0b4`, `ec3775afc`, `59b923437`, `44a5d313d`,
  `a2fbba955`, `37b5f7a2e`, `321c87370`.
- Canonical model digests remain the pinned values above:
  dense SHA-256 `4ceb57b558275776680b9acd78fa4e058abefa994a901eb5253654c51e9981c3`,
  MoE SHA-256 `6f83d448d4bc089aa18debd6601d34c6fd3ce0bab96ee8519d08f6d65121df63`.
- Maintained E-series reproduction, with
  `E2B_MODEL=$HOME/.hipfire/models/gemma4-eseries/gemma4-e2b-it-pr439-q8.hfq`,
  `E4B_MODEL=$HOME/.hipfire/models/gemma4-eseries/gemma4-e4b-it-pr439-q8.hfq`,
  exited `2`: `missing model: /home/bjoern/.hipfire/models/gemma4-eseries/gemma4-e2b-it-pr439-q8.hfq`.
  The E-series directory is absent; this arm is fixture-blocked. Raw output:
  `final/eseries-command.log`.
- Direct Step-only decode (`infer_gemma4`, exact
  `2,9259,236888,575,106`, `--max 32`, `--rep-pen 1.0`,
  `HIPFIRE_GEMMA4_GRAPH=0`, `HIPFIRE_GEMMA4_EAGLE=0`) exited `0` for both models:
  dense FNV `0xa081658763517110` (`final/dense-decode.log`) and MoE FNV
  `0x831756562b0ab110` (`final/moe-decode.log`). These are numerical records only.
- The prior Task 5 text inspection remains malformed:
  dense `111-11111111111111111111  1111111`; MoE
  `-나-111 資랑0나�서와 own나서bed much1 own1 much much나 de1 own way own ownら`.
  No semantic-quality claim is made from the matching counters.
- `serve_harness.py` commands as written in the brief use unsupported positional
  `battery`/`chain` and exited `2` (`final/dense-battery-brief.log`). Equivalent isolated
  `--mode` runs used the newly built daemon (`target/release/daemon`, observed MD5
  `4584d2900ead83aea5e96df87e6c01fc`), unique ports, and `--home` directories.
- Applicable `serve_harness.py` built-in battery prompt MD5s (the exact prompt bytes
  sent in battery and first chain turns) were observed as: code
  `43ca0d15712d3dfb777b51ae76d8fd5f`; reason
  `640e0fd4f55996cb175a422f0a12cef5`; factual
  `8f66b4c97988825bd8e7840aaf44357e`; prose
  `8fe0ad36f61bcf4992cc9df81cdf3817`; instruct
  `8bed8e2d056dc1d47dccae9d32dbecf4`.
  Dense battery/chain exited `0` and showed coherent visible starts, but every row had
  `finish=None`, `gen=0`, `ctx=0`, `cached=0` with null timing/usage; this is an invalid
  user-facing terminal contract, not a pass (`final/dense-{battery,chain}.log/.json`).
  MoE battery/chain exited `0` but all five rows were empty; the daemon reported
  `gemma4 lowered/MoE generate not yet wired on this build (eager dense only) ...`
  for each request (`final/moe-{battery,chain}.log/.json`). MoE production generation
  is explicitly unwired.
- Interactive native daemon lifecycle sequence (isolated homes, exact load/generate/
  unload/diag wire records) completed four cycles per fixture. Dense post-unload free
  memory was `99961, 99961, 99961, 99961 MB`; each generation emitted `Paris` and a
  `done` event but omitted `finish_reason`. MoE post-unload free memory was
  `99826, 99824, 99822, 99820 MB`; generation errored every cycle. The MoE decrease
  is recorded as raw `2 MB/cycle` drift only: no monotonic-leak pass or new threshold
  is claimed. Raw records: `final/dense-lifecycle.log`, `final/moe-lifecycle.log`.
- Dense serve daemon logs observed `[gemma4 hipGraph] captured decode forward — 767
  kernarg blobs retained`; no replay parity is claimed. MoE produced no capture because
  generation is unwired. The pre-existing user-owned daemon PID `819407` was not
  stopped, signaled, or reused.

Because E-series artifacts are absent, dense serving terminals are invalid despite
coherent visible text, MoE serving is empty/unwired, dense lifecycle terminals omit
`finish_reason`, and MoE lifecycle free memory decreases by 2 MB each cycle, the row
remains open.
## Task 8 remediation validation (2026-08-28, validation-only)

**Status: BLOCKED.** Current source revision is `2039883a5a394f60c6cd8c6f1a5587483ff66f0d`. No source implementation files were changed by this validation. Raw evidence is under `/home/bjoern/hipfire-step005/gemma4/remediation/`; the existing user-owned daemon PID `819407` was observed before validation and was not stopped, signaled, or reused.

### Source gates

- The exact scoped `rustfmt --edition 2021 --check --config skip_children=true` command from the Task 8 brief exited `1`; it reported formatting diffs in listed changed files including `hipfire-generate/src/dense.rs`, `hipfire-loader/src/lib.rs`, `hipfire-arch-gemma4/src/lowered.rs`, and `hipfire-cli/src/serve/complete.rs`. Raw output: `remediation/source/rustfmt-check.log`.
- The exact `cargo clippy -p hipfire-generate --lib -- -D warnings`, `cargo clippy -p hipfire-loader --lib -- -D warnings`, and `cargo clippy -p hipfire-arch-gemma4 --lib -- -D warnings` commands each exited `101` before target checking on the shared `hipfire-config/src/rocm.rs` baseline (`doc_overindented_list_items`, `redundant_guards`, and `obfuscated_if_else` under `-D warnings`). Raw output: `remediation/source/clippy-{generate,loader,gemma4}.log`.
- `cargo build` exited `0` (compiler warnings only). Raw output: `remediation/source/cargo-build.log`.
- `cargo test` exited `101`: 59 loader tests passed and one failed, `registry_tests::caps_and_route_tables_are_pinned`, because observed `semantic_contract_version` was `Some(2)` while the pinned expected value was `None`. Raw output: `remediation/source/cargo-test.log`.
- `cargo build --release --locked -p hipfire-cli -p hipfire-daemon` exited `0`. Release binary MD5s: `target/release/hipfire` = `fdab7ca9a2436ea795edcd6d9a09970b`; `target/release/daemon` = `0feb34a6fa99a0a42a1ba4f3c5b70e42`. Raw output: `remediation/source/release-build.log` and `binary-md5s.log`.

### Gate A — isolated HTTP terminal battery/chain

All runs used the release CLI/daemon, `HIP_VISIBLE_DEVICES=0`, `HIPFIRE_GEMMA4_GRAPH=0`, `HIPFIRE_GEMMA4_EAGLE=0`, isolated homes, and unique ports. Built-in prompt MD5s were code `43ca0d15712d3dfb777b51ae76d8fd5f`, reason `640e0fd4f55996cb175a422f0a12cef5`, factual `8f66b4c97988825bd8e7840aaf44357e`, prose `8fe0ad36f61bcf4992cc9df81cdf3817`, instruct `8bed8e2d056dc1d47dccae9d32dbecf4`.

- Dense battery (`gate-a/dense-battery.log/.json`) and dense chain (`dense-chain.log/.json`) exited `0`. Each row had a terminal `finish=length`, nonzero `ctx`/`gen` (battery `gen=64` on all five rows), `[gemma4-12b.mq4|off|{battery,chain} DONE]`, no premature-EOF error, `empty=0`, `attractor=0`, and `retrieval_miss=0`. Every row hit the explicit `max_tokens=64` cap (`runaway=5`), so these are terminal-contract observations, not a semantic-quality pass.
- MoE battery (`gate-a/moe-battery.log/.json`) exited `0`; all five rows had terminal `finish=length`, nonzero `ctx`/`gen=64`, `[DONE]`, no premature EOF, `empty=0`, and `attractor=0`, but decoded text is visibly malformed multilingual/repeated-token output.
- MoE chain (`gate-a/moe-chain.log/.json`) exited `1` after `[DONE]` summary because `attractor=1`; all five rows were `finish=length`, `gen=64`, and visibly malformed. This is a hard quality/serving blocker, not hidden by the outer summary.

### Gate B — direct/product parity, cache, and injected paths

- Current direct canonical fixed-token controls used `2,9259,236888,575,106`, greedy `--rep-pen 1.0`, `HIPFIRE_GEMMA4_GRAPH=0`, and `--max 32`. Dense direct IDs are the recorded canonical hand sequence and final-logit FNV `0x981d38723fe270af` from the temporary eager direct harness (`quality/dense-eager-direct-fixed.log`); MoE direct IDs/FNV are the accepted sequence and `0x831756562b0ab110` (`gate-b/direct-moe.log`).
- The committed ignored `gemma4_lowered_product_matches_accepted_direct_oracle` test passed (`gate-b/gemma4-lowered-product-oracle.log`), proving current lowered MoE product IDs/FNV equal the accepted canonical direct values.
- A raw-text dense eager direct/product comparison used identical tokenizer input `[2,818,5279,529,7001,563]` for `The capital of France is`. Direct generated 13 IDs and FNV `0xf2bc42133d8166dc` (`quality/dense-eager-direct-raw.log`); current daemon product trace sampled the same 13 IDs before EOS and its full step-13 logits FNV was `0xf2bc42133d8166dc` (`gate-b/dense-jsonl-product.json`, `dense-jsonl-trace/`). The trace self-compare was exact (`quality/dense-trace-self-compare.json`); the self-compare is diagnostic and is not substituted for an external HF reference.
- Eager dense and lowered MoE cache probes (`gate-b/{dense,moe}-cache-product.json`) both observed seed `cached_tokens=0`, related suffix `cached_tokens=4` with suffix-only prefill, unrelated request `cached_tokens=0`, and valid correlated `commit_ready`/`done`/`unloaded` envelopes.
- Eager dense and lowered MoE abort probes (`gate-b/{dense,moe}-abort-product.json`) both observed a correlated `abort`, terminal `finish_reason=aborted`, and a following related request with `cached_tokens=4`, demonstrating committed-prefix restoration after abort.
- Focused Gemma transaction tests passed: `gemma_` = 27/27; EOS tests = 2/2; invalid-shape/non-finite/out-of-range tests = 3/3; forward-failure rollback = 1/1; EAGLE settle-failure and invalid-settle-logits rollback = 1/1 each. Raw output: `gate-b/gemma-focused-tests.log`, `gemma-eos-tests.log`, `gemma-invalid-tests.log`, `gemma-forward-failure-test.log`, `gemma-eagle-settle-failure-test.log`, and `gemma-eagle-invalid-test.log`.

### Gate C — four-cycle lifecycle/ownership telemetry

- The direct daemon lifecycle driver completed four dense eager cycles (`gate-c/dense-eager-lifecycle.json`) and four dense lowered cycles (`dense-lowered-lifecycle.json`), each with load → generate (`finish=length`, `tokens=1`) → reset (`rolled_back=true`, `seq_pos=0`, `conversation_len=0`) → unload → post-unload diag. Dense eager post-unload free MB was `99428, 99427, 99427, 99428`; dense lowered was `99454, 99455, 99455, 99454`.
- Dense lowered telemetry emitted eight `[gemma4 alloc]` records. `owner_bytes` stayed exactly `10309160452`; unload `pool_bytes` stayed exactly `10309160448`; `graph_resident=false`, `graph_blob_count=0`, `module_count=19`, and freed labels were `kv_full,kv_sliding,scratch,weights`. The telemetry `cycle=0` field reflects no operator cycle env override; driver cycle labels are 1–4.
- The MoE lifecycle driver completed four cycles (`gate-c/moe-lifecycle.json`) with the same reset/unload envelope. Post-unload free MB was `99289` on all four cycles. Eight telemetry records held `owner_bytes=16554924808`, unload `pool_bytes=16554924804`, `graph_resident=false`, `graph_blob_count=0`, `module_count=22`, and the same complete freed-owner labels; `cycle=0` for the same reason.
- The canonical ignored lowered constructor fault matrix did **not** pass on this revision (`gate-c/gemma4-constructor-fault-matrix.log`): at injected `weights` stage it observed `baseline=104515829760`, `after=104445284352`, and `pool_before_drain=15206432256`, a `70,545,408`-byte free-device deficit. This is recorded as a source ownership blocker, not free-device variance.
- The ignored complete teardown ownership smoke passed (`gate-c/gemma4-teardown-ignored.log`); focused lowered telemetry tests passed 2/2 with one expected ignored GPU teardown test.

### Quality/reference and artifact disposition

- `quality/dense-eager-reference-matrix.log` ran the exact canonical dense input through eager sequential versus batched reference paths: B=1,4,8 passed, but B=2 failed the post-batch KV next-token argmax (`seq=2921`, `bat=236906`; next cosine `0.9975923`), so the available batched-reference matrix is **FAIL**.
- Current dense battery/chain decoded text is coherent at the visible prefix but all rows are cap-terminated; current MoE battery/chain text is visibly malformed. The old Task 5 malformed text remains historical evidence only; no semantic quality is inferred from counters/FNV. No local HF or higher-precision Gemma reference checkpoint was available (`quality/reference-availability.log`); historical Task 7 logs are not promoted to current evidence.

### E-series arm

- `scripts/reproduce-gemma4-eseries-parity.sh` was run only with the exact expected E2B/E4B paths and exited `2` on missing `/home/bjoern/.hipfire/models/gemma4-eseries/gemma4-e2b-it-pr439-q8.hfq`. The E-series directory and both exact files are absent (`eseries/eseries-command.log`, `eseries/fixture-state.log`); this remains an external fixture blocker.

Because source formatting/clippy/test gates, MoE quality/chain, the constructor fault matrix, the available batched-reference case, and E2B/E4B fixtures remain unresolved, the Gemma4 inventory row stays `open`; the original Task 7 verdict stays **BLOCKED**.

## Out of scope (later tasks)

- Muse/Glimmer bespoke decoder — STEP-006 (no SuperOp/Step route; CAP refuses PP/TP/EP).
- Cohere2-MoE, dots.ocr, Qwen35-VL — no SuperOp consumers; STEP-006/VL-001 scope per tracker.
- Gemma4 prefill (`forward_prefill_chunk`) and the `sliding_layer_decode_impl` /
  `full_layer_decode_impl` `stop_before_moe` arms are hand-path prefill helpers, not decode
  SuperOp consumers.
