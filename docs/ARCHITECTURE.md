# Architecture

How a `hipfire run` becomes tokens on the current modular tree. Runtime source
wins on conflict. Mutable command surfaces, model catalogs, and env inventories
live in their own owners ([`CLI.md`](CLI.md), [`MODELS.md`](MODELS.md),
[`env-vars.md`](env-vars.md)); this page maps crates, loaders, and data flow.

Truth-state labels follow [`INDEX.md`](INDEX.md). Implementation capability is
not product admission: a compiled path or runtime gate is not certification.

## Workspace map

Workspace members (`Cargo.toml`):

```text
crates/
├── hip-bridge/              # dlopen HIP FFI (libamdhip64)
├── hsa-bridge/              # thin public HSA queue/AQL helpers
├── rdna-compute/            # Gpu, kernel compile/cache, HIP launch families
├── radiowave/               # signal/scheduling support for the compute layer
├── saddle-core/             # the substrate: grammar, KV, caps, sampling policy
├── hipfire-runtime/         # arch-agnostic infra + Architecture trait; LLaMA forward still lives here
├── hipfire-loader/          # Carrier registry, load_model, LoadedModel, batch staging
├── hipfire-engine/          # arch-free serve engine: scheduler, terminal, emit, prompt
├── hipfire-generate/        # generation bodies: ar, qwen, dense, vision, batch, redline fixtures
├── hipfire-daemon/          # the product binary — [[bin]] name = "daemon", no required-features
├── hipfire-dispatch/        # dtype/arch family tables + MoE/attn pipelines
├── hipfire-dispatch-tests/  # dispatch coverage tests
├── hipfire-arch-*/          # per-family config/weights/state/forward (12)
├── hipfire-ds4-parent/      # DeepSeek-V4 parent/teacher tooling (name provisional)
├── hipfire-pflash/          # PFlash prefill compression — retained legacy research, default off
├── hipfire-quantize/        # CPU encoder (safetensors/GGUF → .mq* / .hfq*)
├── hipfire-detect/          # observational JSONL behavior detectors
├── hipfire-atlas/           # Kernel Atlas schema + emit helpers
├── hipfire-config/          # typed TOML config + migration
├── hipfire-registry/        # strict bundled/dynamic model registry
├── hipfire-client/          # daemon JSONL + OpenAI HTTP/SSE client
├── hipfire-cli/             # native operator and HTTP service
├── hipfire-tui/             # terminal UI
├── hipfire-reap/            # utility crate
├── saddle-lab/              # research examples and one-off harnesses
├── redline/                 # experimental direct-KMD / bare-libdrm research
├── redline-dispatch/        # retained-tape record/replay + plan selection
└── redline-rocr/            # public ROCr/HSA ABI + PM4 packet builders
```

HIP sources live under `kernels/src/`; there is no JavaScript/TypeScript
operator runtime.

### Layering

| Layer | Crates | Role |
|---|---|---|
| Operator | `hipfire-cli`, `hipfire-config`, `hipfire-registry`, `hipfire-client`, `hipfire-tui` | Tag resolve, typed config, pull, HTTP service/client, one-shot daemon spawn |
| Product binary | `hipfire-daemon` | `[[bin]] name = "daemon"`. Message dispatch and process lifetime only |
| Generation | `hipfire-generate` | The generate bodies: `ar`, `qwen`, `dense`, `vision`, `batch`, plus the Redline fixtures |
| Serve engine | `hipfire-engine` | Scheduler, terminal control, emit, prompt. **Zero arch dependencies** |
| Composition root | `hipfire-loader` | Carrier registry, single `load_model` dispatch, `LoadedModel`, continuous-batch staging |
| Arch forward | `hipfire-arch-*` | Config / weights / state / static-dispatch forward (LLaMA exception: canonical forward remains in runtime) |
| Shared infra | `hipfire-runtime` | HFQ/safetensors, tokenizer, sampler, framing, spec primitives, `Architecture` trait |
| Substrate | `saddle-core` | Grammar, KV cache, `ArchCaps`, sampling policy. Depends only on `rdna-compute`, `hip-bridge`, `serde` |
| Kernel select | `hipfire-dispatch`, `rdna-compute` | Family tables + `Gpu` methods that pick WMMA/dot/baseline |
| Device FFI | `hip-bridge`, `hsa-bridge` | HIP runtime; optional HSA AQL helpers |
| Encode | `hipfire-quantize` | No GPU deps; CI-safe quantizer |
| Observability | `hipfire-detect`, `hipfire-atlas` | Detectors; bench corpus schema |
| Retained replay | `redline-dispatch`, `redline-rocr`, `rdna-compute::replay` | Product-integrated Redline path (see below) |
| Direct-KMD research | `redline`, `saddle-lab`, `hipfire-pflash` | Not the serving transport |

The dependency edge runs one way and is checked by `cargo tree`:

```text
saddle-core → hipfire-runtime → hipfire-loader → hipfire-engine
                                      → hipfire-generate → hipfire-daemon
```

Two invariants hold today and are worth keeping:

- **`saddle-core` never depends upward.** Its only dependencies are
  `rdna-compute`, `hip-bridge` and `serde`; `cargo tree -p saddle-core -e normal`
  matches nothing under `hipfire-{runtime,loader,engine,generate,daemon,arch-*}`.
- **`hipfire-engine` declares no arch crate at all.** The daemon reaches an
  architecture only through `hipfire-loader`'s `Carrier` and the
  `hipfire-generate` entry points; `main.rs` names no `hipfire_arch_*` type,
  in shipped code or tests.

`hipfire-runtime` lists the arch crates under `[dev-dependencies]` for its own
examples. That is not a cycle — dev-dependencies do not participate in the
build graph of anything that depends on it.

`hipfire-loader` depends on arch crates and is the **only** top-of-DAG
`load_model` entry the daemon uses. Forward stays **statically dispatched** on
concrete arch types after load (`Architecture` docs in
`crates/hipfire-runtime/src/arch.rs`): the trait is bring-up scaffolding, not a
hot-path `dyn` forward. **LLaMA exception:** `hipfire-arch-llama` is a facade —
canonical dense LLaMA/Mistral/plain-Qwen3 forward and shared transformer types
still live in `hipfire_runtime::llama`; the arch crate re-exports them.

## Request lifecycle

```text
hipfire run <tag-or-path> "…"
        │
        ▼
Native CLI (`crates/hipfire-cli`)
  resolve registry tag → model path under ~/.hipfire/models/ (or local path)
  if serve up AND not forced local → HTTP POST /v1/chat/completions
    forced local when `HIPFIRE_LOCAL` is truthy or any of `--image`,
    `--kv-mode`, `--kv-backend`, `--spec`/`--speculation`, `--model-draft`,
    `--draft-max`, `--dspark-conf-threshold` is passed (`force_local` in
    `crates/hipfire-cli/src/main.rs`; `--json`/`--no-stream` ride the HTTP
    route and do not force local)
    if HTTP fails while serve still live → abort (no local spawn; would collide)
  else → spawn one-shot daemon binary
        │
        ▼
Daemon (crates/hipfire-daemon/src/main.rs)
  hipfire_loader::load_model(path, …, &mut Gpu)
        │
        ▼
hipfire-loader
  ModelSource::from_path  →  HfqFile  |  SafetensorsSource dir
  Carrier registry probe on arch_id (+ is_dir namespace)
  carrier.load → LoadedModel { arch_id, state: ModelState::…, tokenizer, … }
  optional: draft/speculator, VL weights, EP/PP scaffolding
        │
        ▼
generate(…) ladder (daemon.rs)
  EP? → generate_ep
  else arch short-circuits (7/8/9/10/11/12) or qwen35/llama body
  (DFlash/spec, PP generate_multi, VL, …)
        │
        ▼
Arch forward (hipfire-arch-*/src)
  prefill (eager per-token and/or batched where implemented)
  decode_step / verify block
  final norm + lm_head when logits needed
        │
        ▼
rdna-compute Gpu methods  ↔  hipfire-dispatch family tables
  kernels/src/*.hip  →  precompiled hsaco or hipcc JIT cache
  ordinary HIP (hip-bridge)  |  retained replay fork (ReplayController → ROCr)
```

Serve long-running path: same daemon binary, HTTP surface documented in
[`SERVE.md`](SERVE.md); chat attach in [`CHAT.md`](CHAT.md).

## Model sources (not “two model paths”)

Load is **source × carrier**, not a two-file hard split.

### Sources

| Source | Type | Arch id origin |
|---|---|---|
| HFQ / MQ* artifact | `ModelSource::Hfq(HfqFile)` | Header `HfqFile::arch_id` |
| Safetensors / Paro directory | `ModelSource::Dir(SafetensorsSource)` | `derive_arch_id` from `config.json` (`architectures` / `model_type`) |

Defined in `crates/hipfire-runtime/src/loader_api.rs` and
`safetensors_source.rs`. Unrecognized dir `model_type` yields sentinel
`UNCLAIMED_ARCH_ID` (`u32::MAX`) so routing fails closed instead of defaulting
to Qwen3.5.

Tensor names follow family-dependent HuggingFace conventions (e.g. dense LLaMA
`model.layers.{i}.…`; Qwen3.5 nested `model.language_model.layers.…`). GGUF
inputs map only **recognized** llama.cpp names via `gguf_to_safetensors_name`
at quantize time; unknown GGUF names are retained unchanged.

### Carriers

Object-safe `Carrier` trait + static `REGISTRY` in
`crates/hipfire-loader/src/{lib,carriers}.rs`. `load_model` requires **exactly
one** matching carrier (zero → error; two → ambiguous error).

| Carrier | Claims `arch_id` | Arch crate / notes |
|---|---|---|
| `LlamaCarrier` | 0, 1 | `hipfire-arch-llama` (dense LLaMA/Mistral/plain Qwen3) |
| `Qwen2Carrier` | 7 | `hipfire-arch-qwen2` (Q/K/V attention bias path) |
| `Qwen35Carrier` | 5, 6 | `hipfire-arch-qwen35` (+ optional `hipfire-arch-qwen35-vl`) |
| `DotsOcrCarrier` | 8 | `hipfire-arch-dots-ocr` (vision + Qwen2 text decoder fields) |
| `Deepseek4Carrier` | 9 | `hipfire-arch-deepseek4` |
| `MinimaxCarrier` | 10 | `hipfire-arch-minimax` |
| `Lfm2MoeCarrier` | 11 | `hipfire-arch-lfm2moe` |
| `Cohere2MoeCarrier` | 12 | `hipfire-arch-cohere2moe` |

Canonical numeric registry: [`architecture-ids.md`](architecture-ids.md).

`LoadedModel.state` is a closed `ModelState` enum
(`Qwen2 | Qwen35 | Llama | Lfm2Moe | Minimax | Cohere2Moe | Deepseek4`).
dots.ocr keeps config/weights on dedicated `LoadedModel` fields and reuses
`qwen2_state`. EP (expert-parallel) for DeepSeek4 / MiniMax stores rank state
in `LoadedModel.ep`, not `state`.

### Multi-device load variants

| API | Scope |
|---|---|
| `load_model` | Single GPU (and Qwen3.5 PP when `pp > 1` inside the carrier) |
| `load_model_ep` | Expert-parallel: arch_id 9 and 10 only; staging reclaims completed/pushed ranks on failure (constructor-mid-failure can still leak that rank’s partial allocs) |

## Architecture crates

Each `hipfire-arch-*` crate owns family-specific config, weight load, GPU
state, and forward — **except LLaMA**: canonical dense forward and shared
transformer types remain in `hipfire_runtime::llama`; `hipfire-arch-llama` is
the facade/re-export plus bring-up/carrier surface. Other runtime-owned pieces
(sampler, prompt frame, EOS filter, loop guard, generic spec traits) stay in
`hipfire-runtime`.

| Crate | Family summary |
|---|---|
| `hipfire-arch-llama` | Facade for dense FA ids 0/1; forward body is `hipfire_runtime::llama` |
| `hipfire-arch-qwen2` | Standalone Qwen2 text |
| `hipfire-arch-qwen35` | Hybrid DeltaNet + full attention; dense (5) and MoE/A3B (6); DFlash/MTP hooks |
| `hipfire-arch-qwen35-vl` | Vision tower attached to Qwen3.5 ids 5/6 |
| `hipfire-arch-dots-ocr` | dots.ocr / Qwen2-VL-class OCR |
| `hipfire-arch-deepseek4` | DeepSeek V4 Flash (HC, compressed-KV indexer, optional MTP/DSpark) |
| `hipfire-arch-minimax` | MiniMax-M2 MoE |
| `hipfire-arch-lfm2moe` | LFM2.5 dense + LFM2.5-MoE hybrid short-conv / GQA |
| `hipfire-arch-cohere2moe` | Cohere2-MoE / North-Mini-Code |
| `hipfire-arch-toy` | Template only (`arch_id = 0xFF`); daemon must not dispatch |

Bring-up contract: implement `hipfire_runtime::arch::Architecture` (see
`hipfire-arch-toy` and production `hipfire-arch-qwen35/src/arch.rs`).

### Forward shape (typical dense/hybrid layer)

Per layer (names vary by family): pre-norm → mixer (attention and/or short-conv
/ DeltaNet) → residual → FFN norm → gate/up/down or MoE → residual. Final
embedding norm + tied or untied `lm_head` when logits are required. Hybrid
families carry extra recurrent state (DeltaNet, LFM conv tails) alongside KV.

## Dispatch and kernels

### Two related dispatch surfaces

1. **`rdna-compute`** — `Gpu` methods (`gemm_*`, `gemv_*`, attention, norm, MoE,
   sampling, …) used directly by arch forwards. Arch capability predicates live
   in `arch_caps` / feature flags; implementations pick WMMA → specialized →
   baseline. Source modules: `dispatch.rs`, `gemm.rs`, `gemv.rs`,
   `attention.rs`, `moe.rs`, `norm.rs`, `kernels.rs`, `compiler.rs`, `replay.rs`.
2. **`hipfire-dispatch`** — unified family tables and pipelines so callers avoid
   matching on `DType` by hand (`families/`, `tables/`, `pipeline/`, `ops/`).

Principle (both layers): fast paths first; baseline last; drop redundant arch
clauses when a newer predicate subsumes them.

### Kernel build

```text
kernels/src/<name>.hip
kernels/src/<name>.gfx1201.hip          # chip override
kernels/src/<name>.gfx12.hip            # family override (e.g. gfx1200+gfx1201)
        │  scripts/compile-kernels.sh  (chip → family → base)
        ▼
kernels/compiled/<arch>/…               # packaged / tree prebuild output
~/.hipfire_kernels/<arch>/<name>.<hash>.hsaco  # default JIT cache (or HIPFIRE_KERNEL_CACHE)
```

On startup the runtime prefers a hash-matching precompiled blob. Missing or
mismatched hash → hipcc JIT into the cache when hipcc is available; if hipcc is
unavailable, an explicitly warned **unvalidated** precompiled blob may still be
used. `hipfire diag` reports compiled blob/hash counts per arch, not which path
supplied each kernel.

Some arch crates also ship crate-local HIP (registered through their own
`kernels.rs`) for family-specific ops.

## Serve / generate path

Composition root: `crates/hipfire-daemon/src/main.rs`.

After `load_model`, request handling builds sampling defaults
(request → HFQ `generation_config` recommendations → arch ladder) and enters
`generate`:

1. **EP** (`m.ep.is_some()`) → `generate_ep` (ds4 / MiniMax ranks).
2. **Arch short-circuits** with optional n-gram/spec when a `Speculator` is
   loaded and temp policy allows: ids 7, 9, 11, 12, 10, 8.
3. **Qwen3.5 / LLaMA body**: PP (`generate_multi`), DFlash/spec
   (`generate_dflash` / `generate_spec` / MTP), VL (`generate_vl`), else dense
   AR loops.

Capability examples that are **implemented** but not automatically
“product-certified”:

- LFM2.5 gfx1201 batched prefill is **branch-implemented**, not shipped/admitted:
  only the frozen **350M dense MQ4** fixture under `arch_id == 11 && is_gfx1201()`
  with explicit `HIPFIRE_LFM2_PREFILL_BATCH=1`; every other LFM cohort/dtype fails
  closed after that gate. Default remains eager/decode-shaped prefill.
- N-gram draft is model-free opt-in (`HIPFIRE_NGRAM_DRAFT`); many arches
  implement `SpecTarget`, but acceptance and product defaults are separate.
- DFlash draft resolution is CLI-side path/auto-match; daemon loads the path it
  is given. Mode toggles and MoE caveats: config/env owners, not this page.

Speculation inventory snapshot: [`speculation-support-inventory.md`](speculation-support-inventory.md)
(historical — verify in source before claims).

## Redline roles (capability vs certification)

Three crates + one integration controller. Normative contributor procedure:
[`REDLINE.md`](REDLINE.md) (**branch-implemented** vs `origin/beta` at the
comparison base in [`INDEX.md`](INDEX.md)).

| Piece | Owns | Does **not** own |
|---|---|---|
| `redline` | Experimental direct-KMD via libdrm_amdgpu (device, BO, PM4 CS, fences) | Product serve transport |
| `redline-dispatch` | Dispatch-DAG record/validate, artifact/kernarg identity, plan compile/selection, retained AQL/PM4 graph construction; HIP and AQL backends | ROCr ABI lifetimes; model admission policy |
| `redline-rocr` | Dynamically loaded public ROCr/HSA ABI; queues, signals, AQL packets, arch PM4-IB builders | Model scheduling / backend admission |
| `rdna-compute::replay::ReplayController` | Product integration on `Gpu`: record, prepare, route, poison, reset | A fourth transport crate |

Active retained-PM4 mental model (ordinary HIP launches remain the default path):

```text
recorder-aware launch site
  ├─ ordinary HIP → hip-bridge
  └─ retained replay (ReplayController fork when armed/ready)
       → retained tape (opcodes, kernargs, geometry, deps, bindings)
       → arch-specific PM4 lowering
       → public-HSA command memory
       → one PM4-IB AQL packet by default (PreparedPm4Graph::Single)
         or multiple queue/phase packets when explicitly enabled phased
         multi-queue replay is used (PreparedPm4Graph::Phased)
       → ROCr queue / doorbell / completion
```

**Runtime routing/default ≠ admission or certification.** Example runtime
automatic default (source capability/default fact only, `REDLINE.md`; not an
admission or certification record): `mq4r_redline_default` — exact GPU arch
`gfx1100`/`gfx1151`/`gfx1201` + PP=1 + TP=1 + case-insensitive `.mq4r`
extension configures model default replay backend (model-family agnostic; no
`arch_id` gate); `gfx1200` and all other arches remain opt-in. Built-in `hip`
config profile, explicit `HIPFIRE_REPLAY_BACKEND`, or manual capture disables
the automatic default / opts into broader capability.
Promotion and performance claims require the certification ladder in
`REDLINE.md` (parity, timed-arm route proof, matched stationary conditions).
`ReplayState::Ready` alone is not repository certification. Stitching manual
capture to product timed-arm proof without that ladder is **blocked**
([`INDEX.md`](INDEX.md)).

Do not describe Redline as “future only”: retained record/replay and ROCr PM4-IB
paths are **implemented**; what remains gated is **route certification and
default product claims**.

## KV cache (shared policy)

Resolved by `hipfire_runtime::kv_mode` per load-site policy (aliases and
accepted sets differ by carrier). Concrete modes include:

| Mode | Role (summary) |
|---|---|
| `q8` | Q8_0 K and V |
| `asym2` / `asym3` / `asym4` | Lower-bit rotated/Lloyd K; V typically wider |
| `fwht2` / `fwht3` / `fwht4` | FWHT-rotated K tiers |

Exact layouts and math: [`QUANTIZATION.md`](QUANTIZATION.md). Hybrid linear
layers (DeltaNet) use fixed recurrent state instead of FA KV for those layers.

## Observability hooks

Examples (full list: [`env-vars.md`](env-vars.md)):

- `HIPFIRE_GRAPH` — hipGraph capture (debug; AR-oriented)
- `HIPFIRE_PROFILE` / rocprof integration — internal vs external timing
- Daemon log under the serve data dir — load progress, JIT, dispatch decisions
- `hipfire-detect` — post-hoc JSONL detectors (non-blocking)
- `hipfire-atlas` — structured bench rows (`--emit-atlas`)

## Where to contribute

| Task | Start here |
|---|---|
| New model family | `.agents/skills/hipfire-arch-port/` + `hipfire-arch-toy`; register carrier + id |
| New GPU arch / WMMA port | arch-port skill; `rdna-compute` + kernel variants |
| Kernel micro-opt | `kernels/src/<name>.<chip>.hip`; dispatch/family wiring; methodology under `docs/methodology/` |
| Quant format / encoder | `hipfire-quantize`; [`QUANTIZATION.md`](QUANTIZATION.md) / [`QUANTIZE.md`](QUANTIZE.md) |
| Retained replay / PM4 | [`REDLINE.md`](REDLINE.md); do not treat `crates/redline` KMD as serve |
| Validation route | [`VALIDATION.md`](VALIDATION.md) only — no universal GPU gate |

## Related owners

| Concern | Owner |
|---|---|
| Arch id table | [`architecture-ids.md`](architecture-ids.md) |
| Docs lifecycle / truth states | [`INDEX.md`](INDEX.md) |
| Redline certification | [`REDLINE.md`](REDLINE.md) |
| Models / VRAM / sidecars | [`MODELS.md`](MODELS.md) |
| Multi-GPU ops | [`multi-gpu.md`](multi-gpu.md) |
| Perf claim protocol | [`methodology/perf-benchmarking.md`](methodology/perf-benchmarking.md) |
