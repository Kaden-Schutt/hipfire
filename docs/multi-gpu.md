# Multi-GPU operator guide

Operator reference for hipfire multi-device modes. Mutable env inventory lives in
[`env-vars.md`](env-vars.md). Validation route selection lives **only** in
[`VALIDATION.md`](VALIDATION.md) — this page does not invent minimum routes.
Bring-up narrative is historical:
[`multi-gpu-bringup-lessons.md`](multi-gpu-bringup-lessons.md).

| Field | Value |
|---|---|
| Inventory date | 2026-08-26 (post-merge mainline) |
| Audited source ref | `692a726dde53508cb53de1a74c720e75a7c9f33e` (pre-merge pin, historical); current surface: post-merge mainline crate split |
| Page state | **shipped / ref-pinned** for source-wired multi-device behavior (see [`INDEX.md`](INDEX.md)); performance tables in the appendix are **historical** only |
| Orchestration | `crates/hipfire-hardware/src/lib.rs` (`Gpus` + device resolution; `hipfire_runtime::multi_gpu` re-exports) |
| PP load path | `crates/hipfire-loader/src/carriers.rs` (`load_qwen35_pp`) |
| Daemon load / refusals | `crates/hipfire-daemon/src/main.rs` |
| Load admission | daemon `daemon_load_plan` → `hipfire_loader::admit_path` (incl. non-HFQ mesh-source refusal) — preflight runs **before** any prior-model unload; the admitted load itself runs later via `load_admitted` |
| Supporting multi-GPU script | [`scripts/pp-gate.sh`](../scripts/pp-gate.sh) (not a VALIDATION selector minimum route) |
| Admissions | [`admissions.yml`](admissions.yml) — schema v2, exactly one single-GPU retained-PM4 record; no multi-GPU records (fail closed; none inferred) |

Three multi-device modes exist. A load request resolves to **one effective
axis**: more than one of `pp` / `tp` / `ep` greater than 1 is refused —
`daemon_load_plan` surfaces the `hipfire_loader::admit_path` error before any
unload (`tp×ep` = COMP-001, `pp×{tp|ep}` = CAP-001). Admission runs
**before** any prior model or PFlash drafter is unloaded, so a request refused
at admission leaves the previous model loaded.

| Mode | Load knob | What it does | Runtime surface |
|---|---|---|---|
| **Pipeline parallel (PP)** | daemon load `params.pp` (`N`, default `1`) | Contiguous **layer bands** across `N` devices; residual stream crosses bands via `boundary_copy` | Qwen3.5 / 3.6 **HFQ** (`arch_id` 5 dense, 6 MoE/A3B) via the legacy `load_qwen35_pp` path; admitted dense cells — LLaMA QK-norm and no-QK-norm (`arch_id` 0), plain Qwen3 (`arch_id` 1) — via `load_admitted` → `pp_model` (MeshCarrier, **HFQ-only**) |
| **Tensor parallel (TP)** | daemon load `params.tp` | Within-layer weight/attention sharding for admitted dense cells | Admitted dense cells only, **HFQ-only**: LLaMA QK-norm (`arch_id` 0) and plain Qwen3 (`arch_id` 1) via `load_admitted` → `tp_model` (MeshCarrier). Every other cell is Planned/unsupported and refuses at admission |
| **Expert parallel (EP)** | daemon load `params.tp` **or** `params.ep` | Within-layer expert sharding + all-reduce; every rank runs every layer (`Gpus::init_tp`) | Admitted EP cells only: MiniMax-M2 (`arch_id` 10) and DeepSeek V4 Flash (`arch_id` 9). `tp`/`ep` ≥ 2 folds into the admitted parallel request; other arches refuse at admission (e.g. Qwen3.5 MoE `ep`/`tp` = 2 is Planned `AXIS-002` → `[CAP-001]` refusal, prior model kept). No Qwen3.5 EP admission is claimed |

`pp = 1`, `tp = 1`, and `ep` absent/1 are single-GPU. Behavior matches the
pre-multi-GPU paths.

**None of the listed PP/TP/EP routes is an admission or product default.**
[`admissions.yml`](admissions.yml) has no multi-GPU records at schema v2 (the sole earned row is single-GPU `pp=tp=1`).
Source-wired means the load path exists in the merged mainline — not that it is
promoted.

**PP is not tensor-parallel serving.** It does not give multi-user throughput.
It is a **capacity** tool: fit larger context / larger HFQ weights by splitting
layers. **No speedup is promised** for models that already fit on one card;
current historical measurements on one 2× gfx1100 station were slower under
sequential PP=2 (see [Appendix A](#appendix-a-historical-pp-evidence-immutable)).

**CLI note:** the shipped CLI forwards **`tp`** (`hipfire serve --tp N` →
`params.tp` in the serve load message). There is no `HIPFIRE_TP` env read in
the current control plane (retired with the pre-merge CLI). **`params.pp` and
`params.ep` are not first-class CLI flags today** — set them on a
raw daemon JSONL `load` (or any client that builds that message). Examples and
`scripts/pp-gate.sh` do this directly.

## Topology (PP)

Source of truth: `Gpus` in `crates/hipfire-hardware` (re-exported as `hipfire_runtime::multi_gpu`).

1. **Device pick** — `hardware.devices = "0,1,..."` is the physical visibility
   list. Startup installs it as `ROCR_VISIBLE_DEVICES` and gives HIP the
   matching post-filter logical list `0..N-1`; this avoids compounded nonzero
   filters while keeping both backends on the same physical GPUs.
2. **Layer map**
   - Default: `Gpus::init_uniform(pp, n_layers)` — contiguous bands,
     `base = n_layers / N`, remainder distributed so max−min ≤ 1 layer.
   - Escape hatch: `HIPFIRE_PP_LAYERS=a,b,…` → `Gpus::init_layers` (length must
     equal `pp`, sum must equal `n_layers`). Skips the uniform free-VRAM delta
     check; still enforces arch match unless overridden.
   - `Gpus::init_vram_weighted` is **not implemented** (returns a scheduled-for-v1.1 error).
3. **Placement convention (Variant 2)** — `output_device = last` device holds
   `output_norm + lm_head`. Device 0 holds the embedding side of the split.
4. **Boundary traffic** — at each band edge, `boundary_copy` moves the residual
   (`hipMemcpyPeerAsync` when peer access is up; otherwise HIP host-staging).
   Caller waits with `wait_boundary`.
5. **Peer access** — `enable_peer_all` must run **after** weights/KV/scratch that
   need peer maps are allocated. Incomplete peer matrix → host-staging fallback
   (slower, still correct). Partial pair failure does not abort capable pairs.
6. **Preflight**
   - Default: **exact `Gpu.arch` string match** across devices
     (`d.arch != arch0` in `preflight_vram` / `crates/hipfire-hardware`). This is not a
     loose “family” compare.
   - Free-VRAM delta ≤ `HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB` (default **2.0**) for
     `init_uniform` / `init_tp` only.
   - `HIPFIRE_ALLOW_MIXED_ARCH=1` opts into mixed-arch pairs (JIT per arch;
     peer may host-stage).
7. **Threading** — HIP work is **single-threaded** for the daemon lifetime.
   `bind_thread` before peer enable and before device-bound work. No rayon/tokio
   HIP callers in v1.

EP topology is different: `init_tp` sets every device’s layer map as “all layers
on rank 0” for PP helpers, while the EP forward ignores bands and shards experts.
RCCL all-reduce is used unless `HIPFIRE_TP_USE_RCCL=0` (host fallback not
implemented — that opt-out errors).

### Peer / fabric checks (host)

```sh
rocm-smi --showtopo
rocm-smi --showtoponuma
rocm-smi --showtopoaccess
```

A full `True` peer-access matrix is ideal. Missing peer access does not block
load; copies fall back to host staging.

## Launch and config

### PP load (daemon JSONL)

```json
{"type":"load","model":"/path/to/qwen3.5-9b.mq4","params":{"max_seq":16384,"pp":2}}
```

Example process:

```sh
# Persist one physical device list for both HIP and ROCr.
hipfire config set hardware.devices 0,1

# Optional: asymmetric bands (must sum to n_layers, length == pp)
# export HIPFIRE_PP_LAYERS=16,16

# Bit-stable k-split reduction for pp=1 vs pp=2 parity work
export HIPFIRE_DETERMINISTIC=1

cargo run --release -p hipfire-daemon
# then send the load JSON above on stdin
```

### EP load (CLI)

```sh
HIP_VISIBLE_DEVICES=0,1 hipfire serve <minimax-or-deepseek4-tag> --tp 2
# --tp forwards params.tp in the serve load message; HIPFIRE_TP is not read
```

`params.ep` is accepted on raw daemon JSONL loads and folds into the same
admitted EP request (`ep` ≥ 2 on arch 9/10 → EP loader). EP does not honor a
non-default `--kv-mode` the same way single-GPU load does today — the CLI
warns when both are set. Reload without a DFlash draft for EP.

### Emulation (`HIPFIRE_EMULATE_GPUS`) — debug hardware resolution

`HIPFIRE_EMULATE_GPUS=N` is a **debug hardware-resolution emulation enable
switch**: any parsed value ≥ 2 (unset, `< 2`, or unparseable = off) turns on
aliasing of the requested logical device ids onto the physical devices present
(`resolve_device_ids` maps each id by `rem_euclid` into the physical range —
`[0,1] → [0,0]` on a one-GPU box), so the multi-GPU paths (PP/TP/EP) run on
one card for development and parity work.

**The value is not a degree.** The rank count comes from the explicit axis
request: `params.pp=2` creates two PP ranks (aliased), `params.tp=2` two EP/TP
ranks — `HIPFIRE_EMULATE_GPUS=4` with `params.pp=2` still aliases exactly two
ranks, and the variable alone creates no ranks. It does not default `tp` or
any other degree (the pre-merge “default mode is TP” promotion was deleted
with `resolve_parallelism`). It is env-only — there is no
`hardware.emulate_gpus` TOML key.

### Environment (operator-facing)

Canonical table: [`env-vars.md`](env-vars.md) (`MULTI-GPU` group). Short map:

| Variable | Role |
|---|---|
| `hardware.devices` | Persistent physical device list; lowers to ROCr physical selectors and matching HIP logical selectors before initialization |
| `HIP_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES` | Legacy one-shot filters; compatible pairs are normalized and ambiguous pairs fail closed |
| `HIPFIRE_DEVICES` | Legacy compatibility alias for `hardware.devices` |
| `HIPFIRE_PP_LAYERS` | Explicit per-device layer counts for PP |
| `HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB` | `init_uniform` / `init_tp` free-VRAM delta |
| `HIPFIRE_ALLOW_MIXED_ARCH` | Opt into mixed-arch device sets |
| `HIPFIRE_DETERMINISTIC` | Deterministic WMMA reduction path (parity / bisect) |
| `HIPFIRE_EMULATE_GPUS` | Debug hardware-resolution emulation — see [Emulation](#emulation-hipfire_emulate_gpus--debug-hardware-resolution). Not a TP/EP request |
| `HIPFIRE_TP_USE_RCCL` | `0` opts out of RCCL (errors; no host AR yet) |
| `HIPFIRE_PP_PFLASH=1` | **Experimental** — accept PFlash compose with `pp>1` (not a product default; not route-certified) |
| `HIPFIRE_PP_DFLASH=1` | **Experimental** — accept DFlash draft field with `pp>1` (cross-card spec generate is **not** fully implemented; see daemon refusal text) |

Test-only: `HIPFIRE_HAVE_2_GPU`, `HIPFIRE_PP_PARITY_MODEL` (pp_parity harness).

## Refusals, silent paths, and experimental exceptions (`pp > 1`)

Many non-support cases are enforced at **load** (daemon and/or carrier) and fail
closed — do not expect silent degrade to single-GPU for those. **Exceptions and
carrier-specific wording matter**; do not treat the table as one universal
string or a blanket “always refuse at load.”

| Condition | Behavior |
|---|---|
| More than one of `pp` / `tp` / `ep` > 1 | Error: mutually exclusive (`tp×ep` = COMP-001, `pp×{tp|ep}` = CAP-001); refused at admission, prior model kept |
| Admission refusal (`[CAP-001]` / `[COMP-001]` / Planned cell) | **Preflight-before-unload:** `daemon_load_plan` runs `admit_path` before any unload — the previous model + PFlash drafter stay loaded. Effective-TP/EP loads additionally defer the prior-model unload until the new model is loaded; Single/PP unload eagerly. Dense TP/PP are **HFQ-only**: `daemon_load_plan` refuses non-HFQ mesh sources in preflight (before any unload); the loader-internal checks (`load_tp_admitted` / `load_pp_admitted`) are defensive backstops |
| DFlash `draft` set and `HIPFIRE_PP_DFLASH` unset | Error: DFlash requires `pp=1` |
| DFlash `draft` set and `HIPFIRE_PP_DFLASH=1` | **Experimental exception:** load may accept; cross-card speculative generate is **not** fully implemented (daemon message states PR2–4 of the hetero PFlash/DFlash plan are incomplete). Not an admission. |
| CASK / TriAttention sidecar set | Error: requires `pp=1` |
| PFlash drafter / mode on and `HIPFIRE_PP_PFLASH` unset | Error: PFlash requires `pp=1` |
| PFlash on and `HIPFIRE_PP_PFLASH=1` | **Experimental exception:** opt-in only; not a product default and not route-certified performance. |
| Non-admitted carriers at `pp>1` | Carrier-specific error strings (not one universal quote). Examples: `qwen2:` / `dots_ocr:` / `deepseek4:` / `minimax:` / `lfm2moe:` / `cohere2moe:` `pipeline-parallel (pp>1) unsupported` (distinct `safetensors + pp>1 unsupported` on dirs). LLaMA-family (arch 0/1) PP is an **admitted** dense cell and does not refuse. |
| Qwen3.5 **safetensors directory** + `pp>1` | Error: `qwen35: safetensors + pp>1 unsupported` |
| Qwen3.5 / 3.6 **VL HFQ** + `pp>1` | **Not a hard refuse.** `load_qwen35_pp` is the text HFQ loader only — vision weights are **not** loaded on that path, so VL artifacts **silently become text-only** under PP. Serve real VL at `pp=1`. |
| `bench_prefill` / multi-GPU EP | Daemon refuses bench_prefill when `pp>1` or EP is active |

EP-only: admitted EP loads (`tp`/`ep` ≥ 2 on arch 9/10) refuse a DFlash draft;
any other `tp>1` / `ep>1` request refuses at admission (`[CAP-001]`, before
unload) instead of reaching the EP loader — e.g. Qwen3.5 MoE `tp=2` is Planned
`AXIS-002`. No Qwen3.5 EP admission is claimed.

### Architectural limits (current)

- Homogeneous **exact arch string** by default (`ALLOW_MIXED_ARCH` is opt-in).
- No automatic VRAM-weighted split (`init_vram_weighted` stub).
- PP decode is sequential across bands (no async multi-band pipeline / per-band
  graph capture as a documented product path).
- Experimental `HIPFIRE_PP_*` flags are **not** admissions and are not
  route-certified performance features.

## Memory budget

Weights, KV, and scratch land on the devices that own each band. Last device
also carries `output_norm + lm_head`. Per-card headroom depends on quant, KV
mode, `max_seq`, and split.

**Reproduce on your hardware** (2+ visible GPUs, deltanet feature):

```sh
HIP_VISIBLE_DEVICES=0,1 cargo run --release --features deltanet \
  -p saddle-lab --example pp2_vram_probe -- \
  ~/.hipfire/models/qwen3.5-9b.mq4 4096
```

Historical VRAM and throughput tables from the original multi-GPU PP doc are
preserved **verbatim** in [Appendix A](#appendix-a-historical-pp-evidence-immutable).
They are **historical** only: they do not carry a complete measured-identity
manifest (measurement date + binary md5 + model identity on the same report),
so they must not be labeled **measured**, must not be treated as floors, and
must not be edited in place. Re-run the probe / perf protocol before claiming
fit or speed on other SKUs.

PP KV policy for Qwen3.5 multi-GPU defaults through `QWEN35_PP_POLICY` (see
`kv_mode.rs`) — do not assume single-GPU `auto` KV defaults apply unchanged.

## Validation routes

There is **no universal GPU gate** ([`VALIDATION.md`](VALIDATION.md)). Multi-GPU
claims are **manual** and path-specific. **`scripts/pp-gate.sh` is not named in
the VALIDATION claim→route selector** and therefore is **not** a canonical
minimum route ([`VALIDATION.md`](VALIDATION.md) § retired / unnamed gate
scripts). Treat it only as **supporting manual evidence**.

Map claims through the selector’s classes:

| Claim class (selector) | What applies for multi-GPU | Notes |
|---|---|---|
| Forward / fusion / KV **numerical or state parity** | Path-specific parity/state oracle when one exists for the surface | Supporting tool today: `pp_parity_chatml` example (also invoked by `pp-gate.sh`). If no oracle exists for a surface → **blocked**. Not `serve_harness.py`. |
| Forward / serve **user-facing semantics** | `scripts/serve_harness.py` with the exact model (after parity if numbers/state can break) | Semantics only |
| Perf improvement under PP or EP | [`methodology/perf-benchmarking.md`](methodology/perf-benchmarking.md) + stationary matched runs; `speed-gate.sh` / `gates.sh` perf arm when applicable | Bench numbers without protocol/identity are not promotion evidence. Historical appendix rows are not floors. |
| Arch port (new device family behavior) | [`methodology/arch-port-validation.md`](methodology/arch-port-validation.md) | Channel + speed; no retired coherence battery as acceptance |
| Model/route **admission** | Row in [`admissions.yml`](admissions.yml) | Schema v2 exact-row only — multi-GPU PP/EP are **not** admitted |
| Docs-only edits | No-GPU CI / `scripts/no-gpu-ci.sh` | Never substitutes for GPU parity |
| Unknown multi-device surface | **Blocked** until VALIDATION grows a row | Fail closed |

Supporting manual commands (not selector minimums):

```sh
# Supporting multi-GPU battery (parity + daemon e2e + refusals). Skips cleanly
# with <2 usable devices. Not automatic merge proof; not a VALIDATION minimum.
./scripts/pp-gate.sh

# Faster: parity example only
./scripts/pp-gate.sh --skip-end-to-end

# Topology filter dry-run (no GPU work)
./scripts/pp-gate.sh --dry-run

# Direct parity example
HIP_VISIBLE_DEVICES=0,1 cargo run --release --features deltanet \
  -p hipfire-runtime --example pp_parity_chatml -- \
  ~/.hipfire/models/qwen3.5-0.8b.mq4
```

**pp-gate knobs** (see script header): `PP_GATE_DEVICES`,
`HIPFIRE_PP_GATE_INCLUDE_IGPU`, `HIPFIRE_PP_GATE_HETEROGENEOUS`,
`HIPFIRE_PP_GATE_MODEL`, `HIPFIRE_PP_GATE_REQUIRE_SYSFS`. Filters drop known APU
iGPUs and skip heterogeneous ISA families unless overridden.

Pre-commit (when hooks installed) runs pp-gate if staged paths match the
multi-GPU hotspot regex in [`.githooks/pre-commit`](../.githooks/pre-commit).
That is a **path-gated local guard**, not full product admission.

Retired `scripts/coherence-gate-*.sh` batteries are **not** acceptance for PP
([`VALIDATION.md`](VALIDATION.md) § retired).

## Related

- Env inventory: [`env-vars.md`](env-vars.md)
- Serve / EP CLI: [`SERVE.md`](SERVE.md), [`CLI.md`](CLI.md)
- Validation selector: [`VALIDATION.md`](VALIDATION.md)
- Historical bring-up: [`multi-gpu-bringup-lessons.md`](multi-gpu-bringup-lessons.md)
- Issue tracker context: [#58](https://github.com/warpfront/hipfire/issues/58)

---

## Appendix A — Historical PP evidence (immutable)

> **Lifecycle — historical only.** The block below is the prior multi-GPU PP
> document body retained for provenance. It is **not** current procedure, **not**
> a product floor, and **not** an admission. Truth state: **historical** (not
> **measured** — the retained tables lack a complete same-report measurement
> date + binary identity + model identity manifest required by
> [`INDEX.md`](INDEX.md)). Do not edit the evidence body; amend only by adding
> new dated sections outside this appendix. Warnings in the active sections
> above supersede any stronger claim language inside the retained body
> (including “measured”, “Status: v1 feature-complete”, absolute speedup
> wording, and gate-as-acceptance framing).

# Multi-GPU Pipeline-Parallel

**Status:** v1 feature-complete on `feat/multi-gpu-pp` branch — tracking
issue [#58](https://github.com/warpfront/hipfire/issues/58). Stages
0–9 of the v2 plan are merged; refusal contracts (DFlash / VL / CASK +
pp>1) are wired and validated. This doc is the source of truth for
memory budget, deployment recipes, throughput, and known limitations.

## Why PP

hipfire on a single 24 GB card hits VRAM walls on:

- 27B at `--max-ctx ≥ 16K` with `kv_mode=asym3` (`AGENTS.md:356`)
- 35B-A3B at `--max-ctx ≥ 4K` with FP32 KV
- hypothetical 80B-A3B at any context

Pipeline-Parallel (PP) shards layers across N devices. Each device owns a contiguous "band"
of consecutive layers. The residual stream `s.x` flows through the bands sequentially: dev_0
runs layers `0..k1`, copies `s.x` to dev_1, dev_1 runs layers `k1..k2`, and so on. Final
`output_norm + lm_head` run on the last device (dev_last) — its `s.logits` is read by the
sampler in place.

**What PP gives you on 2× 24 GB:**
- Run 27B / 35B-A3B that don't fit on one card with extended context
- Unlock max_ctx on 27B beyond single-GPU OOM limits
- ~50-70% of single-GPU throughput on already-fitting models (sequential PP=2 is slower per token)

**What PP does NOT give you:**
- Faster multi-user serving — that's TP (tensor parallel), separate roadmap
- Speedup on models that already fit on one card

## Memory budget (per-card, PP=2)

Numbers below are **measured** on 2× Radeon RX 7900 XTX (gfx1100, 25.8 GiB VRAM
each) via `crates/saddle-lab/examples/pp2_vram_probe.rs` —
`hipMemGetInfo` deltas captured at each allocation stage
(`load_weights_multi`, `Qwen35ScratchSet`,
`KvCache::new_gpu_asym3_capped_multi`, `DeltaNetState`). Per-card columns
report the worst-of-two (the device that holds more — typically dev_last,
which carries `output_norm + lm_head`). `total` is the sum across both cards.

| Model | quant | n_layers | dim | KV mode | ctx | weights | KV/card | scratch+DN/card | total | per-card max | fits 24 GiB? |
|-------|-------|----------|-----|---------|-----|---------|---------|-----------------|-------|--------------|--------------|
| qwen3.5:0.8b | mq4 | 24 | 1024 | asym3 | 4096 | 1.3 GB | 50 MB | 8 MB | 1.3 GB | 0.7 GB | yes |
| qwen3.5:4b | mq4 | 32 | 2560 | asym3 | 4096 | 4.0 GB | 134 MB | 15 MB | 4.0 GB | 2.0 GB | yes |
| qwen3.5:9b | mq4 | 32 | 4096 | asym3 | 4096 | 5.6 GB | 134 MB | 19 MB | 5.6 GB | 2.8 GB | yes |
| qwen3.5:9b | mq4 | 32 | 4096 | asym3 | 16K | 6.2 GB | 436 MB | 46 MB | 6.2 GB | 3.1 GB | yes |
| qwen3.5:9b | mq3 | 32 | 4096 | asym3 | 4096 | 4.4 GB | 134 MB | 19 MB | 4.4 GB | 2.2 GB | yes |
| qwen3.5:27b *(via 3.6 proxy)* | mq4 | 64 | 5120 | asym3 | 4096 | 15.5 GB | 268 MB | 42 MB | 15.5 GB | 7.8 GB | yes |
| qwen3.5:27b *(via 3.6 proxy)* | mq4 | 64 | 5120 | asym3 | 16K | 16.8 GB | 872 MB | 80 MB | 16.8 GB | 8.4 GB | yes |
| qwen3.5:27b | mq3 | 64 | 5120 | asym3 | 4096 | 12.6 GB | 268 MB | 40 MB | 12.6 GB | 6.3 GB | yes |
| qwen3.5:27b | mq3 | 64 | 5120 | asym3 | 16K | 13.8 GB | 872 MB | 78 MB | 13.8 GB | 6.9 GB | yes |
| qwen3.6:35b-a3b | mq4 | 40 | 2048 | asym3 | 4096 | 23.5 GB | 103 MB | 21 MB | 23.5 GB | 11.8 GB | yes |
| (hypothetical) 80B-a3b | mq4 | 80 | 8192 | asym3 | 4096 | ~42 GB | ~250 MB | ~2 GB | ~44 GB | ~22 GB | estimate, tight |

Notes:
- `qwen3.5:27b mq4` rows use `qwen3.6:27b mq4` as a measurement proxy (same
  `n_layers=64`, `dim=5120`, `head_dim=256`, `n_kv_heads=4` — VRAM-equivalent).
  When `qwen3.5:27b mq4` lands as a downloadable artifact the rows can be
  re-measured directly.
- `qwen3.6:35b-a3b mq4` is the current-generation A3B; `qwen3.5:35b-a3b mq4`
  ships local-only with the same MoE shape — measurement carries over.
- The 80B row stays an estimate — no public artifact exists.
- "scratch+DN/card" combines `Qwen35ScratchSet::per_device[i]` (residual
  stream, attention/FFN scratch, flash partials, logits) and the
  `DeltaNetState` slice owned by that device's LA-layer band.

**Asymmetry under Variant 2 (lm_head on dev_last):**
- dev_0 carries `token_embd`
- dev_last carries `output_norm + lm_head`
- Layer count shifts by ±1 for `n_layers % 2 != 0` (uniform split formula
  `base + (i < rem ? 1 : 0)`)

Probed on this hardware: per-card max < 24 GiB at every measured shape. The
A3B model at 11.8 GB/card has the largest headroom consumer; everything
else stays under 8.5 GB/card with 4 K context, under 9 GB/card at 16 K.

To reproduce on your hardware:

```sh
HIP_VISIBLE_DEVICES=0,1 cargo run --release --features deltanet \
    -p saddle-lab --example pp2_vram_probe -- \
    ~/.hipfire/models/qwen3.5-9b.mq4 4096
```

## Deployment recipes

The daemon takes a `pp` field in the load message (default `1` =
single-GPU, identical behavior to pre-PP code paths):

```sh
# Filter to the two 7900 XTX (drop the iGPU)
HIP_VISIBLE_DEVICES=0,1 hipfire run qwen3.5:27b --max-ctx 16384 "Hi"

# Bypass the inherited `gemm_..._wmma_ksplit` non-determinism (k-split
# atomicAdd reduction varies by warp scheduling — see commit f54ca71
# and kernels/src/gemm_hfq4g256_residual_wmma_ksplit.hip:22). Required
# for byte-equivalent output across processes / pp configurations.
HIPFIRE_DETERMINISTIC=1 HIP_VISIBLE_DEVICES=0,1 hipfire run qwen3.5:9b "Hi"

# Override uniform-VRAM tolerance (default 2 GiB; arches must match)
HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB=4 hipfire run qwen3.5:27b "Hi"
```

Direct daemon JSON (driving without the CLI):

```json
{"type":"load","model":".../qwen3.5-27b.mq4","params":{"max_seq":16384,"pp":2}}
```

### Environment variables

| Variable | Effect |
|----------|--------|
| `hardware.devices = "3,1"` | ROCr physical filter `3,1`; HIP and the engine receive matching logical devices `0,1` |
| `HIPFIRE_DETERMINISTIC=1` | Force k2 WMMA reduction (no atomicAdd) — bit-identical across processes/pp configs at ~33% perf cost on small-batch decode |
| `HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB=N` | Pre-flight VRAM-asymmetry tolerance for `Gpus::init_uniform` (default 2.0) |
| `HIPFIRE_PREFILL_BATCHED=0` | Disable batched WMMA prefill (per-token fallback). Diagnostic for ksplit non-det isolation |
| `HIPFIRE_PREFILL_MAX_BATCH=N` | Override per-chunk prefill batch (default `PREFILL_MAX_BATCH`); chunks > N split with peer-copy at boundary |
| `HIPFIRE_WO_WMMA_VARIANT={k2,ksplit,k4,…}` | Manual override of the wo-residual GEMM variant — see `dispatch.rs` auto-dispatch |

### Current axis policy

The loader's current behavior is architecture-specific; `PAR-001` records
that behavior and the policy below. `CAP-001` will provide consistent early
enforcement for planned and unsupported cells; it is not implemented here.

| Axis/cell | Current policy | Production gate or owner |
|-----------|----------------|--------------------------|
| Plain Qwen PP/TP | Implemented (admitted dense route); physical proof pending | `HW-003` / `HW-006` |
| LLaMA PP | Implemented (admitted dense route, incl. no-QK-norm); physical proof pending | `HW-003` |
| LLaMA TP | Implemented for admitted QK-norm artifacts; non-QK-norm refused | `AXIS-001`, then `HW-006` for production closure |
| Qwen2/VibeThinker PP/TP | Planned; not currently supported | `AXIS-001`, then `HW-003` / `HW-006` |
| Qwen35 arch-resident PP | Partial; generation remains arch-resident | `GEN-001`, then `HW-004` |
| DeepSeek4 EP | Existing code; pending physical proof | `HW-001` |
| MiniMax EP | Existing code; pending physical proof | `HW-002` |
| Other planned PP/TP and MoE-EP cells | Current architecture-specific behavior remains under `PAR-001` characterization; consistent early planned/unsupported-cell enforcement belongs to `CAP-001` and the owning `AXIS-*`/`GEN-*` implementation | `AXIS-001`–`AXIS-004`, `GEN-001`, `PAR-002` |
| Dense `EP > 1` | Normalized to one effective replica before mesh/device/allocation/collective construction (`CAP-001`; dense rows marked `normalized-to-single`) | `CAP-001` (complete) |
| `TP > 1` × `EP > 1` | Explicitly refused (COMP-001) | Out-of-scope decision recorded; see `.agent-progress/device-mesh-refactor-tracker.md` |

Each implemented cell requires its applicable named hardware gate (for
example, `HW-001`, `HW-002`, `HW-003`, `HW-004`, `HW-006`, or another gate
listed by its owner) before production status. Emulated parity and
allocation evidence never substitute for the applicable physical gate.

## Throughput baseline (gfx1100 × 2)

Measured on 2× Radeon RX 7900 XTX, ROCm 6.4.3, with
`HIPFIRE_DETERMINISTIC=1` (bit-equivalent pp=1 ↔ pp=2 output).

| Model | Prompt | pp=1 prefill | pp=2 prefill | pp=1 decode | pp=2 decode | pp=2/pp=1 decode |
|-------|--------|--------------|--------------|-------------|-------------|------------------|
| 0.8B mq4 | 22 tok | 838 tok/s | 588 tok/s | 332 tok/s | 227 tok/s | 68% |
| 0.8B mq4 | 322 tok (chunked) | 6493 tok/s | 5490 tok/s | 315 tok/s | 212 tok/s | 67% |
| 35B-A3B mq4 (MoE) | 15 tok | 331 tok/s | 258 tok/s | 142 tok/s | 97 tok/s | 68% |

Current limitation: pp=2 decode is per-token, and
`forward_scratch_multi` launches each kernel per layer without graph
capture. Async stream/pipeline execution, per-band graph capture, and
pipelined prefill are not currently available.

## Current boundaries

- The axis table is a policy and current-behavior summary, not a claim that
  `CAP-001` has landed.
- Documented refusal guards for `pp > 1` with DFlash or CASK/TriAttention
  remain in force; this document does not broaden those paths.
- Current `pp > 1` Qwen35 loading bypasses VL detection and lacks explicit
  load-time refusal. This is not PP+VL support; `CAP-001` must turn it into
  an explicit admission error.
- Homogeneous arch only (`init_uniform` hard-fails on arch mismatch)
- Uniform layer split — `init_layers(per_device)` is the manual escape hatch;
  `init_vram_weighted` is stubbed.
- Per-token decode has no async stream pipeline or per-band graph capture.
- Pipelined prefill (chunk N+1 on dev_0 while chunk N processes on dev_1) is
  not yet available.

## Validation (Stage 9)

```sh
# Multi-GPU gate. Skips silently when fewer than 2 GPU visible.
./scripts/pp-gate.sh

# Just the parity smoke (no daemon end-to-end), faster
./scripts/pp-gate.sh --skip-end-to-end

# Underlying byte-equivalence example
HIP_VISIBLE_DEVICES=0,1 cargo run --release --features deltanet \
    -p hipfire-runtime --example pp_parity_chatml -- \
    ~/.hipfire/models/qwen3.5-0.8b.mq4
```

The `pp-gate.sh` battery checks:
1. Per-token `forward_scratch_multi` ≡ `forward_scratch` bit-exact (pp_parity_chatml)
2. Daemon `pp=1` ≡ `pp=2` byte-identical with `HIPFIRE_DETERMINISTIC=1` (greedy ChatML)
3. DFlash + pp=2 refusal at load
4. CASK + pp=2 refusal at load

A pre-commit hook calls `pp-gate.sh` automatically when staged files
match the `multi_gpu|pp_|peer_access|pipeline|stages` hotspot regex.

## Verifying peer access on your hardware

```sh
rocm-smi --showtopo            # weights, hops, link types
rocm-smi --showtoponuma        # NUMA placement
rocm-smi --showtopoaccess      # peer accessibility matrix
```

A `True` in every cell of `--showtopoaccess` for the cards you plan to use means peer-access
should work. If not, hipfire falls back to host-staging via pinned buffers (slower but correct).

## Open questions (will be filled in as Stages land)

- DFlash + PP integration scope — pending maintainer guidance on issue #58
- Whether mixed-arch should be soft-warn or hard-fail — currently hard-fail
- `hardware.devices` is the physical visibility source of truth; startup lowers ROCr physical selectors to matching HIP logical `0..N-1`
