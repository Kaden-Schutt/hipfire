<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Kaden Schutt
hipfire — see LICENSE and NOTICE in the project root.
-->

# Architecture-port validation

Evidence tiers for **GPU arch** ports (RDNA/CDNA kernel/dispatch routes)
and **model-arch** forward bring-up. This file is procedure and pitfall
guidance only.

| Field | Value |
|---|---|
| Inventory date | 2026-07-19 |
| Audited source ref | `692a726dde53508cb53de1a74c720e75a7c9f33e` |
| Route selector | [`docs/VALIDATION.md`](../VALIDATION.md) |
| Executable GPU-arch skill | [`.agents/skills/hipfire-arch-port/`](../../.agents/skills/hipfire-arch-port/) |
| ISA / fit-view skill | [`.agents/skills/hipfire-kernel-atlas/`](../../.agents/skills/hipfire-kernel-atlas/) |

**Rules**

1. Pick routes only from [`VALIDATION.md`](../VALIDATION.md). This page
   does not invent gates.
2. There is **no universal replacement gate** and **no C/A attestation**.
3. `scripts/coherence-gate-*.sh` batteries are **retired as acceptance**
   (historical reproduction only). Do not require them for merge,
   promotion, or arch-port certification language.
4. Hardware-unwitnessed paths are **blocked**, not assumed green.
5. A green automatic no-GPU CI run is never GPU-arch proof.

---

## What “arch port” means here

Two orthogonal surfaces share this methodology owner:

| Surface | Meaning | Primary evidence |
|---|---|---|
| **GPU arch** | New/changed `gfx*` routing in kernels, `dispatch.rs`, WMMA/MFMA builtins | Channel + speed (+ path oracle when the change can break numbers) |
| **Model arch** | New/changed forward for a model family (`hipfire-arch-*`) | Path-specific hidden-state / logit / state parity; serve only for user-facing semantics |

WMMA operand shapes, builtin names, and lane layouts live in the
arch-port skill (`wmma-matrix.md`), not here.

---

## Evidence tiers (fail closed)

Use the **narrowest** tier set that covers the changed surface. Higher
tiers do not replace lower ones when lower ones apply.

### Tier C — Kernel channel (GPU numeric)

| Item | Path / command |
|---|---|
| Wrapper | [`scripts/test-kernels.sh`](../../scripts/test-kernels.sh) `[arch]` |
| QA wrapper | [`scripts/test-kernelsQA.sh`](../../scripts/test-kernelsQA.sh) `[arch]` |
| Binaries | `cargo build --release --features deltanet -p hipfire-runtime --example test_kernels --example test_kernelsQA` then `./target/release/examples/test_kernels` / `test_kernelsQA` |
| Role | Element-wise GPU vs CPU reference on the **detected** arch. Catches silent WMMA/MFMA C-mapping bugs that throughput and chat smoke miss. |
| Not this tier | Dispatch `bind_thread` coverage ([`scripts/verify-bind-thread.sh`](../../scripts/verify-bind-thread.sh)); model coherence; perf floors. |

**Arch argument caveat:** `test-kernels.sh [arch]` does **not** select or
assert the GPU under test. The optional argument only changes the wrapper
banner (`=== hipfire kernel test harness (${ARCH}) ===`); the binary
independently detects and prints the real GPU. Always confirm the binary’s
detected arch in the run log. When an expected-arch assertion is required,
use `test-kernelsQA.sh <arch>` (passes `--expected-arch` to the QA binary).

**Pass:** every exercised case reports OK/PASS; any FAIL/MISMATCH blocks
the port for that kernel.

### Tier P — Path-specific parity / state oracle (model or fusion)

| Item | Path |
|---|---|
| Comparator | [`scripts/compare_hidden_states.py`](../../scripts/compare_hidden_states.py) |
| Tiny oracles | [`scripts/gen_tiny_oracle.py`](../../scripts/gen_tiny_oracle.py), worked: [`scripts/gen_tiny_minimax.py`](../../scripts/gen_tiny_minimax.py), [`scripts/gen_tiny_lfm2moe.py`](../../scripts/gen_tiny_lfm2moe.py) |
| HF dump helper | [`scripts/dump_hf_hidden_states.py`](../../scripts/dump_hf_hidden_states.py) |
| Hipfire dumpers (examples) | `crates/hipfire-arch-*/examples/dump_*_hidden_states.rs` (e.g. minimax, lfm2moe, qwen35 in runtime examples) |
| Graph / batch oracles (when present) | Arch-owned examples such as `graph_parity_*`, `prefill_batch_parity`, `conv1d_*_parity` under the touched crate |

**Role:** per-layer hidden-state cosine / rel-L2, final-token logit
parity, KV/conv/`n_tokens` state parity — whatever the oracle for that
surface defines.

**Blocked:** if no oracle exists for the changed surface, numerical
parity is **blocked**. Do not substitute
[`scripts/serve_harness.py`](../../scripts/serve_harness.py) or retired
coherence-gate scripts for parity.

Cosine guidance used on prior ports (not universal floors): ~≥0.999
Q8-grade plumbing; ~≥0.99 common for 4-bit expert noise once structure
is right. Always state the tol the oracle/docs for that format define.

### Tier S — Speed floor (when a baseline file exists)

| Item | Path |
|---|---|
| Script | [`scripts/speed-gate.sh`](../../scripts/speed-gate.sh) |
| Baselines | [`tests/speed-baselines/<arch>.txt`](../../tests/speed-baselines/) |
| Bench binary | `./target/release/examples/bench_qwen35_mq4` (built by the script) |
| Modes | default all sizes; `--fast` (4B-oriented); `--update-baselines`; `--tolerance`; `--verbose` |

**Present baseline files (inventory):** `gfx1013`, `gfx1030`, `gfx1031`,
`gfx1100`, `gfx1100x2_pp`, `gfx1151`, `gfx1201`, `gfx906`, `gfx908`,
`gfx942`.

**Role:** prefill/decode vs committed floor for the **detected** arch
(`HIPFIRE_BASELINE_ARCH` / probe). Default tolerance 5% unless overridden.

**Blocked / not certification:**

- No matching `tests/speed-baselines/<arch>.txt` → speed-floor claim for
  that arch is **blocked** until a baseline is earned under
  [`perf-benchmarking.md`](perf-benchmarking.md) and recorded deliberately.
- Speed-gate green is **not** model admission and **not** Redline route
  proof ([`REDLINE.md`](../REDLINE.md)).
- Dirty-tree or stale-binary comparisons are measurement failures, not
  wins. Force a clean bench binary rebuild when comparing diffs
  (`rm target/release/examples/bench_qwen35_mq4` before re-run so
  `ensure_build` rebuilds).

### Tier U — User-facing serve semantics (optional, claim-scoped)

Only when the port changes observable serve behavior:

| Harness | Path | Scope |
|---|---|---|
| Generic serve | [`scripts/serve_harness.py`](../../scripts/serve_harness.py) | Model-agnostic battery / chain / session |
| LFM framing | [`scripts/serve_harness.py`](../../scripts/serve_harness.py) with the exact `lfm2.5:*` tag | LFM thinking / combined output only |
| Maintained wrapper | [`scripts/gates.sh`](../../scripts/gates.sh) | Optional Redline capture + serve + optional `probe_commits.sh`; requires `--model`; **does not** call retired coherence-gate scripts |

Semantics success ≠ numerical parity ≠ product timed-arm / PM4 proof.

### Tier R — Measurement corpus (Kernel Atlas)

For Amdahl, ISA fit, and candidate experiment queues — not acceptance:

- CLI: [`scripts/kernel_atlas.py`](../../scripts/kernel_atlas.py)
- Method: [`kernel-atlas.md`](kernel-atlas.md)
- Architecture: [`kernel-atlas-architecture.md`](kernel-atlas-architecture.md)
- Agent wrapper: `.agents/skills/hipfire-kernel-atlas/`

Atlas `status: ok` is a successful observation only. Rows become INDEX
**measured** evidence only with a complete fixture/binary/model identity and
date manifest; incomplete rows stay exploratory. Neither class admits a route
into [`admissions.yml`](../admissions.yml).

---

## GPU arch port — minimum route

Per [`VALIDATION.md`](../VALIDATION.md) “Arch port” row:

1. **Tier C** on the **target** GPU (real hardware; no emulator path).
2. **Tier S** on every baseline arch whose dispatch/kernel path the
   diff can touch (commonly the contributor’s baseline card **and** any
   arch with a committed floor that shares the edited predicates).
3. **Tier P** when the change can alter numeric forward/state for a
   model path that has an oracle; otherwise leave parity **blocked**
   rather than inventing a gate.
4. Never restore retired `coherence-gate-*.sh` as the acceptance bar.

If target hardware is unavailable, the port stays **blocked** for
channel proof. Coordinate with a hardware holder
(`.agents/skills/hipfire-arch-port/contributor-onboarding.md`); do not
merge on “should be identical” alone.

---

## Model-arch forward bring-up — cheap parity loop

Use when plumbing an existing kernel family into a new model crate.
Assumes kernels already exist; new HIP kernels need Tier C first.

1. **Tiny reference oracle** from HF/upstream modeling via
   `scripts/gen_tiny_oracle.py` (adapt the marked arch block) or a
   worked sibling (`gen_tiny_minimax.py`, `gen_tiny_lfm2moe.py`).
2. **Dump post-residual, pre-final-norm** hidden states both sides in
   HFHS: magic `HFHS\0\0\0\0`, then `<IIII>` =
   `(n_layers, n_pos, hidden, reserved)`, then
   `n_layers × [n_pos, hidden]` f32 row-major.
3. **Compare** with `scripts/compare_hidden_states.py`
   (`rms`, `rel_L2`, `mean_cos`, `min_cos`).
4. **Bisect** inside the first bad layer (post-attn vs FFN/MoE;
   isolated `block_out = post − pre`).
5. **Precision-sweep** suspect blocks (e.g. 4-bit → higher) to separate
   quant noise (error shrinks) from structural bugs (error flat).
6. **Stage-dump** router/topk/gate intermediates vs numpy/torch from F32
   weights; first diverging stage is the bug.
7. **Read kernel/dispatch source** for RoPE convention, RMSNorm `+1`,
   route scale, rotation matching — do not guess.

### Tiny-oracle pitfall checklist

- Match hardcoded kernel `k_top` (e.g. `_k8_` → `num_experts_per_tok=8`).
- Every quantized 2D weight: `k % group_size == 0` (often 256).
- Keep **real** `head_dim` / `rotary_dim`; shrink head/layer counts, not head_dim.
- Keep routing tensors on well-behaved paths (Q8 `gemv_q8_0`); avoid F16
  router hits on lm-head GEMM shapes.
- Re-split packed expert layouts to the runtime’s split layout when needed.
- Confirm standard vs Gemma RMSNorm (`weight` vs `1+weight`) before load.

### When this does not generalize

If the model needs a **new** HIP kernel or quant family with no decode
path, Tier C (and any kernel micro-oracle) comes first. Cosine harnesses
only validate plumbing once the kernel exists.

Historical worked example (MiniMax-M2 / arch_id 10): tiny HF oracle
caught top-k vs `_k8` overflow, F16 router→lm-head GEMM, and quant-vs-bug
via MQ4→MQ6 sweep — **measured** on that campaign, not a live floor.

---

## Explicit non-routes

| Anti-pattern | Disposition |
|---|---|
| Retired coherence-gate script batteries as current acceptance | **Rejected** (historical only; see VALIDATION) |
| Generic bare coherence-gate entry script | **Absent** — do not invent it; route via VALIDATION |
| Serve harness as numerical/state parity | **Rejected** |
| No-GPU CI as GPU-arch proof | **Rejected** |
| Atlas / bench tok/s as admission or certification | **Rejected** |
| Claiming an unwitnessed WMMA/MFMA path “works” | **Blocked** |
| Universal “three gates green = ship” ritual including coherence | **Rejected** — use VALIDATION tiers only |
| Inferred [`admissions.yml`](../admissions.yml) row | **Rejected** under schema v2 (no wildcards; exact earned rows only) |

---

## Related owners

| Concern | Owner |
|---|---|
| Claim → route map | [`docs/VALIDATION.md`](../VALIDATION.md) |
| Perf measurement protocol | [`perf-benchmarking.md`](perf-benchmarking.md) |
| Bench-suite layout | [`bench-suite.md`](bench-suite.md) |
| Kernel Atlas usage | [`kernel-atlas.md`](kernel-atlas.md) |
| Kernel Atlas architecture | [`kernel-atlas-architecture.md`](kernel-atlas-architecture.md) |
| rocprof vs internal profile | [`rocprof-coverage.md`](rocprof-coverage.md) |
| Redline certification | [`docs/REDLINE.md`](../REDLINE.md) |
| GPU-arch skill (playbook, WMMA matrix) | `.agents/skills/hipfire-arch-port/` |
