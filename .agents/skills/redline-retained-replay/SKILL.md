---
name: redline-retained-replay
description: Discovery hook for retained AQL/PM4 (Redline) work. Use when admitting a model to retained replay, changing recorder/tape/PM4/queue policy, model reset or fallback, kernels on a retained route, or making a Redline-attributed bench claim. Routes to docs/REDLINE.md and docs/VALIDATION.md — does not fork procedure.
---

# redline-retained-replay

Thin discovery/route hook only. **Normative procedure lives in** [`docs/REDLINE.md`](../../../docs/REDLINE.md). Validation harness ownership lives in [`docs/VALIDATION.md`](../../../docs/VALIDATION.md). Runtime source is authoritative for executable behavior. Do not treat this skill as a second manual.

## Reach for this when

- Admitting a model to retained AQL or retained PM4
- Changing recorder coverage, launch/kernarg identity, artifacts, effects, or bindings
- Changing retained-plan construction, PM4 lowering, or queue/hazard policy
- Changing model reset, pointer lifetime, replay failure, or fallback behavior
- Changing a kernel, fusion, Radiowave, or scheduling overlay on a retained route
- Making a benchmark or product claim attributed to Redline

## Not Redline

- Ordinary serial HIP
- HipGraph stream capture/replay
- Launch-count reduction or fusion on a serial-HIP path
- A stable partial recorder fingerprint without a complete retained tape
- Experimental direct-KMD `crates/redline` (not the serving transport)
- Prefill, speculative/MTP, or another non-plain-AR path merely because it can be captured

## Three-crate distinction

| Crate | Role |
|---|---|
| `redline` | Experimental direct-KMD/bare-libdrm; **not** the product serving route |
| `redline-dispatch` | Dispatch-DAG recording/validation, artifact/kernarg identity, plan compilation, retained AQL/PM4 graph construction |
| `redline-rocr` | Public ROCr/HSA ABI, queue/memory/packet/signal lifetimes, AQL encoding, arch PM4 builders |

Product integration uses `rdna-compute::replay::ReplayController` via `Gpu`. That does **not** make `rdna-compute` a fourth Redline transport crate.

## Read next (canonical owners)

1. **Full procedure + terminology** — [`docs/REDLINE.md`](../../../docs/REDLINE.md)
2. **Porting recipe** — [§5](../../../docs/REDLINE.md#5-reproducible-model-and-architecture-porting-recipe)
3. **Certification / route-proof ladder** — [§7](../../../docs/REDLINE.md#7-certification-and-route-proof-ladder)
4. **Benchmark record schema + claim language** — [§8](../../../docs/REDLINE.md#8-benchmark-record-schema-and-claim-language)
5. **New-route checklist** — [§12](../../../docs/REDLINE.md#12-copyable-new-route-checklist)
6. **Harness ownership** — [`docs/VALIDATION.md`](../../../docs/VALIDATION.md) (`redline_daemon_harness.py` = discovery/correctness under manual capture; **not** product timed-arm route proof alone)
7. **Env knobs (mutable inventory)** — [`docs/env-vars.md`](../../../docs/env-vars.md) Redline section; do not restate here

## Fail-closed stops

- **No C/A attestation** from this skill.
- **Tooling-gap stop:** manual shadow/capture report + product timing report cannot be stitched into positive timed-arm route proof. Stop full Redline-attributed promotion until a route-proof-capable product report records controller, observed-replay, transport, and anti-fallback evidence required by REDLINE §7.
- Discovery evidence (phase fingerprint, shadow/parity under `HIPFIRE_REPLAY_MANUAL_CAPTURE`) ≠ installed/routed product PM4. Promotion stays policy-gated by REDLINE + admissions ownership in VALIDATION.
- Historical benchmarks remain historical; redline performance is not route-certified without same-report timed-arm proof.

## Workflow

1. Confirm the change is in scope (above). If not, stop — use ordinary HIP/kernel skills.
2. Read `docs/REDLINE.md` for the exact lifecycle, ownership, and certification ladder.
3. Pick harnesses only from `docs/VALIDATION.md` (do not invent gates).
4. Never claim Redline speedup without proving the retained route was active (controller + observed replay + transport + anti-fallback per §7–§8).
