# Skill: redline-retained-replay

Trigger-oriented discovery hook for retained AQL/PM4 work in hipfire. The only
normative procedure is [`docs/REDLINE.md`](../REDLINE.md); read it first. Use
this page only to decide whether Redline applies and where to continue reading.

## Reach for this when

- admitting a model to retained AQL or retained PM4
- changing recorder coverage, launch/kernarg identity, artifacts, effects, or bindings
- changing retained-plan construction, PM4 lowering, or queue/hazard policy
- changing model reset, pointer lifetime, replay failure, or fallback behavior
- changing a kernel, fusion, Radiowave, or scheduling overlay on a retained route
- making a benchmark or product claim attributed to Redline

## Examples that are not Redline

- ordinary serial HIP
- HipGraph stream capture/replay
- launch-count reduction or fusion on a serial-HIP path
- a stable partial recorder fingerprint without a complete retained tape
- experimental direct-KMD `crates/redline`, which is not the serving transport
- prefill, speculative/MTP, or another non-plain-AR path merely because it can
  be captured

## Three-crate distinction

| Crate | Role |
|---|---|
| `redline` | Experimental direct-KMD/bare-libdrm machinery, not the active product serving route |
| `redline-dispatch` | Dispatch-DAG recording/validation, artifact and kernarg identity, plan compilation, and retained AQL/PM4 graph construction |
| `redline-rocr` | Public ROCr/HSA ABI loading, queue/memory/packet/signal lifetimes, AQL encoding, and architecture PM4 builders |

Product integration uses `rdna-compute::replay::ReplayController` through
`Gpu`; that does not make `rdna-compute` a fourth Redline transport crate.

## Read first and continue here

- [Canonical guide and terminology](../REDLINE.md)
- [§5 — Reproducible model and architecture porting recipe](../REDLINE.md#5-reproducible-model-and-architecture-porting-recipe)
- [§7 — Certification and route-proof ladder](../REDLINE.md#7-certification-and-route-proof-ladder)
- [§8 — Benchmark record schema and claim language](../REDLINE.md#8-benchmark-record-schema-and-claim-language)
- [§12 — Copyable new-route checklist](../REDLINE.md#12-copyable-new-route-checklist)

**Tooling-gap stop:** the current manual shadow/capture report and product
timing report cannot be stitched into positive timed-arm route proof. Stop any
full Redline-attributed promotion until a route-proof-capable product report
records the controller, observed-replay, transport, and anti-fallback evidence
required by §7.
