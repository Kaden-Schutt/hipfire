# Skill: redline-retained-replay

Trigger-oriented discovery hook for retained AQL/PM4 work in hipfire.
**Not a second manual.** The only normative procedure is
[`docs/REDLINE.md`](../REDLINE.md). Read that first, then keep this file as
a short apply/stop checklist.

## Reach for this when

Any of the following is in scope:

- admitting a model to retained AQL or retained PM4
- recorder coverage, launch/kernarg identity, artifacts, effects, or bindings
- retained-plan construction, PM4 lowering, queue/wait/fence/acquire policy
- model reset, pointer lifetime, replay failure, or fallback wording
- a kernel, fusion, Radiowave, or scheduling overlay **on a retained route**
- any benchmark or product claim attributed to Redline

## Not Redline

Do not treat these as retained-route work or as proof that Redline routed:

- ordinary serial HIP
- HipGraph stream capture/replay
- launch-count reduction or fusion on a serial-HIP path
- stable partial recorder fingerprints without a complete tape
- experimental direct-KMD `crates/redline` (not the serving transport)
- prefill, speculative/MTP reseed-proposal-verify, or other non-plain-AR paths
  merely because something can be captured

Redline certifies ordinary sequential single-token autoregressive
continuation. Capability, opt-in availability, automatic default, guide
certification, and dated evidence are separate classifications.

## Mandatory first read

1. Open [`docs/REDLINE.md`](../REDLINE.md) and follow it for the full recipe,
   lifecycle, failure atlas, certification ladder, and claim language.
2. Confirm current executable behavior in runtime source/tests. The guide is
   contributor policy; source wins for what the binary does today.
3. Do not copy procedure into other docs. Link the guide.

## Collect before editing

Freeze a reproducible fixture before changing admission, tape, or lowering:

- UTC date; branch/commit; clean/dirty; daemon/binary digest
- GPU product, gfx arch, ROCm/runtime/driver identity
- model path, arch id, quantization, artifact digest(s)
- topology (`pp`/`tp`), KV mode, continuation API
- exact prompt or deterministic token stream + digest
- baseline route and every route-affecting `HIPFIRE_*` setting (including unset)

If any identity field is missing, stop and complete the fixture first.

## Non-negotiable gates (names only)

Pass in order. Failure blocks promotion even if a later metric looks good.
Details and required evidence live in `docs/REDLINE.md` §7–§8.

1. **Baseline correctness** — ordinary HIP is stable on the exact fixture.
2. **Capture completeness** — compute = retained + named external launches;
   sequence/artifact identity stable across positions and fresh processes.
3. **ABI/artifact validation** — exact loaded HSACO/symbol, padded kernargs,
   geometry, effects, bounded dynamic bindings.
4. **Multi-position shadow parity** — HIP, retained route, and exact
   HIP-kernarg-blob oracle agree on logits and every mutable model state.
5. **Route proof** — preparation, `Ready`, observed multi-position replay,
   transport/packet/queue identity, and proof the timed arm was not silent
   HIP/HipGraph.
6. **Production serve** — user-facing decode health, finish state,
   repetition/attractor behavior, and model framing.
7. **Stationary matched performance** — identical binary/model/prompt/
   settings/process/clocks; tok/s and ms/token; complete claim record.
8. **Long-context and lifecycle** — dynamic position, KV/recurrent growth,
   reset, failure semantics, model swap.

`ReplayState::Ready` is a runtime state, not repository certification.
Harness success without route proof is insufficient.

## Stop conditions

Stop and do not promote when any of these hold:

- fixture identity is incomplete or the baseline is incoherent
- admission boundary can arm/record/consume on an ineligible call
- launch counts do not reconcile, or a partial tape is treated as complete
- artifact/alias/ABI/geometry/lifetime contract is unresolved
- any mutable state surface is missing from the oracle
- timed arm lacks positive observed-replay / anti-fallback proof
- serve framing or semantic health fails
- performance arms are cross-harness, cross-binary, cross-prompt, mixed
  process policy, or silent-fallback
- replay-execution failure is described as same-forward HIP retry
  (that path does not exist; current forward errors, later sticky fallback)
- the only positive evidence is a fingerprint, microbench, or unmatched ratio

## Three-crate distinction

| Crate | Owns | Does not own |
|---|---|---|
| `redline` | Experimental direct-KMD/bare-libdrm machinery | Active product serving route |
| `redline-dispatch` | Dispatch-DAG record/validate, artifact/kernarg identity, plan compile, retained AQL/PM4 graph construction | ROCr ABI lifetimes; model-specific admission |
| `redline-rocr` | Public ROCr/HSA ABI load; queue/memory/packet/signal lifetimes; AQL encoding; arch PM4 builders | Model scheduling; dispatch-DAG policy; backend admission |

Product integration sits on `rdna-compute::replay::ReplayController` held by
`Gpu`. That controller role does not make `rdna-compute` a fourth Redline
transport crate.

## Runtime facts to keep straight

- Automatic default is the narrow gfx12 + `arch_id==6` + `pp=tp=1` + `.mq4r`
  predicate (see guide §3 / `gfx12_mq4r_redline_default`).
- `HIPFIRE_REPLAY_TRANSPORT` changes **only** transport; it does not by itself
  enable replay or bypass the model-default predicate.
- Automatic product path can go `Captured → Ready` on prepare success; it does
  **not** require `ShadowValidated`. Certification policy is stricter.
- Replay execution failure: poison + error this forward; sticky fallback later;
  **no same-call HIP retry**.

## Performance methodology (generic + Redline)

For ordinary measurement hygiene (warmup, noise band, fresh-process discipline):

- [`docs/methodology/perf-benchmarking.md`](../methodology/perf-benchmarking.md)
- [`docs/BENCHMARKS.md`](../BENCHMARKS.md) (dated tables = historical snapshots)

For any claim **attributed to Redline**, also satisfy guide §8: full identity
manifest, route-proof ledger per timed arm, matched stationary comparison, and
dated fixture-bound wording. Direct transport value (fixed kernel stack,
baseline vs retained) is separate from enabling value (PM4 vs PM4+one overlay).
Dated checkpoint rows are evidence for those fixtures only—not current defaults
or timeless floors.

## Related surfaces (links only)

- Canonical procedure: [`docs/REDLINE.md`](../REDLINE.md)
- Graft / dated provenance: `crates/redline-dispatch/HIPFIRE-GRAFT.md`
- ROCr ABI provenance: `crates/redline-rocr/PROVENANCE.md`
- Diagnostics: `scripts/redline_daemon_harness.py`
- Product stationary bench: `scripts/redline_product_bench.py`
