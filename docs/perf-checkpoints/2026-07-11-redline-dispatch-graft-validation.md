# Redline Dispatch default-off graft validation

Branch: `feat/redline-dispatch-graft`
Base: `45cb5abfc` (`origin/master` at graft creation)

## Passing checks

- `cargo check -p redline-rocr -p redline-dispatch -p rdna-compute --lib`
- `cargo test -p redline-rocr -p redline-dispatch`
  - 34 `redline-dispatch` unit tests
  - 11 dispatch integration tests
  - 8 identity/invalidation integration tests
  - 12 `redline-rocr` tests
- `cargo test -p rdna-compute --lib replay::tests`
  - default HIP never records or routes
  - auto requires two passing shadow samples and explicit plan installation
  - any failed gate creates sticky fallback
- `git diff --check`

The library checks emit existing unused-code warnings elsewhere in Hipfire; the
new crates and replay controller add no compile errors.

## Canonical coherence gate: environment-blocked

Command:

```text
HIPFIRE_COHERENCE_OUT=.redline-work/product-cert/coherence-dflash-fast.md \
  ./scripts/coherence-gate-dflash.sh --fast
```

Both cases fail before token generation at
`dflash_spec_demo.rs:658` while allocating `DflashScratch`. A direct minimal
reproduction with `HIPFIRE_REPLAY_BACKEND=hip` reports:

```text
VRAM @ after target load: used 15.81 GB, free 1.29 GB
VRAM @ after draft load: used 16.77 GB, free 0.32 GB
alloc draft scratch: HipError { code: 2, message: "hipMalloc: out of memory" }
```

This is not counted as a coherence pass. It is an environment/capacity block:
the default controller records nothing and allocates no GPU memory in HIP mode,
and the failure occurs before any replay/shadow route exists. Re-run the gate on
a card with sufficient free VRAM, or with a canonical target/draft pair whose
scratch allocation fits, before merge or enablement.

## Enablement conclusion

The branch is suitable for review as a default-off integration seam. It is not
ready to route a Hipfire model through AQL: a model adapter must still declare
resource accesses and kernarg ABI, build the prepared plan, and pass the shadow
and coherence gates. Do not merge or enable based on this checkpoint alone.
