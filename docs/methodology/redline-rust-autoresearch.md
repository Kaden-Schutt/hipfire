<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 Kaden Schutt -->

# Redline Rust autoresearch hooks

`hipfire-ar-hook` is the engine-owned edge of Redline's staged Rust
autoresearch controller. It is intentionally a small JSON stdin/stdout binary;
the controller does not import Hipfire model code and Hipfire does not own the
search ledger or fleet scheduler.

The hook targets the retained-PM4 surface on this `origin/redline` line:

- `census` converts Hipfire's existing `bod_<arch>.json` into the typed Loop 0
  census;
- `model` starts the baseline and candidate daemons, captures the real decode
  tape, and measures `redline_shadow_pm4` after the configured warmups. It
  requires bit-exact HIP/blob/PM4 logits, KV, and recurrent state, and also
  compares those hashes across the two daemon builds;
- `certify` repeats that shadow gate and runs the committed
  `scripts/serve_harness.py` in A/B/B/A order. The harness uses a fixed seed,
  production sampling, 2048 tokens, and fails on changed output, attractors,
  empty output, or a performance regression. A coherent length-cap stop remains
  recorded but is not itself a failure.

The controller and multi-GPU workers remain Rust. Loop 3 calls the existing
serve harness because that is Hipfire's golden product battery; it is not used
inside the cheap Loop 1 kernel-mining loop.

## Build and hook configuration

```bash
cargo build --release -p hipfire-ar-hook
```

Copy `examples/redline-ar-hook.json` into the repository-backed run directory
and replace the model/BOD/daemon paths. In the Redline pipeline, configure:

```json
{
  "loop0_census": {
    "command": [
      "/path/to/hipfire/target/release/hipfire-ar-hook",
      "census",
      "--config",
      "/path/to/run/hipfire-hook.json"
    ]
  },
  "loop2_model": {
    "budget": 8,
    "evaluator": {
      "command": [
        "/path/to/hipfire/target/release/hipfire-ar-hook",
        "model",
        "--config",
        "/path/to/run/hipfire-hook.json"
      ]
    },
    "fallback_generator": null,
    "fallback_budget": 0
  },
  "loop3_certify": {
    "budget": 2,
    "certifier": {
      "command": [
        "/path/to/hipfire/target/release/hipfire-ar-hook",
        "certify",
        "--config",
        "/path/to/run/hipfire-hook.json"
      ]
    }
  }
}
```

The baseline daemon must be built from the exact `RunKey.baseline` revision.
The candidate daemon may come from the hook config or from the candidate's
typed plan, which lets each finalist point at its own isolated worktree:

```json
{
  "launch": {
    "hipfire": {
      "candidate_daemon": "/repo/.redline-work/ar/builds/candidate-v7/daemon"
    }
  }
}
```

Every build, daemon log, serve row, capsule, and event stays below
`.redline-work`; no LLVM/model/build artifact is staged through `/tmp`.

## Four-card mining on `hiptrx`

Use one Redline `fleet` coordinator with four workers whose
`ROCR_VISIBLE_DEVICES` values are `0`, `1`, `2`, and `3`. Loop 1 distributes
different candidates across all four R9700 cards. An initial pass is then
replicated once per card; the frontier gate uses the median plus the configured
worst-card floor and rejects duplicate PCI identities.

Do not start four controllers and do not share one writable event ledger across
processes. Workers communicate only through atomic, repository-backed job
leases; the coordinator is the sole ledger writer.
