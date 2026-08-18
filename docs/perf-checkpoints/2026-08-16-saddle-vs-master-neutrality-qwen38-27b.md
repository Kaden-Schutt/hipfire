# 2026-08-16 — Saddle refactor is perf-neutral: decode 37.7 vs 37.7 tok/s

**Lifecycle: `historical`.** Evidence under the exact fixture and method below.
Not a current default, not an automatic baseline, not an admission decision.

## Why this run exists

The saddle programme moved 118k lines across 407 files, split `hipfire-runtime`
into seven crates, and replaced the 12-variant `ModelState` enum with
`Box<dyn ArchModel>`. None of that touches a kernel, so the expected result is
zero. The question worth answering is not "is it faster" but **"did it cost
anything"**, and two mechanisms could plausibly have:

1. **vtable dispatch** replacing an enum `match` on the decode path.
2. **Lost cross-crate inlining.** There is no `[profile.release]` section in the
   workspace `Cargo.toml`, so cargo defaults apply: `lto = false`,
   `codegen-units = 16`. Seven new crate boundaries with LTO off is a real
   inlining hazard, and on master the whole thing was one 48k-line crate the
   optimizer saw at once.

Mechanism 1 was ruled out statically before measuring: **zero downcasts occur
inside a per-token loop** — the trait is reached on setup and teardown paths
only. Mechanism 2 can only be answered by running it.

## Fixture

| | |
|---|---|
| host | `hiptrx`, 4× R9700 (gfx1201), ROCm 7.14, HIP 7.14 |
| model | `~/.hipfire/models/qwen3.8-27b.mq4r`, md5 `129909ad0fed21dcf72b5b9225e85604` |
| loaded | arch `qwen3_5`, dim 5120, layers 64, vocab 248320 |
| arm A | `master` @ `8510ca5f2` — `hipfire` md5 `78335c1acef85d621aee63a7f89204d7`, daemon (`examples/daemon`) md5 `4b5b90596abfb660fb8b973e3c733999` |
| arm B | `arch/saddle` @ `116745d56` — `hipfire` md5 `3201052c1e4a8a05c100c827e0ea0fe5`, daemon (`target/release/daemon`) md5 `35bfcc9ff5f12071d84a389af7deae1e` |
| harness | `hipfire bench --runs 5 --warmups 3 --max-tokens 128 --backend noslots --workload stateless --json` |
| isolation | separate `HOME` per arm, fresh daemon per arm, single stream |

`--concurrency` deliberately unused: it leaves the single-stream path and
answers a different question.

## Result

| metric | master | saddle | delta |
|---|---:|---:|---:|
| **decode tok/s** (median) | **37.7** | **37.7** | **0.0 %** |
| prefill tok/s (median) | 425.2 | 424.1 | −0.26 % |
| wall tok/s (median) | 37.0 | 37.0 | 0.0 % |
| TTFT ms (median) | 56.4 | 56.6 | +0.35 % |

Spread, 5 runs each:

| | master | saddle |
|---|---|---|
| decode | 37.6–37.7, stdev 0.040 | 37.6–37.7, stdev 0.049 |
| prefill | 423.3–425.6, stdev 0.833 | 422.2–424.6, stdev 0.830 |
| ttft | 56.4–56.7, stdev 0.120 | 56.5–56.8, stdev 0.098 |

The prefill and TTFT deltas are smaller than the run-to-run spread on either
arm. Decode is identical to the reported precision.

## Reading

**The refactor cost nothing measurable, and gained nothing measurable.** That is
the correct outcome for a layering change and should be stated as such rather
than dressed up.

The inlining hazard did not materialise, and the reason is arithmetic: at 37.7
tok/s on a 27B model the host has roughly 26 ms per token, against a GPU-bound
inner loop. Cross-crate call overhead is invisible at that ratio. **This result
should not be generalised to a smaller model** — the same measurement on a 0.6B,
where the host fraction is far larger, is a different experiment and was not
run here.

`lto = false` / `codegen-units = 16` remains the default and is now a known,
untested-at-scale property rather than an assumption. Enabling LTO is a
separate, unmeasured lever; the seven new crate boundaries would make it more
valuable than it was on master, not less.

## Build-time note, measured incidentally

Master's documented build is:

```
cargo build --release --features deltanet --example daemon \
  --example dflash_spec_demo --example encode_prompt --example run \
  -p hipfire-runtime
```

Saddle's is `cargo build --release`. This was confirmed by accident: the first
saddle build in this session failed with `no example target named 'daemon'`
because master's incantation was reused, and on saddle the daemon is a crate
(`hipfire-daemon`, binary `target/release/daemon`) rather than an example of
`hipfire-runtime`.

## Method note

Run with `hipfire bench`, not `examples/dflash_spec_demo`. The playbook in
`AGENTS.md` cited the example binary in nine places and did not mention
`hipfire bench` at all; that guidance was corrected in the same commit as this
record. `hipfire bench` drives the native daemon protocol — the path a user's
request actually takes — and owns warmup, repetition and JSON reporting.
