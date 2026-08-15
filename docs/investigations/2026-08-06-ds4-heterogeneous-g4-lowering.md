<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev> -->

# DS4 heterogeneous G4 — frozen MQ2R lowering

## Verdict

G4 is complete. The frozen DeepSeek V4 Flash 0731 MQ2R artifact now loads and
runs with the complete non-routed tier on exact `gfx1100` and all routed
experts on exact `gfx1151`. The canonical 2,048-prompt / 512-generation greedy
output is byte-identical to the certified single-`gfx1151` oracle. Routing,
logits, KV and recurrent state remain within the path-specific oracle limits,
and the production serve route passes fresh-process battery, eight-turn
session, and real client-disconnect rollback checks.

This gate is correctness, not a throughput promotion. G5 begins with the
canonical direct-HIP product measurement and profile. Short serving requests
ran around 31 tok/s, but they are diagnostics at different context depths and
are not substituted for the required 2,048/512 G5 number.

## Identity

- Branch: `ds4-beta-staging`
- Implementation and certification commits: `4130b311c` through `b54f5e45d`
- Model SHA256:
  `cbf2bbcfa3f47b1712a071836b2c48232dad7dfb763813a720f7d348a9318cce`
- Local artifact: `/home/kaden/models/deepseek-v4-flash-0731.mq2r`
- Dense owner: device 0, `gfx1100`, PCI `0000:66:00.0`
- Routed owner: device 1, `gfx1151`, PCI `0000:bf:00.0`
- ROCm/HIP: 7.14
- Mode: direct HIP, no speculation, batch 1, top-k 6, greedy, Q8 request mode

The user-facing selector is typed configuration:

```toml
[hardware]
devices = "0,1"
deepseek4_compute_placement = "dense-expert-split(dense=arch:gfx1100,experts=arch:gfx1151)"
```

`serve_harness.py --devices 0,1` now writes the visibility set into its
isolated TOML. The invalid attempt that omitted it exposed only `gfx1100` and
failed closed before inference; it is preserved and excluded.

## Ownership and residency

The model index is opened once and records are routed directly to their owner.
No whole-model load-and-migrate step exists.

| Metric | `gfx1100` dense owner | `gfx1151` routed owner |
|---|---:|---:|
| Tensor allocations | 1,198 | 172 packed allocations |
| Projected weight bytes | 4,272,562,988 | 77,913,567,232 |
| Actual bytes including state/scratch/tables | 8,040,480,768 | 78,068,580,352 |
| Free bytes after load | 17,672,699,904 | 24,844,959,744 |
| Ownership violations | 0 | 0 |

The `gfx1100` owner holds the canonical residual, attention, compressor,
router, shared expert, HC and output-head state. The `gfx1151` owner holds only
the routed payloads, pointer tables and routed branch scratch. RMSNorm/FWHT are
not recomputed on `gfx1151`.

## Canonical generation oracle

| Field | Value |
|---|---|
| Prompt | `benchmarks/prompts/ds4_heterogeneous_code_2048.txt` |
| Prompt MD5 | `593234a767e71b97a3a4dad6431b47ce` |
| Prompt tokens | 2,048 |
| Generated tokens | 512 |
| Output bytes | 2,491 |
| Output MD5 | `ee05ab4f07393fb7d624d966a7dde4af` |
| Token equality | exact |
| Decoded-byte equality | exact |

The heterogeneous and single-`gfx1151` arms generated the same 512 token IDs
and decoded bytes. The example's internal elapsed times are deliberately not a
product benchmark: state/routing capture and per-position oracle work change
the launch and synchronization regime.

## State and routing parity

Seven certification positions were checked: 127, 511, 1023, 2047, 2175,
2303 and 2555. At each position the oracle compared 12 state tensors covering
218,774 values. The worst absolute difference was `0.000335693359375` in
`residual_streams` at position 2555, below the `1e-3` gate. Greedy output
remained byte-identical.

The route audit compared 110,037 records per arm:

| Metric | Result |
|---|---:|
| Selected-expert set equality | 100% |
| Top-1 equality | 100% |
| Selected-member recall | 100% |
| Mean route-weight L1 | 0.000001 |

Four late layers contained numerically tied experts whose ordering swapped in
0.04% of records; the selected set, top-1, membership, routed arithmetic and
decoded output were unchanged. This is recorded rather than misreported as
raw-bit state identity.

## User-facing semantics

Two independent fresh-process five-prompt batteries passed with 5/5 natural
stops, zero runaways, zero empty outputs, zero attractors and zero retrieval
misses. Their diagnostic average decode rate was about 31.2 tok/s.

The committed eight-turn `EmberIndex` session then passed in another fresh
process:

- 8/8 turns stopped normally;
- zero runaway, empty, attractor or retrieval failures;
- all 21 expected recall assertions passed;
- diagnostic average decode rate: 31.0 tok/s;
- final turn recalled project, hash, reader count, buffer size and config.

The harness now records session `expected_substrings` and
`retrieval_missing`; the earlier false summary that omitted session retrieval
checks is fixed and excluded.

## Cancellation and rollback

`scripts/test-ds4-heterogeneous-abort-resume.sh` closes a real streaming client
socket during decode. Cancellation is observed only before a layer or after
the shared/routed branches rejoin, so an abort arriving in flight drains to a
safe cross-device boundary before owner state is cleared.

The accepted run:

- closed after 16,384 streamed bytes and 154 committed tokens;
- logged `abort=client rollback=attested post_join=true`;
- synchronized and reset both exact owners before terminal release;
- immediately served a fresh heterogeneous request;
- returned exactly `the quick brown fox` and stopped normally at 31.746 tok/s.

No daemon, serve process or GPU lock remained afterward.

## Generic target and transport boundary

ROCm 7.14 clang accepts Code Object V6 `gfx11-generic`, and commit
`dde67bb16` proves one generic object loads and produces raw-bit exact output
on both `gfx1100` and `gfx1151`. That is the right portability and bring-up
fallback. It is not the product hot-kernel target: G1-G3 retain exact-target
HSACO so scheduling portability does not discard device-specific register,
cache and instruction selection.

RCCL was tested rather than assumed. Each card passed alone, while the mixed
communicator failed `invalid device function` under ROCm 7.14. ROCr SDMA was
therefore selected in G0, and the G1-G3 exact-target AQL schedule uses device
signals without a host wait inside the 43-layer graph. G4's direct-HIP serve
route retains one terminal prefill fence and fail-closed abort synchronization;
neither is a per-layer coordination mechanism. G6 replaces the remaining HIP
launch regime with dual-device retained ROCr/AQL/PM4 after G5 establishes the
direct-HIP profile.

## Validation

- `cargo test -p hipfire-runtime --example daemon abort -- --nocapture`:
  12 passed, 0 failed.
- `python3 scripts/serve_harness.py --self-test`: all three proof groups pass.
- `python3 -m py_compile scripts/serve_harness.py`: passed.
- `bash -n scripts/test-ds4-heterogeneous-abort-resume.sh`: passed.
- `scripts/fmt-changed.sh` on the changed daemon file: passed.
- `git diff --check`: passed.
- GPU lock released after every accepted and rejected GPU run.

ShellCheck was not available locally: the installed wrapper attempted to
download its binary into a read-only global Node directory. No ShellCheck
result is claimed.

## Evidence

Durable root on hipx:

```text
/home/kaden/ds4-gfx1151-evidence/2026-08-06-ds4-heterogeneous-g4/
```

| Evidence | SHA256 |
|---|---|
| `generation-state-route-oracle.log` | `acc95f094bd02e4d21095b2eb51486765d348322cdf20b7c9475435b51cb66ca` |
| `route-parity.txt` | `ba5b98a439413ed97dca184f6a3ad264d296378eb6858a0e56cae2df6b76c741` |
| `serve/battery-d.json` | `784a32f6a860cb7a15e4f8e3951389069013196238f9f6e4d954cf1762c3b288` |
| `serve/battery-e.json` | `65bd853369611b869a94d8cfbec747e6d73cb6995fe212eb1b4bbb7c3513c480` |
| `serve/session-f.json` | `130de54a151f3c957343f5cce313c2f092da167d5ed687cd1323d6bb4aa6938d` |
| `serve/session-f-serve.log` | `591e6fd1d5b732e87c1e744de7b9e0db6b3e05c0c75332aa400245494e2bf26c` |
| `serve/abort-resume-v2/serve-log-slice.txt` | `c6e091e50c461e0ac19667fbe59092d4e0173364999b820aadea5e54175ca97a` |
| `serve/abort-resume-v2/follow-up.json` | `a0f25b516d27aade48b89457c41d4f92431064375516c55fd61b6814b381edcd` |

Rejected evidence is preserved alongside the accepted files:

- NAS-backed early runs are excluded from load-time interpretation.
- short `max_tokens` batteries are excluded where the cap itself caused a
  runaway classification.
- `session-d` is excluded because its final assertion required words the
  requested labeled format did not contain; the decoded answer was correct.
- `session-e` is excluded because device visibility was implicit and the
  selector failed closed before inference.
- `abort-resume-serve.log` is excluded because an incorrect config-root path
  selected the user default model; it was terminated before a request.

## Next gate

G5 measures the direct heterogeneous route on the committed 2,048/512 prompt,
then profiles the actual critical path. The first performance question is not
whether LLVM can emit generic gfx11—it can—but which hot `gfx1100` DS4 kernels
are still using the portable fallback and how much queue/transport time remains
after the correct split is composed.
