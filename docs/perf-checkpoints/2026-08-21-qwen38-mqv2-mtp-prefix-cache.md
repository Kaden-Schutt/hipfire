# Qwen3.8 MQ V2 native MTP, n-gram composition, and cached serve — 2026-08-21

**Lifecycle:** historical
**Disposition:** measured implementation checkpoint; not a product baseline or promotion gate
**Source:** `e1b9cc5d9` (`feat(mtp): unify sampled prefix-cached speculation`) plus `21141cce6` (`fix(dflash): honor registry nucleus sampling`)

## Scope

This checkpoint records native Qwen3.8 MTP enablement for MQ4V2 XT/Base/Pro, the generic sampled-MTP cutover, honest LCP prefix-cache reuse, MTP+n-gram composition, and the registry-sampled DFlash route fix needed by the same full-session campaign.

The campaign completed the short AR/MTP matrix, guarded n-gram proofs on gfx1100 and gfx1151, and 12 full eight-turn `session_coding.json` arms. It did **not** complete the remaining six available-device spec arms or any gfx1201 arm: the gfx1100 endpoint went link-down during the seventh turn of the Base MTP arm, both guards aborted, and no recovery or further GPU work was attempted.

## Fixtures

| item | identity |
|---|---|
| target MQ4V2 XT | md5 `e45d15bfe0c9a87132697101d17cbed6` |
| target MQ4V2 Base | md5 `d1292b4d5bd6046693604201a6ca8074` |
| target MQ4V2 Pro | md5 `279c563786499c6651de6a3a57b42b02` |
| MTP sidecar | md5 `1a78dd0d2c8c8a97abfc2f873193ae58` (same bytes beside all three targets) |
| DFlash2 MQ4V2 draft | md5 `013395583cd04206c8aa68f4d061983d`; declared all-sliding window W=2048 |
| short perf prompt | `merge_sort_thinking_off.txt`; md5 `253c7ac50857fe6d0e10fb0d2c5e35c0` |
| full session | `benchmarks/prompts/session_coding.json`; md5 `c0d470288bde3f1e54e4bba04da8f8a2` |
| host/runtime | `hipx`; HIP 7.14 |
| gfx1100 | RX 7900 XTX, HIP device 0, BDF `0000:66:00.0`, 24,560 MiB reported VRAM |
| gfx1151 | Radeon 8060S, HIP device 1, 98,304 MiB reported VRAM |

Binary identities changed as the lifecycle and sampled-route fixes landed:

| evidence | `hipfire` md5 | `daemon` md5 |
|---|---|---|
| three-fresh-process AR/MTP matrix | `5d620ef15f1167d548386ad067118fd9` | `215d5eaad1bea3ea356268c1c964dd15` |
| generic sampled-cache + n-gram proof and AR full sessions | `fb1b0d8bac63cc8ca19867a0b5edb662` | `33f7d27831b9ef22c6e746d09f2cad55` |
| gfx1100 registry-DFlash/full spec sessions | `9c21ac994d13ec82204d5282e06abeae` | `39fd92de875eeedebfa00c729a82b742` |
| final build, including the non-neutral-penalty AR guard; gfx1151 DFlash sessions | `4b8c6f03465eaf0f339b92bdffea3b2d` | `bbd3a1defd1fa2c620321e4352c1fb18` |

The final-build change after the gfx1100 DFlash binary only rejects non-neutral repeat/presence/frequency penalties. The recorded registry profile uses neutral values, so that guard does not change those fixture results.

## Short AR versus native MTP matrix

Protocol: three fresh processes per arm; each process used 3 warmups + 5 measured runs, 128 output tokens, Q8 VMM KV, `noslots`, stateless workload, graph capture off, and the byte-identical prompt above. Values below are medians of the three per-process medians.

| arch | tier | AR tok/s | MTP tok/s | delta |
|---|---|---:|---:|---:|
| gfx1100 | XT | 47.4 | 76.4 | +61.2% |
| gfx1100 | Base | 44.7 | 66.1 | +47.9% |
| gfx1100 | Pro | 43.2 | 66.5 | +53.9% |
| gfx1151 | XT | 14.2 | 33.4 | +135.2% |
| gfx1151 | Base | 13.5 | 28.2 | +108.9% |
| gfx1151 | Pro | 13.0 | 27.9 | +114.6% |

Full JSON, including every five-sample array, is preserved at:

- `/home/kaden/qcal/mtp-matrix-gfx1100/matrix.json`
- `/home/kaden/qcal/mtp-matrix-gfx1151/matrix.json`

### The requested gfx1151 40 tok/s threshold was not met

Fresh final-binary checks on MQ4V2 XT, using the same prompt and 5 measured / 3 warmup protocol:

- K7 with `HIPFIRE_MTP_P_MIN=0.6`: median **37.1 tok/s**, samples `37.3, 37.2, 37.1, 37.1, 37.0`.
- K8: median **35.4 tok/s**, samples `35.5, 35.4, 35.4, 35.6, 35.1`.
- The best earlier canonical result in this campaign was 38.2 tok/s.

This checkpoint therefore records the threshold as **not achieved**, rather than rounding or promoting a short outlier.

## Persistent sampled MTP and n-gram composition

A sampled, thinking-enabled three-turn MTP chain on gfx1151 returned prefix-cache counts `0, 66, 117`, generated visible answers `ONE, TWO, ONE`, recalled the first answer, and reported no empty output, attractor, or retrieval miss. Evidence: `/home/kaden/qcal/generic-mtp-sampled-cache-gfx1151.json`.

A separate greedy/thinking-off repetition fixture exercised MTP+n-gram takeover and retirement:

| arch | turn | cached | n-gram windows | drafts | accepted | accept rate | tau | decode tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gfx1100 | 2 | 329 | 5 | 320 | 264 | 0.825 | 30.56 | 100.9 |
| gfx1100 | 3 | 635 | 5 | 320 | 269 | 0.841 | 34.50 | 79.2 |
| gfx1151 | 2 | 329 | 5 | 320 | 264 | 0.825 | 30.56 | 37.1 |
| gfx1151 | 3 | 635 | 5 | 320 | 269 | 0.841 | 34.50 | 28.9 |

All repeated code blocks were byte-identical at the visible surface. The first turn in each process used native MTP only; subsequent turns proved nonzero cache reuse, external n-gram candidates, positive acceptance, and retirement. Evidence:

- `/home/kaden/qcal/mtp-ngram-fixed-gfx1100.json`
- `/home/kaden/qcal/mtp-ngram-fixed-gfx1151.json`

## Full eight-turn sampled-thinking sessions

Protocol: native `scripts/serve_harness.py --mode session`, committed `session_coding.json`, registry tag `qwen3.8:27b`, registry sampling (`temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0`, neutral penalties), `--thinking-effort xhigh`, uncapped thinking, Q8 VMM KV, and seed 7. No explicit `--max-seq` or `--max-tokens` was passed: registry policy resolved 262,144 context tokens and an 81,920-token output allowance. Every completed turn self-terminated. DFlash used the draft-declared all-sliding W=2048 mode. Verify and MTP proposal graphs were explicitly off.

`avg tok/s` is the arithmetic mean of the eight per-turn decode rates, not aggregate throughput. `final ctx` is the eighth request context. Every completed arm had 8/8 thinking turns, no empty output, no runaway, no attractor, and nonzero `cached_tokens` on turns 2–8.

| arch | tier | route | avg tok/s | avg tau | final ctx | strict lexical gate |
|---|---|---|---:|---:|---:|---|
| gfx1100 | XT | AR | 39.75 | — | 30,479 | pass |
| gfx1100 | Base | AR | 37.36 | — | 31,343 | `dedupe` absent on turns 7/8 |
| gfx1100 | Pro | AR | 35.73 | — | 38,468 | pass |
| gfx1100 | XT | DFlash | 15.53 | 2.840 | 32,973 | pass |
| gfx1100 | Base | DFlash | 17.24 | 2.859 | 28,290 | pass |
| gfx1100 | Pro | DFlash | 17.74 | 2.654 | 26,190 | pass |
| gfx1100 | XT | MTP | 13.70 | 1.629 | 24,951 | `dedupe` absent on turn 8 |
| gfx1151 | XT | AR | 13.54 | — | 20,122 | pass |
| gfx1151 | Base | AR | 12.71 | — | 25,876 | `dedupe` absent on turns 7/8 |
| gfx1151 | Pro | AR | 12.30 | — | 23,310 | pass |
| gfx1151 | XT | DFlash | 9.47 | 2.644 | 32,759 | pass |
| gfx1151 | Base | DFlash | 11.64 | 2.839 | 27,981 | pass |

The lexical misses were read manually. They were not empty or incoherent: the responses discussed `content_hash`, `dedup`/deduplication, or the `fdedup` tool but omitted the fixture's exact token `dedupe`. They remain recorded as misses.

Full decoded output and per-turn metadata are preserved under:

- `/home/kaden/qcal/session-gfx1100-xt-ar.json`
- `/home/kaden/qcal/full-sessions-gfx1100/`
- `/home/kaden/qcal/spec-sessions-gfx1100/`
- `/home/kaden/qcal/session-gfx1151-xt-ar.json`
- `/home/kaden/qcal/full-sessions-gfx1151/`
- `/home/kaden/qcal/spec-sessions-gfx1151/`

## Guarded abort and incomplete arms

At `2026-08-21T13:38:47Z`, `gpusentry` reported:

```text
overall=degraded
0000:66:00.0 state=wedged reason=device-lost-from-bus link-down
```

The gfx1100 Base MTP arm had completed six turns with cache counts `0, 4189, 8735, 15443, 23739, 28741`; the endpoint disappeared during turn 7. The gfx1151 Pro DFlash arm had completed four turns with cache counts `0, 3452, 6556, 12543` when its guard observed the same fleet degradation. Both guards returned 124 and used `gpukill` for clean daemon unwind. No reboot, module action, sysfs recovery, or additional GPU run followed.

Excluded from the completed table: gfx1100 Base/Pro MTP, gfx1151 Pro DFlash and all three MTP tiers, and every gfx1201 arm (gfx1201 was absent before the campaign). Partial rows after the guard abort are not acceptance evidence.

## Verification

- `cargo check -p hipfire-daemon -p hipfire-cli` — pass.
- `cargo test -p hipfire-runtime ngram_mod` — 13 passed.
- `cargo test -p hipfire-generate --test qwen_dflash_semantic_terminal_tests --test generation_route_matrix_tests --test continuous_batch` — 81 passed before the final route-capability addition.
- `cargo test -p hipfire-generate --test generation_route_matrix_tests --test continuous_batch` — 18 passed after the final route and penalty guards.
- `cargo build --release` — pass.
- Guarded gfx1100 registry-sampling DFlash smoke after the route fix: visible `READY`, 48 generated tokens, 37 thinking words, `dflash=true`, tau 4.33, 31.0 decode tok/s, no detector flags.

This is historical evidence under the exact fixtures above. The 40 tok/s target and the interrupted arms remain explicitly unresolved.
