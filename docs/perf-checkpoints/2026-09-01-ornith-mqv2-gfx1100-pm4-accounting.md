# Ornith MQV2 — gfx1100 retained-PM4 accounting and reorder screen

**Date:** 2026-09-01 UTC
**Lifecycle:** `historical`
**Authority:** Dated, fixture-bound **Measured** diagnostic evidence only.
**Disposition:** PM4 accounting retained; existing width-reorder candidate rejected for gfx1100. Not a current product baseline, admission result, performance floor, SLA, or transferable claim.

This record follows `2026-09-01-ornith-mqv2-v1-v2-gfx1100-paired-profile.md`. That profile found a retained-PM4 sign reversal and a 187-dword difference between its recorded V1 and V2 tapes. This screen adds deterministic preparation-time packet/register accounting, reruns a controlled V1/V2 pair, and tests whether the existing bounded conflict-DAG reorder removes any of the measured command growth.

## Plan and expected outcome

1. Account every retained dword by packet opcode and packet width.
2. Attribute every `SET_SH_REG` payload to its first register and following kernel transition.
3. Reconcile the accounting sum exactly to the frozen stream length without mutating or retaining diagnostic state.
4. Only if the accounting exposes avoidable state transitions, admit exact gfx1100 to the already default-off bounded reorder experiment and run the PM4 shadow oracle.
5. Retain a compaction change only when it reduces dwords and preserves exact state parity.

The guaranteed outcome was attribution with zero default behavior change. A compaction win was conditional: if the reorder removed `R` dwords, the expected V2 stream was `16920 - R`, with `0 < R <= 187`. `R = 0` was an explicit rejection outcome, not a failed deliverable.

## Fixture

| field | value |
|---|---|
| Host | `hipx` |
| Worktree | `/home/kaden/mqv2-paired-profile` |
| GPU route | `HIP_VISIBLE_DEVICES=0` |
| Device proof | logs report `GPU dev 0: gfx1100` and `.hipfire_kernels/gfx1100` |
| Accounting commit | `fee8f9c4a` |
| Accounting / Redline daemon md5 | `b77820aa15f4698704cb6ccccb8e562a` |
| Final serving-check daemon md5 | `5973e154fe6ac3e978708b19159fd4fd` |
| Final serving-check CLI md5 | `9a5077ca6e2a347cfecd85d393f4f92a` |
| Temporary reorder-screen daemon md5 | `8e4ec7ddd025bd5ee81b4bd0dae4fbb3` |
| V1 artifact | `/mnt/nas/kaden/hipfire/models/Qwen3.6-35b-a3b/qwen3.6-35b-a3b.mq4r` |
| V1 SHA-256 | `4685c140c46b1a6f31a0fd9053bf09d5faf1d2529d715b84794249b66cde0428` |
| V2 artifact | `/mnt/nas/kaden/hipfire/models/Ornith-1.5-35B-A3B/ornith-1.5-35b-a3b.mq4r` (official zero-qt13) |
| V2 SHA-256 | `84103fcc8ade42aa2ac8ec01176df7a4ead5e94810597c9fae2f6763152a3ac6` |
| KV / state | Q8 KV; production-default Q8 error-feedback DeltaNet state |
| Decode context | 128 |
| PM4 policy | one queue, `static` register policy, resource waits, required-only acquires, GCR trim on |

Accounting is opt-in with `HIPFIRE_REPLAY_PM4_STREAM_ACCOUNTING=1`. It runs after terminal-idle encoding and before retained graph creation, logs one deterministic report, and is dropped. The unset path does not perform the diagnostic walk.

Both controlled arms used:

```bash
HIP_VISIBLE_DEVICES=0 \
HIPFIRE_REPLAY_TRANSPORT=pm4 \
HIPFIRE_REPLAY_PM4_STATEFUL=static \
HIPFIRE_REPLAY_PM4_WAIT_POLICY=resource \
HIPFIRE_REPLAY_PM4_ACQUIRE_POLICY=required-only \
HIPFIRE_REPLAY_PM4_GCR_TRIM=1 \
HIPFIRE_REPLAY_PM4_NATIVE_PHASES=0 \
HIPFIRE_REPLAY_PM4_DYNAMIC_GRID=0 \
HIPFIRE_REPLAY_PM4_STREAM_ACCOUNTING=1 \
python3 scripts/redline_daemon_harness.py \
  --model "$MODEL" --daemon target-profile/release/daemon \
  --skip-prefill --decode-context 128 --capture-repeats 2 \
  --measure-repeats 3 --decode-iterations 16 \
  --shadow-iterations 2 --max-seq 512 --pm4
```

## Controlled V1/V2 accounting result

| counter | V1 | V2 | V2 − V1 |
|---|---:|---:|---:|
| Dispatches | 603 | 603 | 0 |
| Dependency waits | 492 | 492 | 0 |
| Dependency acquires | 492 | 492 | 0 |
| Command dwords | 16,773 | 16,920 | **+147** |
| `SET_SH_REG` packets | 2,767 | 2,816 | **+49** |
| `SET_SH_REG` value dwords | 3,294 | 3,343 | **+49** |
| Repeated emitted values | 0 | 0 | 0 |

Exact packet-class delta:

| packet | V1 | V2 | delta dwords |
|---|---:|---:|---:|
| `DISPATCH_DIRECT`, 5 dwords | 603 | 603 | 0 |
| `EVENT_WRITE`, 2 dwords | 493 | 493 | 0 |
| `ACQUIRE_MEM`, 8 dwords | 493 | 493 | 0 |
| `SET_SH_REG`, 3 dwords | 2,241 | 2,290 | **+147** |
| `SET_SH_REG`, 4 dwords | 525 | 525 | 0 |
| `SET_SH_REG`, 5 dwords | 1 | 1 | 0 |

The entire controlled **+147 dwords** reconcile to **49 additional three-dword `SET_SH_REG` packets**. Every register-write counter is identical except offset `0x228`:

| GFX10/GFX11 register | V1 writes | V2 writes | delta |
|---|---:|---:|---:|
| `COMPUTE_PGM_RSRC3_GFX10` (`0x228`) | 554 | 603 | **+49** |

No emitted value restated the previous visible value. Therefore the existing stateful register encoder had no redundant consecutive write to delete in this controlled V2 tape; the extra packets reflect real `PGM_RSRC3` transitions between adjacent loaded kernel descriptors.

This controlled rerun does **not** silently rewrite the earlier `+187` observation. Its V1 execution sequence is different (`0d96787fe0a263e5`, 22 unique kernels), while V2 is `b75038a329d2c827` (20 unique kernels). The earlier checkpoint remains authoritative for its own `16,733`/`16,920` tapes. This run explains the current pair's `+147`; the remaining 40 dwords in the earlier cross-fixture gap are not attributed here.

Raw reports:

- `hipx:/home/kaden/mqv2-paired-profile/profiles/pm4-accounting-v1.{json,log}`
- `hipx:/home/kaden/mqv2-paired-profile/profiles/pm4-accounting-v2.{json,log}`

Both Redline reports passed. Their full-state PM4 shadows reported `exact=true` and `gdn_frame_exact=true`.

## Existing bounded reorder screen

A temporary, unretained candidate admitted exact gfx1100 to the existing default-off `HIPFIRE_REPLAY_PM4_SINGLE_IB_REORDER` conflict-DAG scheduler. It changed no scheduler logic and no default. The official zero-qt13 V2 artifact was screened at four windows:

| window | moved launches | max displacement | command dwords | exact state parity |
|---|---:|---:|---:|---|
| 8 | 10 / 603 | 1 | 16,920 | yes |
| 16 | 4 / 603 | 1 | 16,920 | yes |
| 32 | 2 / 603 | 1 | 16,920 | yes |
| max | 0 / 603 | 0 | 16,920 | yes |

The screen produced `R = 0` at every window. The scheduler can widen nearby independent work, but it did not coalesce any `PGM_RSRC3` transition in this tape. The gfx1100 admission patch was therefore dropped and was not pushed.

A 64-position A/B/B/A shadow timing screen used the temporary candidate binary above:

| arm | total PM4 host time for 64 positions | derived tok/s |
|---|---:|---:|
| A1, reorder off | 275,035.6 µs | 232.70 |
| B1, window 8 | 274,382.6 µs | 233.25 |
| B2, window 8 | 287,688.6 µs | 222.46 |
| A2, reorder off | 280,370.3 µs | 228.27 |
| A mean | 277,703.0 µs | 230.46 |
| B mean | 281,035.6 µs | 227.73 |

The paired mean is **−1.19%** for the reorder arm, inside the observed run spread and accompanied by zero dword reduction. It is a rejection result, not a regression claim.

Raw screen reports:

- `hipx:/home/kaden/mqv2-paired-profile/profiles/pm4-reorder-v2-{w8,w16,w32,max}.{json,log}`
- `hipx:/home/kaden/mqv2-paired-profile/profiles/pm4-screen-{a1-off,b1-w8,b2-w8,a2-off}.{json,log}`

## Final serving check

The retained accounting commit was rebuilt after dropping the reorder candidate. `scripts/serve_harness.py` generated 96 greedy tokens from committed prompt `benchmarks/prompts/bare_factual.txt`:

- prompt md5: `1d32df5f12c414d3e34c7b35b6611e6c`
- observed route: `transport=pm4`, `replays=96`
- prepared stream: 603 dispatches, 16,920 dwords
- decoded response correctly named Paris, the Seine, and the Eiffel Tower, then began the requested historical explanation
- decode: 240.0 tok/s in this single behavioral run; diagnostic only, not a baseline claim
- no empty output or token attractor was observed

Artifacts:

- `hipx:/home/kaden/mqv2-paired-profile/profiles/serve-pm4-accounting.json`
- `hipx:/home/kaden/mqv2-paired-profile/profiles/serve-pm4-accounting.log`

## Disposition

- **Retained:** deterministic opt-in PM4 stream accounting and its exact reconciliation checks.
- **Rejected:** gfx1100 admission to the existing width reorder (`R = 0`; no admitted throughput benefit).
- **Next PM4 lever:** if pursued, it must target `COMPUTE_PGM_RSRC3_GFX10` transition locality explicitly while preserving the conflict-DAG order. The existing width-oriented scheduler is not that compiler pass.
