# Amendment 2 — compressed MTP draft-head screen

**Lifecycle:** historical
**Disposition:** artifact/mechanism screen; not a gfx1151 performance claim or product default
**Amends:** [`2026-08-21-qwen38-mqv2-mtp-prefix-cache.md`](./2026-08-21-qwen38-mqv2-mtp-prefix-cache.md)

## Reason for amendment

The original checkpoint recorded 38.2 tok/s as the best gfx1151 MQ4V2 XT MTP result and found no bounded source-only lever with enough margin to prove 40 tok/s. The runtime and `mtp_extract` already support FastMTP-style compressed draft heads, but the production Qwen3.8 sidecar contains only the 15 base tensors and therefore reads the full 248,320-row trunk lm-head once per proposal step.

This amendment screens a corpus-derived 32,768-row compressed draft head on gfx1201. It establishes a concrete, already-supported artifact lever with enough measured margin to justify a direct gfx1151 trial after `hipx` recovers. It does **not** substitute the gfx1201 result for the required gfx1151 proof.

## Artifact construction

The vocabulary corpus was built from all 72 assistant outputs in the nine completed gfx1201 `session_coding.json` arms (XT/Base/Pro × AR/DFlash/MTP):

| item | value |
|---|---:|
| corpus records | 72 |
| tokenizer output tokens | 276,185 |
| unique token IDs | 8,923 |
| selected draft vocabulary | 32,768 IDs |
| measured corpus coverage | 100.00% |

`build_mtp_vocab_sidecar.py` force-included the tokenizer's special/added tokens and filled the unused remainder deterministically. `mtp_extract --quant mq4 --vocab-sidecar ...` then extracted the existing Qwen3.8 MTP layer and appended:

- `lm_head_draft.weight`: `[32768, 5120]`, MQ4G256, 89,128,960 bytes
- `lm_head_draft.vocab_map`: `[32768]`, F32-packed token IDs

The resulting 17-tensor container passed `mtp_extract`'s HFQ round-trip verification.

| artifact | md5 |
|---|---|
| compressed MTP sidecar | `d393fcc8bc4b718ec82ecec81287d83a` |
| 32K vocabulary map | `a769032aa2b02a98bb85a9da1256bbe2` |
| 72-output corpus JSONL | `c99e0b9c5289610649820f7c1ece4eee` |
| MQ4V2 XT target | `e45d15bfe0c9a87132697101d17cbed6` |
| original full-vocab MTP sidecar | `1a78dd0d2c8c8a97abfc2f873193ae58` |

No runtime source change was required: this uses the existing compressed-head loader and `spec_step_mtp_compressed_serial_with_k` branch.

## gfx1201 A/B

Hardware: `hiptrx` HIP device 3, Radeon AI PRO R9700 gfx1201, BDF `0000:13:00.0`, HIP 7.14. Binary md5s matched the full-session campaign (`hipfire` `4b8c6f03465eaf0f339b92bdffea3b2d`, daemon `bbd3a1defd1fa2c620321e4352c1fb18`).

Protocol: byte-identical `merge_sort_thinking_off.txt` (md5 `253c7ac50857fe6d0e10fb0d2c5e35c0`), K7, `p_min=0`, Q8 VMM KV, `noslots`, 3 warmups + 5 measured 128-token runs, verify/proposal graphs off.

| arm | decode median | five samples | wall median | tau |
|---|---:|---|---:|---:|
| original full-vocab head | 101.5 tok/s | 101.5, 101.5, 101.5, 101.4, 101.3 | 92.5 | 4.29 |
| compressed 32K head | 113.4 tok/s | 113.4, 113.5, 113.5, 113.4, 113.3 | 102.2 | 4.08 |

The compressed head improved decode by **11.7%** and wall throughput by **10.5%** despite a 4.9% reduction in tau. This clears the raw 4.8% margin that gfx1151 needs to move 38.2 past 40 tok/s, but architecture transfer remains an inference until measured on gfx1151.

An earlier local A/B row named `mtp-cvs32k-gfx1201.json` was rejected: the target was a symlink, model-path canonicalization selected the original adjacent sidecar, and the daemon log proved the compressed head did not load. The accepted compressed result is `mtp-cvs32k-gfx1201-r2.json`; its target is a hardlink and the daemon log names the compressed sidecar explicitly.

## Decoded-output check

A separate five-prompt greedy battery used the compressed sidecar with K7 and graphs off. All five turns stopped naturally with `mtp=true`, no empty output, no runaway, and no attractor. The visible answers contained:

- a correct stable merge of two sorted lists,
- the correct 210-mile calculation,
- an accurate explanation of seasons,
- a coherent lighthouse story,
- five concrete maintainability practices.

Evidence:

- `/home/kaden/qcal/mtp-cvs32k-full-gfx1201.json`
- `/home/kaden/qcal/mtp-cvs32k-gfx1201-r2.json`
- `/home/kaden/qcal/mtp-cvs32k-battery-gfx1201.json`
- `/home/kaden/qcal/qwen3.8-27b.mtp-cvs32k`
- `/home/kaden/qcal/q38-mtp-vocab-32768.json`
- `/home/kaden/qcal/q38-mtp-output-corpus.jsonl`

The compressed sidecar and vocabulary map were also staged under `/home/kaden/qcal/` on `hipx`. Direct gfx1151 A/B and the 40 tok/s claim remain blocked while `hipx` reports gfx1100 link-down and `dstate=16`.
