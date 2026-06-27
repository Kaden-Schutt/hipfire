# hipfire eval summary

- model: `/home/sadara/.hipfire/models/matrix/qwen3.5-9b.oq8.hfq`
- model hash: `hfq:xxh64:bc4da55223ae6c61`
- tier: `fast`
- tier target: `60` seconds (small-model smoke and admission canaries)
- CI suitable: `true`
- hipfire version: `0.3.0`
- runner: `hipfire-eval 0.3.0`
- git commit: `e1241b510e13e54452813af9906f41cb1d3f10c8`
- git branch: `chaingun`
- git describe: `v0.1.0-alpha-3287-ge1241b51-dirty`
- git dirty: `true`
- binary hash: `8b5a2ec9ee2a256f42fb852c5ca1212299bcb4e600c6e8c0877e902f8a0b7431`
- arch: `gfx1103`
- ROCm: `7.14.60850-d34cbb6409`
- hardware bucket: `apu_uma:gfx1103:0x15bf:12cu:1gib:ddr5:128bit:90gbps`
- host profile hash: `fnv64:67c969894d1171d7`
- rows: 2 pass / 0 fail / 0 skip

## Models

| role | identifier | exists | file hash | tag hash | metadata | quantization hash |
|---|---|---|---|---|---|---|
| candidate | /home/sadara/.hipfire/models/matrix/qwen3.5-9b.oq8.hfq | true | hfq:xxh64:bc4da55223ae6c61 |  | pass | {"algorithm":"xxh64","payload_bytes":9080431616,"producer":{"git_branch":"chaingun","git_commit":"e1241b510e13e54452813af9906f41cb1d3f10c8","git_describe":"v0.1.0-alpha-3287-ge1241b51-dirty","git_dirty":true,"hipfire_version":"0.3.0","package":"hipfire-quantize"},"scope":"hfq_tensor_index_and_payload_v1","seed":0,"tensor_count":427,"value":"bc4da55223ae6c61"} |

## Datasets

| suite | status | source | repo | revision | digest | license | selected | selected items | cache | reason |
|---|---|---|---|---|---|---|---:|---|---|---|
| none | Skip | none |  |  |  |  | 0 |  |  | no dataset-backed suites selected |

## Comparisons

- status: `Skip`
- reason: `no --compare or --reference provided`
- cases: `0`

## Admission

- status: `Pass`
- verdict: `measured`
- reason: `no --compare or --reference provided; comparison skipped`
- required evidence: `1`
- findings: `0`

| evidence | status | rows | reason |
|---|---|---|---|
| performance | Pass | 2 |  |

### Observed Evidence

| evidence | status | rows | reason |
|---|---|---|---|
| phase_timings | Pass | 4 |  |
| launch_counts | Pass | 2 |  |
| moe_router_histogram | Skip | 0 | no observed moe_router_histogram evidence rows |
| memory | Pass | 2 |  |
| dflash_trace | Skip | 0 | no observed dflash_trace evidence rows |
| path_c_trace | Skip | 0 | no observed path_c_trace evidence rows |
| module_evidence | Skip | 0 | no observed module_evidence evidence rows |
| coherence | Skip | 0 | no observed coherence evidence rows |
| profiling | Skip | 0 | profiling disabled by --profile off |

## Evidence Artifacts

| artifact | status | path | detail |
|---|---|---|---|
| admission | pass | artifacts/admission.json | measured |
| coherence | not_collected | artifacts/coherence.json | 0 |
| comparisons | skip | artifacts/comparisons.json | 0 |
| dflash_trace | not_collected | artifacts/dflash_trace.json | 0 |
| launch_counts | collected | artifacts/launch_counts.json | 2 |
| memory | collected | artifacts/memory.json | 2 |
| module_evidence | not_collected | artifacts/module_evidence.json | 0 |
| moe_router_histogram | not_collected | artifacts/moe_router_histogram.json | 0 |
| path_c_trace | not_collected | artifacts/path_c_trace.json | 0 |
| performance | collected | artifacts/performance.json | 4 |
| phase_timings | collected | artifacts/phase_timings.json | 4 |
| profiling | disabled | artifacts/profiling.json | 0 |
| quality | not_collected | artifacts/quality.json | 0 |
| run_metadata | collected | artifacts/run_metadata.json |  |

## Rows

| battery | suite | case | item | model | model hash | prompt hash | status | reason |
|---|---|---|---|---|---|---|---|---|
| speed |  | daemon_prefill_decode_first |  | /home/sadara/.hipfire/models/matrix/qwen3.5-9b.oq8.hfq | hfq:xxh64:bc4da55223ae6c61 | 7ea8e9a94c81b85b661956b75532e80562eca61d256a11b6a2a4826bdc63c74b | Pass |  |
| speed |  | daemon_prefill_decode_reset |  | /home/sadara/.hipfire/models/matrix/qwen3.5-9b.oq8.hfq | hfq:xxh64:bc4da55223ae6c61 | 7ea8e9a94c81b85b661956b75532e80562eca61d256a11b6a2a4826bdc63c74b | Pass |  |
