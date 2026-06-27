# Quant x KV speed matrix

Total rows: 324 | pass: 215 | non-pass: 109

Metric shown: warm-decode `tok_s` (decode_tok_s for qwen) / prefill_tok_s. Pruned-after-bench; numbers are from `hipfire eval --battery speed`.


## Decode tok/s by model x format x KV

| model | format | q8 | asym4 | asym3 | asym2 | fp32 |
|---|---|---|---|---|---|---|
| LFM2.5-350M | q8f16 | 100.0 | · | · | · | · |
| LFM2.5-350M | hfq4 | 153.8 | · | · | · | · |
| LFM2.5-350M | hfq6 | 122.0 | · | · | · | · |
| LFM2.5-350M | mq3 | _fail_ | · | · | · | · |
| LFM2.5-350M | mq4 | 149.5 | · | · | · | · |
| LFM2.5-350M | mq6 | 111.1 | · | · | · | · |
| LFM2.5-350M | oq4 | 135.1 | · | · | · | · |
| LFM2.5-350M | oq8 | _fail_ | · | · | · | · |
| LFM2.5-350M | oq4+ | _fail_ | · | · | · | · |
| LFM2.5-350M | oq4++ | _fail_ | · | · | · | · |
| LFM2.5-350M | oq8+ | _fail_ | · | · | · | · |
| LFM2.5-350M | oq8++ | 94.2 | · | · | · | · |
| qwen3.5-0.8b | q8f16 | 38.8 | 38.7 | 39.3 | 38.8 | · |
| qwen3.5-0.8b | hfq4 | 59.8 | 59.1 | 60.6 | 58.9 | · |
| qwen3.5-0.8b | hfq6 | 49.3 | 48.9 | 49.8 | 48.8 | · |
| qwen3.5-0.8b | mq3 | 64.4 | 63.6 | 65.2 | 65.8 | · |
| qwen3.5-0.8b | mq4 | 59.4 | 58.7 | 59.8 | 58.5 | · |
| qwen3.5-0.8b | mq6 | 49.2 | 48.5 | 49.5 | 48.6 | · |
| qwen3.5-0.8b | oq4 | 59.5 | 58.8 | 60.1 | 58.9 | · |
| qwen3.5-0.8b | oq8 | 43.1 | 42.7 | 43.5 | 42.7 | · |
| qwen3.5-0.8b | oq4+ | 58.0 | 57.0 | 58.8 | 57.5 | · |
| qwen3.5-0.8b | oq4++ | 58.2 | 57.5 | 58.8 | 57.3 | · |
| qwen3.5-0.8b | oq8+ | 41.2 | 40.9 | 41.6 | 40.9 | · |
| qwen3.5-0.8b | oq8++ | 41.3 | 40.9 | 41.5 | 40.7 | · |
| llama-3.2-1b-instruct | q8f16 | 18.8 | · | · | · | 18.5 |
| llama-3.2-1b-instruct | hfq4 | 24.2 | · | · | · | 24.3 |
| llama-3.2-1b-instruct | hfq6 | 21.3 | · | · | · | 14.4 |
| llama-3.2-1b-instruct | mq3 | 18.0 | · | · | · | 15.9 |
| llama-3.2-1b-instruct | mq4 | 24.6 | · | · | · | 14.6 |
| llama-3.2-1b-instruct | mq6 | _fail_ | · | · | · | _fail_ |
| llama-3.2-1b-instruct | oq4 | _fail_ | · | · | · | _fail_ |
| llama-3.2-1b-instruct | oq8 | _fail_ | · | · | · | _fail_ |
| llama-3.2-1b-instruct | oq4+ | _collect_fail_ | · | · | · | _collect_fail_ |
| llama-3.2-1b-instruct | oq4++ | _collect_fail_ | · | · | · | _collect_fail_ |
| llama-3.2-1b-instruct | oq8+ | _collect_fail_ | · | · | · | _collect_fail_ |
| llama-3.2-1b-instruct | oq8++ | _collect_fail_ | · | · | · | _collect_fail_ |
| qwen3.5-2b | q8f16 | 18.0 | 18.0 | 18.0 | 17.9 | · |
| qwen3.5-2b | hfq4 | _fail_ | 28.4 | _fail_ | 28.4 | · |
| qwen3.5-2b | hfq6 | 22.1 | 22.0 | 22.1 | 21.9 | · |
| qwen3.5-2b | mq3 | 32.0 | 31.9 | 32.2 | 31.8 | · |
| qwen3.5-2b | mq4 | 28.5 | 28.3 | 28.6 | 28.3 | · |
| qwen3.5-2b | mq6 | 22.0 | 21.9 | 22.0 | 21.9 | · |
| qwen3.5-2b | oq4 | 27.7 | 27.6 | 28.0 | 27.5 | · |
| qwen3.5-2b | oq8 | 18.7 | 18.6 | 18.7 | 17.0 | · |
| qwen3.5-2b | oq4+ | 27.8 | 27.6 | 27.8 | 27.5 | · |
| qwen3.5-2b | oq4++ | 27.8 | 27.6 | 27.9 | 27.5 | · |
| qwen3.5-2b | oq8+ | 18.7 | 18.6 | 16.9 | 12.2 | · |
| qwen3.5-2b | oq8++ | 18.7 | 18.7 | 11.1 | 10.0 | · |
| qwen3.6-35b-a3b | q8f16 | _quant_fail_ | _quant_fail_ | _quant_fail_ | _quant_fail_ | · |
| qwen3.6-35b-a3b | hfq4 | _quant_fail_ | _quant_fail_ | _quant_fail_ | _quant_fail_ | · |
| qwen3.6-35b-a3b | hfq6 | _quant_fail_ | _quant_fail_ | _quant_fail_ | _quant_fail_ | · |
| qwen3.6-35b-a3b | mq3 | _quant_fail_ | _quant_fail_ | _quant_fail_ | _quant_fail_ | · |
| qwen3.6-35b-a3b | mq4 | _quant_fail_ | _quant_fail_ | _quant_fail_ | _quant_fail_ | · |
| qwen3.6-35b-a3b | mq6 | _quant_fail_ | _quant_fail_ | _quant_fail_ | _quant_fail_ | · |
| qwen3.6-35b-a3b | oq4 | _quant_fail_ | _quant_fail_ | _quant_fail_ | _quant_fail_ | · |
| qwen3.6-35b-a3b | oq8 | _quant_fail_ | _quant_fail_ | _quant_fail_ | _quant_fail_ | · |
| qwen3.6-35b-a3b | oq4+ | _collect_fail_ | _collect_fail_ | _collect_fail_ | _collect_fail_ | · |
| qwen3.6-35b-a3b | oq4++ | _collect_fail_ | _collect_fail_ | _collect_fail_ | _collect_fail_ | · |
| qwen3.6-35b-a3b | oq8+ | _collect_fail_ | _collect_fail_ | _collect_fail_ | _collect_fail_ | · |
| qwen3.6-35b-a3b | oq8++ | _collect_fail_ | _collect_fail_ | _collect_fail_ | _collect_fail_ | · |
| qwen3.5-4b | q8f16 | 8.5 | 7.7 | 7.0 | 4.9 | · |
| qwen3.5-4b | hfq4 | 14.1 | _fail_ | _fail_ | _fail_ | · |
| qwen3.5-4b | hfq6 | 10.2 | 10.1 | 10.2 | 10.1 | · |
| qwen3.5-4b | mq3 | 16.5 | 16.5 | 16.6 | 16.5 | · |
| qwen3.5-4b | mq4 | 14.1 | 14.1 | 14.2 | 14.1 | · |
| qwen3.5-4b | mq6 | 10.1 | 10.1 | 10.2 | 10.1 | · |
| qwen3.5-4b | oq4 | 13.9 | _fail_ | 14.1 | 14.0 | · |
| qwen3.5-4b | oq8 | 8.5 | 5.0 | 4.4 | 4.1 | · |
| qwen3.5-4b | oq4+ | 13.9 | 13.9 | 14.0 | 13.9 | · |
| qwen3.5-4b | oq4++ | 13.9 | _fail_ | 14.0 | 13.9 | · |
| qwen3.5-4b | oq8+ | 7.5 | 4.3 | 3.9 | 3.9 | · |
| qwen3.5-4b | oq8++ | 4.0 | 3.9 | 3.9 | 3.9 | · |
| qwen3.5-9b | q8f16 | 4.3 | 2.2 | 2.0 | 1.9 | · |
| qwen3.5-9b | hfq4 | _fail_ | 9.1 | 9.1 | 9.1 | · |
| qwen3.5-9b | hfq6 | 5.7 | 4.6 | 3.2 | 2.9 | · |
| qwen3.5-9b | mq3 | 11.4 | 11.4 | 11.5 | 11.0 | · |
| qwen3.5-9b | mq4 | 9.0 | _fail_ | 9.1 | 9.0 | · |
| qwen3.5-9b | mq6 | 5.7 | 5.6 | 3.8 | 3.2 | · |
| qwen3.5-9b | oq4 | 8.9 | 8.9 | 8.9 | 8.9 | · |
| qwen3.5-9b | oq8 | 2.9 | 2.3 | 2.1 | 2.1 | · |
| qwen3.5-9b | oq4+ | _collect_fail_ | _collect_fail_ | _collect_fail_ | _collect_fail_ | · |
| qwen3.5-9b | oq4++ | _collect_fail_ | _collect_fail_ | _collect_fail_ | _collect_fail_ | · |
| qwen3.5-9b | oq8+ | _collect_fail_ | _collect_fail_ | _collect_fail_ | _collect_fail_ | · |
| qwen3.5-9b | oq8++ | _collect_fail_ | _collect_fail_ | _collect_fail_ | _collect_fail_ | · |
| qwen3.6-27b | q8f16 | 0.9 | 0.7 | 0.7 | 0.7 | · |
| qwen3.6-27b | hfq4 | _fail_ | 2.9 | 2.5 | 1.6 | · |
| qwen3.6-27b | hfq6 | 2.0 | 1.0 | 1.0 | 0.9 | · |
| qwen3.6-27b | mq3 | 3.6 | 2.4 | 1.7 | 1.6 | · |
| qwen3.6-27b | mq4 | 2.9 | 2.1 | 1.5 | 1.4 | · |
| qwen3.6-27b | mq6 | 2.0 | 1.1 | 1.0 | 0.9 | · |
| qwen3.6-27b | oq4 | 3.7 | 3.7 | 3.4 | 2.5 | · |
| qwen3.6-27b | oq8 | 0.9 | 0.9 | 0.9 | 0.9 | · |
| qwen3.6-27b | oq4+ | _collect_fail_ | _collect_fail_ | _collect_fail_ | _collect_fail_ | · |
| qwen3.6-27b | oq4++ | _collect_fail_ | _collect_fail_ | _collect_fail_ | _collect_fail_ | · |
| qwen3.6-27b | oq8+ | _collect_fail_ | _collect_fail_ | _collect_fail_ | _collect_fail_ | · |
| qwen3.6-27b | oq8++ | _collect_fail_ | _collect_fail_ | _collect_fail_ | _collect_fail_ | · |

## Prefill tok/s by model x format x KV (where reported)

| model | format | q8 | asym4 | asym3 | asym2 | fp32 |
|---|---|---|---|---|---|---|
| qwen3.5-0.8b | q8f16 | 837 | 809 | 835 | 823 | · |
| qwen3.5-0.8b | hfq4 | 1319 | 1285 | 1390 | 1376 | · |
| qwen3.5-0.8b | hfq6 | 893 | 889 | 897 | 883 | · |
| qwen3.5-0.8b | mq3 | 1223 | 1269 | 1241 | 1200 | · |
| qwen3.5-0.8b | mq4 | 1276 | 1278 | 1343 | 1312 | · |
| qwen3.5-0.8b | mq6 | 891 | 882 | 885 | 876 | · |
| qwen3.5-0.8b | oq4 | 1256 | 1250 | 1311 | 1276 | · |
| qwen3.5-0.8b | oq8 | 62 | 62 | 63 | 62 | · |
| qwen3.5-0.8b | oq4+ | 1239 | 1219 | 1242 | 1243 | · |
| qwen3.5-0.8b | oq4++ | 1242 | 1223 | 1239 | 1232 | · |
| qwen3.5-0.8b | oq8+ | 58 | 58 | 59 | 58 | · |
| qwen3.5-0.8b | oq8++ | 59 | 58 | 59 | 58 | · |
| llama-3.2-1b-instruct | q8f16 | 464 | · | · | · | 19 |
| llama-3.2-1b-instruct | hfq4 | 1002 | · | · | · | 25 |
| llama-3.2-1b-instruct | hfq6 | 552 | · | · | · | 17 |
| llama-3.2-1b-instruct | mq3 | 20 | · | · | · | 17 |
| llama-3.2-1b-instruct | mq4 | 988 | · | · | · | 17 |
| qwen3.5-2b | q8f16 | 348 | 345 | 345 | 343 | · |
| qwen3.5-2b | hfq4 | · | 729 | · | 728 | · |
| qwen3.5-2b | hfq6 | 396 | 397 | 398 | 395 | · |
| qwen3.5-2b | mq3 | 581 | 565 | 585 | 578 | · |
| qwen3.5-2b | mq4 | 725 | 713 | 716 | 714 | · |
| qwen3.5-2b | mq6 | 398 | 391 | 395 | 394 | · |
| qwen3.5-2b | oq4 | 513 | 515 | 527 | 503 | · |
| qwen3.5-2b | oq8 | 25 | 25 | 25 | 25 | · |
| qwen3.5-2b | oq4+ | 515 | 510 | 503 | 503 | · |
| qwen3.5-2b | oq4++ | 509 | 519 | 513 | 504 | · |
| qwen3.5-2b | oq8+ | 25 | 25 | 25 | 19 | · |
| qwen3.5-2b | oq8++ | 25 | 25 | 18 | 16 | · |
| qwen3.5-4b | q8f16 | 138 | 126 | 126 | 106 | · |
| qwen3.5-4b | hfq4 | 322 | · | · | · | · |
| qwen3.5-4b | hfq6 | 163 | 162 | 163 | 163 | · |
| qwen3.5-4b | mq3 | 232 | 230 | 232 | 232 | · |
| qwen3.5-4b | mq4 | 322 | 317 | 324 | 316 | · |
| qwen3.5-4b | mq6 | 163 | 161 | 160 | 162 | · |
| qwen3.5-4b | oq4 | 278 | · | 272 | 271 | · |
| qwen3.5-4b | oq8 | 10 | 7 | 5 | 5 | · |
| qwen3.5-4b | oq4+ | 274 | 259 | 274 | 270 | · |
| qwen3.5-4b | oq4++ | 272 | · | 268 | 266 | · |
| qwen3.5-4b | oq8+ | 10 | 6 | 5 | 5 | · |
| qwen3.5-4b | oq8++ | 6 | 5 | 5 | 5 | · |
| qwen3.5-9b | q8f16 | 49 | 42 | 34 | 33 | · |
| qwen3.5-9b | hfq4 | · | 202 | 204 | 204 | · |
| qwen3.5-9b | hfq6 | 78 | 94 | 64 | 48 | · |
| qwen3.5-9b | mq3 | 111 | 117 | 119 | 119 | · |
| qwen3.5-9b | mq4 | 198 | · | 202 | 201 | · |
| qwen3.5-9b | mq6 | 76 | 89 | 70 | 60 | · |
| qwen3.5-9b | oq4 | 75 | 81 | 78 | 84 | · |
| qwen3.5-9b | oq8 | 4 | 3 | 2 | 2 | · |
| qwen3.6-27b | q8f16 | 12 | 10 | 10 | 10 | · |
| qwen3.6-27b | hfq4 | · | 61 | 62 | 61 | · |
| qwen3.6-27b | hfq6 | 21 | 15 | 14 | 14 | · |
| qwen3.6-27b | mq3 | 22 | 24 | 20 | 19 | · |
| qwen3.6-27b | mq4 | 61 | 60 | 61 | 60 | · |
| qwen3.6-27b | mq6 | 23 | 15 | 14 | 14 | · |
| qwen3.6-27b | oq4 | 24 | 24 | 24 | 24 | · |
| qwen3.6-27b | oq8 | 1 | 1 | 1 | 1 | · |

## Non-pass cells

| model | format | status | kv | reason |
|---|---|---|---|---|
| LFM2.5-350M | mq3 | fail | q8 | daemon speed anchor returned empty; zero-token; or replacement-character output; |
| LFM2.5-350M | oq8 | fail | q8 | daemon speed anchor returned empty; zero-token; or replacement-character output; |
| llama-3.2-1b-instruct | mq6 | fail | q8 | daemon load failed for speed eval model /home/sadara/.hipfire/models/matrix/llam |
| llama-3.2-1b-instruct | oq4 | fail | q8 | daemon load failed for speed eval model /home/sadara/.hipfire/models/matrix/llam |
| llama-3.2-1b-instruct | oq8 | fail | q8 | daemon load failed for speed eval model /home/sadara/.hipfire/models/matrix/llam |
| qwen3.5-2b | hfq4 | fail | q8 | daemon-backed speed executor failed: daemon stdout closed unexpectedly; |
| qwen3.6-35b-a3b | q8f16 | quant_fail | q8 | HFQ input pipeline failed: open HFQ input: HFQM tensor model.language_model.laye |
| qwen3.6-35b-a3b | hfq4 | quant_fail | q8 | HFQ input pipeline failed: open HFQ input: HFQM tensor model.language_model.laye |
| qwen3.6-35b-a3b | hfq6 | quant_fail | q8 | HFQ input pipeline failed: open HFQ input: HFQM tensor model.language_model.laye |
| qwen3.6-35b-a3b | mq3 | quant_fail | q8 | HFQ input pipeline failed: open HFQ input: HFQM tensor model.language_model.laye |
| qwen3.6-35b-a3b | mq4 | quant_fail | q8 | HFQ input pipeline failed: open HFQ input: HFQM tensor model.language_model.laye |
| qwen3.6-35b-a3b | mq6 | quant_fail | q8 | HFQ input pipeline failed: open HFQ input: HFQM tensor model.language_model.laye |
| qwen3.6-35b-a3b | oq4 | quant_fail | q8 | HFQ input pipeline failed: open HFQ input: HFQM tensor model.language_model.laye |
| qwen3.6-35b-a3b | oq8 | quant_fail | q8 | HFQ input pipeline failed: open HFQ input: HFQM tensor model.language_model.laye |
| qwen3.5-4b | hfq4 | fail | asym4 | daemon-backed speed executor failed: daemon stdout closed unexpectedly; |
| qwen3.5-4b | hfq4 | fail | asym3 | [chat_template] using HFQ-embedded tokenizer_config.chat_template; |
| qwen3.5-4b | oq4 | fail | asym4 | daemon-backed speed executor failed: daemon stdout closed unexpectedly; |
| qwen3.5-9b | hfq4 | fail | q8 | daemon-backed speed executor failed: daemon stdout closed unexpectedly; |
| qwen3.5-9b | mq4 | fail | asym4 | daemon-backed speed executor failed: daemon stdout closed unexpectedly; |
| qwen3.6-27b | hfq4 | fail | q8 | [chat_template] using HFQ-embedded tokenizer_config.chat_template; |
| LFM2.5-350M | oq4+ | fail | q8 | daemon speed anchor returned empty; zero-token; or replacement-character output; |
| LFM2.5-350M | oq4++ | fail | q8 | daemon speed anchor returned empty; zero-token; or replacement-character output; |
| LFM2.5-350M | oq8+ | fail | q8 | daemon speed anchor returned empty; zero-token; or replacement-character output; |
| llama-3.2-1b-instruct | oq4+ | collect_fail | q8 | note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace; |
| llama-3.2-1b-instruct | oq4++ | collect_fail | q8 | note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace; |
| llama-3.2-1b-instruct | oq8+ | collect_fail | q8 | note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace; |
| llama-3.2-1b-instruct | oq8++ | collect_fail | q8 | note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace; |
| qwen3.6-35b-a3b | oq4+ | collect_fail | q8 | note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace; |
| qwen3.6-35b-a3b | oq4++ | collect_fail | q8 | note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace; |
| qwen3.6-35b-a3b | oq8+ | collect_fail | q8 | note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace; |
| qwen3.6-35b-a3b | oq8++ | collect_fail | q8 | note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace; |
| qwen3.5-4b | oq4++ | fail | asym4 | daemon-backed speed executor failed: daemon stdout closed unexpectedly; |
| qwen3.5-9b | oq4+ | collect_fail | q8 | note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace; |
| qwen3.5-9b | oq4++ | collect_fail | q8 | note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace; |
| qwen3.5-9b | oq8+ | collect_fail | q8 | note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace; |
| qwen3.5-9b | oq8++ | collect_fail | q8 | note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace; |
| qwen3.6-27b | oq4+ | collect_fail | q8 | note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace; |
| qwen3.6-27b | oq4++ | collect_fail | q8 | note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace; |
| qwen3.6-27b | oq8+ | collect_fail | q8 | note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace; |
| qwen3.6-27b | oq8++ | collect_fail | q8 | note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace; |
