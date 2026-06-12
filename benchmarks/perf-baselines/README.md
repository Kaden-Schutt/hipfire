# Hipfire Perf Baselines

Committed performance baselines live here when a regression gate needs a
source-controlled comparison point. Fresh measurements and full evidence
bundles should be produced by `hipfire eval` under `~/.hipfire/eval-results/`;
files in this directory are curated snapshots only.

Baseline filenames use:

```text
<gfx>-<hardware-profile-hash>.json
```

The hardware profile hash is a stable identity for the machine class, not the
benchmark result. New baselines should use the `host_profile_hash` emitted by
`hipfire eval`, which includes GPU model, GFX target, CU count, memory class,
memory width/clock/bandwidth, memory aperture, and device IDs. Legacy converted
baselines may use a documented `legacy-derived` hash until recaptured through
the eval harness.

Each file may contain multiple suites under `baselines`:

```json
{
  "schema": "hipfire.perf_baseline.v1",
  "arch": "gfx1151",
  "hardware_profile_hash": "...",
  "baselines": {
    "speed": [
      {
        "label": "4b_pp32_prefill_decode",
        "model_id": "qwen3.5-4b-mq4",
        "model_size": "4b",
        "format": "mq4",
        "prefill_tokens": 32,
        "prefill_tok_s": 590.7,
        "gen_tok_s": 65.5
      }
    ],
    "pflash_niah": []
  }
}
```

Regression gates should compare exactly when the hardware profile matches. For
same-arch but different speed-class comparisons, report drift as context rather
than treating it as an apples-to-apples failure.

Speed rows must include `model_id`, normally the canonical `.hfq` stem without
the extension. Gates match on `model_id` and `prefill_tokens`; `model_size` and
`format` are metadata for grouping/reporting, not sufficient identity by
themselves.
