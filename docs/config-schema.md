# hipfire config schema

| Key | Type | Required | Default | Scopes | Mutability | Impact | Description |
|-----|------|----------|---------|--------|------------|--------|-------------|
| `admin_user` | `string` | optional | `admin` | `global`, `runtime` | `static` | `none` | Username for the /admin console login. The password is set separately with `hipfire admin set-password` (argon2id hash stored in ~/.hipfire/admin.passwd, never in config). |
| `cask` | `bool` | optional | `false` | `global`, `model`, `runtime` | `load_time` | `none` | Enable CASK/TriAttention behavior where supported. |
| `cask_auto_attach` | `bool` | optional | `true` | `global`, `model`, `runtime` | `load_time` | `none` | Whether compatible CASK/TriAttention sidecars may auto-attach. |
| `cask_beta` | `u32` | optional | `128` | `global`, `model`, `runtime` | `load_time` | `none` | CASK beta control value. |
| `cask_budget` | `u32` | optional | `512` | `global`, `model`, `runtime` | `load_time` | `none` | CASK token or block budget. |
| `cask_core_frac` | `f64` | optional | `0.5` | `global`, `model`, `runtime` | `load_time` | `none` | Fraction of CASK core candidates to keep. |
| `cask_fold_m` | `u32` | optional | `2` | `global`, `model`, `runtime` | `load_time` | `none` | CASK fold factor. |
| `cask_sidecar` | `path` | required when `cask == true && cask_auto_attach == false` | - | `global`, `model`, `runtime` | `load_time` | `none` | Explicit CASK/TriAttention sidecar path. |
| `cors_allowed_origins` | `json` | optional | `[]` | `global`, `runtime` | `static` | `none` | Browser origins allowed to call the HTTP API cross-origin. Empty disables CORS (same-origin only); ["*"] allows any origin; otherwise an explicit allowlist such as ["http://localhost:8080"]. |
| `default_model` | `string` | optional | - | `global`, `runtime` | `load_time` | `none` | Model tag, alias, or path to pre-load or use by default. |
| `dflash_adaptive_b` | `bool` | optional | `true` | `global`, `model`, `runtime` | `load_time` | `none` | Whether DFlash may adapt draft batch size. |
| `dflash_mode` | `enum(off|auto|on)` | optional | `off` | `global`, `model`, `runtime` | `load_time` | `none` | DFlash speculative decode mode. |
| `dflash_ngram_block` | `json` | optional | `"auto"` | `global`, `model`, `runtime` | `load_time` | `none` | DFlash n-gram blocking policy; accepts boolean or auto. |
| `flash_mode` | `enum(auto|always|never)` | optional | `auto` | `global`, `model`, `runtime` | `load_time` | `none` | Flash-attention selection policy. |
| `gpu_slab_load` | `enum(auto|off|on)` | optional | `auto` | `global`, `model`, `runtime` | `load_time` | `none` | GPU slab loading policy for model weights. |
| `host` | `string` | optional | `127.0.0.1` | `global`, `runtime` | `static` | `none` | Bind host for the OpenAI-compatible HTTP server. Defaults to loopback; set to 0.0.0.0 to expose on all interfaces. |
| `idle_timeout` | `u32` | optional | `300` | `global`, `runtime` | `runtime_reloadable` | `none` | Seconds of inactivity before the server may evict an idle model. |
| `kv_adaptive` | `enum(off|auto)` | optional | `off` | `global`, `model`, `runtime` | `load_time` | `none` | Adaptive KV-cache policy. |
| `kv_cache` | `enum(auto|q8|asym2|asym3|asym4|kvarn2|kvarn|kvarn4|kvarn8)` | optional | `auto` | `global`, `model`, `runtime` | `load_time` | `none` | KV-cache precision and memory policy. |
| `max_seq` | `u32` | optional | `8192` | `global`, `model`, `runtime` | `load_time` | `none` | Maximum context/KV-cache capacity allocated at model load. |
| `max_tokens` | `u32` | optional | `512` | `global`, `model`, `request` | `request_only` | `none` | Default maximum number of generated tokens per request. |
| `mmq_screen` | `enum(auto|off|on)` | optional | `auto` | `global`, `model`, `runtime` | `load_time` | `none` | MMQ safety screening mode. |
| `mmq_screen_threshold` | `f64` | optional | `0.10` | `global`, `model`, `runtime` | `load_time` | `none` | MMQ screening rejection threshold. |
| `model_overrides` | `json` | optional | `{}` | `global`, `model` | `load_time` | `reload_model` | Sparse per-model override map layered on top of global config. |
| `models_network_dir` | `path` | optional | - | `global`, `runtime` | `static` | `none` | Optional extra read-only model root (e.g. an NFS share such as /srv/hipfire). When set, the network-facing server routes resolve model identifiers within this root in addition to ~/.hipfire/models. Unset by default; local CLI/eval callers are unaffected. |
| `mtp_k` | `u32` | optional | `3` | `global`, `model`, `runtime` | `load_time` | `none` | Number of MTP candidate tokens to consider. |
| `mtp_mode` | `enum(auto|off|on)` | optional | `auto` | `global`, `model`, `runtime` | `load_time` | `none` | Multi-token prediction sidecar mode. |
| `port` | `u16` | optional | `11435` | `global`, `runtime` | `static` | `none` | Bind port for the OpenAI-compatible HTTP server. |
| `prefill_alpha` | `f64` | optional | `0.85` | `global`, `model`, `runtime` | `load_time` | `none` | Prefill compression scoring alpha. |
| `prefill_block` | `u32` | optional | `128` | `global`, `model`, `runtime` | `load_time` | `none` | Block size used by prefill compression. |
| `prefill_compression` | `enum(off|auto|on)` | optional | `off` | `global`, `model`, `runtime` | `load_time` | `none` | Long-context prefill compression mode. |
| `prefill_drafter` | `path` | required when `prefill_compression != 'off' && prefill_drafter_device >= 0` | - | `global`, `model`, `runtime` | `load_time` | `none` | Optional drafter artifact for prefill compression. |
| `prefill_drafter_device` | `i32` | optional | `-1` | `global`, `host`, `node`, `model` | `load_time` | `none` | Preferred accelerator device for the prefill drafter. |
| `prefill_keep_ratio` | `f64` | optional | `0.05` | `global`, `model`, `runtime` | `load_time` | `none` | Fraction of prefill blocks to keep under compression. |
| `prefill_min_keep` | `u32` | optional | `2048` | `global`, `model`, `runtime` | `load_time` | `none` | Minimum tokens or blocks retained during prefill compression. |
| `prefill_profile` | `bool` | optional | `false` | `global`, `model`, `runtime` | `load_time` | `none` | Emit prefill compression profiling details. |
| `prefill_recent` | `u32` | optional | `1024` | `global`, `model`, `runtime` | `load_time` | `none` | Recent context size retained during prefill compression. |
| `prefill_sink` | `u32` | optional | `256` | `global`, `model`, `runtime` | `load_time` | `none` | Prefix sink size retained during prefill compression. |
| `prefill_sparse_threshold` | `u32` | optional | `32768` | `global`, `model`, `runtime` | `load_time` | `none` | Context threshold for sparse prefill behavior. |
| `prefill_threshold` | `u32` | optional | `32768` | `global`, `model`, `runtime` | `load_time` | `none` | Context length threshold for prefill compression. |
| `prompt_normalize` | `bool` | optional | `true` | `global`, `model`, `request` | `request_only` | `none` | Whether prompts are normalized before tokenization. |
| `repeat_penalty` | `f64` | optional | `1.05` | `global`, `model`, `request` | `request_only` | `none` | Default repeat penalty for generated text. |
| `sdapi_max_batch_size` | `u32` | optional | `8` | `global`, `runtime` | `static` | `none` | Upper bound on SD API batch_size. |
| `sdapi_max_dimension` | `u32` | optional | `4096` | `global`, `runtime` | `static` | `none` | Upper bound on any single SD API dimension (width/height and their highres/firstphase variants). Requests above it get a 400. The admin's DoS ceiling; clients may request smaller, never larger. |
| `sdapi_max_n_iter` | `u32` | optional | `16` | `global`, `runtime` | `static` | `none` | Upper bound on SD API n_iter. |
| `sdapi_max_steps` | `u32` | optional | `200` | `global`, `runtime` | `static` | `none` | Upper bound on SD API step counts (steps and hr_second_pass_steps). |
| `sdapi_max_total_batches` | `u32` | optional | `32` | `global`, `runtime` | `static` | `none` | Upper bound on batch_size × n_iter (total images generated per request). |
| `sdapi_output_root` | `string` | optional | `/tmp/hipfire-sdapi` | `global`, `runtime` | `static` | `none` | Root directory for images saved by the SD API compatibility routes (save_images: true). Client-supplied outdir_* override_settings are ignored; every SD API image write stays under this root. |
| `temperature` | `f64` | optional | `0.3` | `global`, `model`, `request` | `request_only` | `none` | Default sampling temperature. |
| `thinking` | `enum(off|on)` | optional | `off` | `global`, `model`, `request` | `request_only` | `none` | Reasoning/thinking display policy for compatible models. |
| `top_p` | `f64` | optional | `0.8` | `global`, `model`, `request` | `request_only` | `none` | Default nucleus sampling probability. |
