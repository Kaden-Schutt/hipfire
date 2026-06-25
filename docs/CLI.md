# Command-Line Help for `hipfire`

This document contains the help content for the `hipfire` command-line program.

**Command Overview:**

* [`hipfire`↴](#hipfire)
* [`hipfire serve`↴](#hipfire-serve)
* [`hipfire chat`↴](#hipfire-chat)
* [`hipfire list`↴](#hipfire-list)
* [`hipfire eval`↴](#hipfire-eval)
* [`hipfire host-profile`↴](#hipfire-host-profile)
* [`hipfire collect-artifacts`↴](#hipfire-collect-artifacts)
* [`hipfire repack`↴](#hipfire-repack)
* [`hipfire lock`↴](#hipfire-lock)
* [`hipfire lock acquire`↴](#hipfire-lock-acquire)
* [`hipfire lock release`↴](#hipfire-lock-release)
* [`hipfire lock status`↴](#hipfire-lock-status)
* [`hipfire diffusion`↴](#hipfire-diffusion)
* [`hipfire diffusion import`↴](#hipfire-diffusion-import)
* [`hipfire diffusion inspect`↴](#hipfire-diffusion-inspect)
* [`hipfire diffusion preflight`↴](#hipfire-diffusion-preflight)
* [`hipfire diffusion txt2img`↴](#hipfire-diffusion-txt2img)
* [`hipfire diffusion img2img`↴](#hipfire-diffusion-img2img)
* [`hipfire diffusion smoke`↴](#hipfire-diffusion-smoke)
* [`hipfire admin`↴](#hipfire-admin)
* [`hipfire admin status`↴](#hipfire-admin-status)
* [`hipfire admin chat`↴](#hipfire-admin-chat)
* [`hipfire admin health`↴](#hipfire-admin-health)
* [`hipfire admin models`↴](#hipfire-admin-models)
* [`hipfire admin config`↴](#hipfire-admin-config)
* [`hipfire admin training`↴](#hipfire-admin-training)
* [`hipfire admin diagnostics`↴](#hipfire-admin-diagnostics)
* [`hipfire admin logs`↴](#hipfire-admin-logs)
* [`hipfire admin get`↴](#hipfire-admin-get)
* [`hipfire admin set-password`↴](#hipfire-admin-set-password)

## `hipfire`

hipfire LLM inference CLI

**Usage:** `hipfire <COMMAND>`

###### **Subcommands:**

* `serve` — Start the hipfire HTTP server (OpenAI-compatible)
* `chat` — Load a model and generate a response (one-shot)
* `list` — List locally available models
* `eval` — Run the quant admission/model evaluation harness
* `host-profile` — Measure host, GPU-copy, and model storage bandwidth
* `collect-artifacts` — Collect Tier-1 calibration artifacts (Hessian/imatrix/router-histogram) in one model load
* `repack` — Reshuffle a canonical .hfq into an arch-optimal layout (<model>.<arch>.hfq)
* `lock` — GPU resource lock for multi-agent coordination (acquire/release/status)
* `diffusion` — Import and inspect diffusion models stored as .hfq artifacts
* `admin` — Query the running hipfire admin API for scripts and agents



## `hipfire serve`

Start the hipfire HTTP server (OpenAI-compatible)

**Usage:** `hipfire serve [OPTIONS]`

###### **Options:**

* `--host <HOST>` — Override bind host
* `-p`, `--port <PORT>` — Override bind port
* `-m`, `--model <MODEL>` — Pre-load a model on startup
* `--debug-chat` — Log full raw chat requests and raw model replies



## `hipfire chat`

Load a model and generate a response (one-shot)

**Usage:** `hipfire chat [OPTIONS] <PROMPT>`

###### **Arguments:**

* `<PROMPT>` — Prompt text

###### **Options:**

* `-m`, `--model <MODEL>` — Model name, alias, or path. Falls back to the `default_model` config value when omitted
* `--max-tokens <MAX_TOKENS>` — Max tokens to generate
* `--temperature <TEMPERATURE>` — Sampling temperature
* `--attach <FILE>` — Attach a file to the prompt (repeatable). The type is detected from the extension. Only images are wired today (PNG/JPEG/WebP/GIF/BMP); text, video, and audio are recognized but not yet supported and will error



## `hipfire list`

List locally available models

**Usage:** `hipfire list`



## `hipfire eval`

Run the quant admission/model evaluation harness

**Usage:** `hipfire eval [ARGS]...`

###### **Arguments:**

* `<ARGS>` — Arguments forwarded to hipfire-eval



## `hipfire host-profile`

Measure host, GPU-copy, and model storage bandwidth

**Usage:** `hipfire host-profile [ARGS]...`

###### **Arguments:**

* `<ARGS>` — Arguments forwarded to hipfire-host-profile



## `hipfire collect-artifacts`

Collect Tier-1 calibration artifacts (Hessian/imatrix/router-histogram) in one model load

**Usage:** `hipfire collect-artifacts [ARGS]...`

###### **Arguments:**

* `<ARGS>` — Arguments forwarded to the collect_artifacts runner



## `hipfire repack`

Reshuffle a canonical .hfq into an arch-optimal layout (<model>.<arch>.hfq)

**Usage:** `hipfire repack [ARGS]...`

###### **Arguments:**

* `<ARGS>` — Arguments forwarded to the oq4_repack runner



## `hipfire lock`

GPU resource lock for multi-agent coordination (acquire/release/status)

**Usage:** `hipfire lock <COMMAND>`

###### **Subcommands:**

* `acquire` — Acquire the GPU lock (blocks until free). A detached holder keeps it until `release` or the calling shell exits
* `release` — Release the GPU lock (SIGTERM the holder recorded in the lockfile)
* `status` — Print lock status: "gpu is free" or "gpu BUSY: <holder>"



## `hipfire lock acquire`

Acquire the GPU lock (blocks until free). A detached holder keeps it until `release` or the calling shell exits

**Usage:** `hipfire lock acquire [OPTIONS] <LABEL>`

###### **Arguments:**

* `<LABEL>` — Human label recorded in the lockfile (who/what holds it)

###### **Options:**

* `--watch-pid <WATCH_PID>` — Pid whose death auto-releases the lock (default: the calling shell)
* `--timeout-secs <TIMEOUT_SECS>` — Hard cap in seconds to wait for a busy lock; 0 = wait forever

  Default value: `1800`
* `--poll-secs <POLL_SECS>` — Cadence of "busy" messages while waiting, in seconds

  Default value: `5`



## `hipfire lock release`

Release the GPU lock (SIGTERM the holder recorded in the lockfile)

**Usage:** `hipfire lock release`



## `hipfire lock status`

Print lock status: "gpu is free" or "gpu BUSY: <holder>"

**Usage:** `hipfire lock status`



## `hipfire diffusion`

Import and inspect diffusion models stored as .hfq artifacts

Runtime note: runnable `.hfq` diffusion artifacts currently use the native
UNet-family CPU reference path for CLIP text conditioning, UNet denoising, VAE
decode, and VAE encode for img2img. ROCm preflight can validate the planned
device buffers plus individual diffusion kernels for model-input scaling,
classifier-free guidance, Euler scheduler updates, f32 NCHW Conv2D, f32 NCHW
GroupNorm, and RGB conversion, but full generation is not routed through a GPU
UNet runtime yet.
The runtime accepts Q4F16_G64, f16, bf16, f32, Q8F16, Q4_K, HFQ4G128,
HFQ4G256, and HFQ6G256 tensor payloads even when the artifact `weight_format`
records a future quantized format such as `oq4`; OQ/MQ/HFP and other packed
payloads still require a matching diffusion dequantizer/HIP runtime before they
can generate images.

**Usage:** `hipfire diffusion <COMMAND>`

###### **Subcommands:**

* `import` — Convert a Diffusers snapshot or single-file checkpoint into a Hipfire .hfq artifact
* `inspect` — Inspect a diffusion .hfq artifact and print its server-facing summary
* `preflight` — Plan HIP diffusion buffers and optionally run a ROCm device preflight
* `txt2img` — Generate PNG images directly from a diffusion .hfq artifact
* `img2img` — Generate PNG images from init images with a diffusion .hfq artifact
* `smoke` — Run an end-to-end diffusion admission smoke and validate output PNGs



## `hipfire diffusion import`

Convert a Diffusers snapshot or single-file checkpoint into a Hipfire .hfq artifact.

The importer extracts tensors from common Diffusers single-file and sharded safetensors layouts first, then falls back to legacy PyTorch .bin archives or opaque source weight entries when a component cannot be indexed yet.

**Usage:** `hipfire diffusion import [OPTIONS] --output <OUTPUT> <SOURCE>`

###### **Arguments:**

* `<SOURCE>` — Diffusers snapshot directory containing model_index.json, or a .safetensors/.ckpt checkpoint

###### **Options:**

* `-o`, `--output <OUTPUT>` — Output .hfq artifact path
* `--model-name <MODEL_NAME>` — Model name to store in the diffusion metadata; defaults to the source directory name
* `--max-batch <MAX_BATCH>` — Maximum batch size declared by the artifact. Runtime kernels may cap this lower initially

  Default value: `1`
* `--metadata-only` — Import configs/tokenizers only and skip weight indexing for fast planning/inspection



## `hipfire diffusion inspect`

Inspect a diffusion .hfq artifact and print its server-facing summary

**Usage:** `hipfire diffusion inspect <MODEL>`

###### **Arguments:**

* `<MODEL>` — Diffusion .hfq artifact to inspect



## `hipfire diffusion preflight`

Plan HIP diffusion buffers and optionally run a ROCm device preflight

The preflight command prints a deterministic memory plan for the requested
resolution, batch, scheduler, and prompt set. Builds compiled with
`--features rocm` also initialize the selected HIP device, allocate the planned
buffer classes, run a small host/device roundtrip probe, and launch diffusion
kernels for model-input scaling, classifier-free guidance, Euler scheduler
updates, f32 NCHW Conv2D, f32 NCHW GroupNorm, and RGB conversion. Each kernel
probe is checked against the CPU reference and reported in the JSON output.

**Usage:** `hipfire diffusion preflight [OPTIONS] --model <MODEL>`

###### **Options:**

* `-m`, `--model <MODEL>` — Diffusion .hfq artifact to inspect
* `-p`, `--prompt <PROMPT>` — Prompt text. Repeat for batched planning, or use --batch-size with one prompt

  Default value: `hipfire diffusion preflight`
* `--negative-prompt <NEGATIVE_PROMPT>` — Negative prompt text. Omit for empty negatives, pass once to reuse, or repeat per prompt
* `--width <WIDTH>` — Output image width in pixels

  Default value: `512`
* `--height <HEIGHT>` — Output image height in pixels

  Default value: `512`
* `--steps <STEPS>` — Denoising steps

  Default value: `20`
* `--cfg-scale <CFG_SCALE>` — Classifier-free guidance scale

  Default value: `7`
* `--scheduler <SCHEDULER>` — Scheduler/sampler name

  Default value: `Automatic`
* `--seed <SEED>` — Seed. Omit for zero, pass once to reuse, or repeat per prompt
* `--subseed <SUBSEED>` — Optional subseed. Pass once to reuse or repeat per prompt
* `--subseed-strength <SUBSEED_STRENGTH>` — Blend strength for subseed latents

  Default value: `0`
* `--batch-size <BATCH_SIZE>` — Batch size when a single prompt is supplied

  Default value: `1`
* `--device-id <DEVICE_ID>` — ROCm device id to preflight when built with --features rocm

  Default value: `0`



## `hipfire diffusion txt2img`

Generate PNG images directly from a diffusion .hfq artifact

**Usage:** `hipfire diffusion txt2img [OPTIONS] --model <MODEL> --prompt <PROMPT> --output <OUTPUT>`

###### **Options:**

* `-m`, `--model <MODEL>` — Diffusion .hfq artifact to run
* `-p`, `--prompt <PROMPT>` — Prompt text. Repeat for batched generation, or use --batch-size with one prompt
* `--negative-prompt <NEGATIVE_PROMPT>` — Negative prompt text. Omit for empty negatives, pass once to reuse, or repeat per prompt
* `-o`, `--output <OUTPUT>` — Output PNG file for one image, or output directory for batches
* `--width <WIDTH>` — Output image width in pixels

  Default value: `512`
* `--height <HEIGHT>` — Output image height in pixels

  Default value: `512`
* `--steps <STEPS>` — Denoising steps

  Default value: `20`
* `--cfg-scale <CFG_SCALE>` — Classifier-free guidance scale

  Default value: `7`
* `--scheduler <SCHEDULER>` — Scheduler/sampler name, such as Automatic, Euler, Euler Karras, DDIM, or DPM++ 2M Karras

  Default value: `Automatic`
* `--seed <SEED>` — Seed. Omit for zero, pass once to reuse, or repeat per prompt
* `--subseed <SUBSEED>` — Optional subseed. Pass once to reuse or repeat per prompt
* `--subseed-strength <SUBSEED_STRENGTH>` — Blend strength for subseed latents

  Default value: `0`
* `--batch-size <BATCH_SIZE>` — Batch size when a single prompt is supplied

  Default value: `1`



## `hipfire diffusion img2img`

Generate PNG images from init images with a diffusion .hfq artifact

**Usage:** `hipfire diffusion img2img [OPTIONS] --model <MODEL> --prompt <PROMPT> --init-image <INIT_IMAGE> --output <OUTPUT>`

###### **Options:**

* `-m`, `--model <MODEL>` — Diffusion .hfq artifact to run
* `-p`, `--prompt <PROMPT>` — Prompt text. Repeat for batched generation, or use --batch-size with one prompt
* `--negative-prompt <NEGATIVE_PROMPT>` — Negative prompt text. Omit for empty negatives, pass once to reuse, or repeat per prompt
* `--init-image <INIT_IMAGE>` — Input image path. Repeat for an image batch, or pass once to reuse across prompts
* `--mask <MASK>` — Optional mask image path for inpaint-capable artifacts
* `-o`, `--output <OUTPUT>` — Output PNG file for one image, or output directory for batches
* `--width <WIDTH>` — Output image width in pixels. Defaults to the init image width
* `--height <HEIGHT>` — Output image height in pixels. Defaults to the init image height
* `--steps <STEPS>` — Denoising steps

  Default value: `20`
* `--cfg-scale <CFG_SCALE>` — Classifier-free guidance scale

  Default value: `7`
* `--scheduler <SCHEDULER>` — Scheduler/sampler name, such as Automatic, Euler, Euler Karras, DDIM, or DPM++ 2M Karras

  Default value: `Automatic`
* `--seed <SEED>` — Seed. Omit for zero, pass once to reuse, or repeat per prompt
* `--subseed <SUBSEED>` — Optional subseed. Pass once to reuse or repeat per prompt
* `--subseed-strength <SUBSEED_STRENGTH>` — Blend strength for subseed latents

  Default value: `0`
* `--batch-size <BATCH_SIZE>` — Batch size when a single prompt is supplied

  Default value: `1`
* `--denoising-strength <DENOISING_STRENGTH>` — Img2img denoising strength in [0, 1]

  Default value: `0.75`



## `hipfire diffusion smoke`

Run an end-to-end diffusion admission smoke and validate output PNGs

The smoke command validates PNG dimensions and rejects visually degenerate
single-color or near-flat outputs. The JSON report includes per-image pixel
statistics (`unique_rgb_values`, RGB range, and luma range) for admission
evidence.

**Usage:** `hipfire diffusion smoke [OPTIONS] --model <MODEL>`

###### **Options:**

* `-m`, `--model <MODEL>` — Diffusion .hfq artifact to run
* `-p`, `--prompt <PROMPT>` — Prompt text for the smoke run

  Default value: `hipfire diffusion smoke test`
* `--negative-prompt <NEGATIVE_PROMPT>` — Negative prompt text

  Default value: ``
* `--output-dir <OUTPUT_DIR>` — Output directory for smoke PNGs

  Default value: `/tmp/hipfire-diffusion-smoke`
* `--width <WIDTH>` — Output image width in pixels

  Default value: `64`
* `--height <HEIGHT>` — Output image height in pixels

  Default value: `64`
* `--steps <STEPS>` — Denoising steps

  Default value: `1`
* `--cfg-scale <CFG_SCALE>` — Classifier-free guidance scale

  Default value: `1`
* `--scheduler <SCHEDULER>` — Scheduler/sampler name

  Default value: `Euler`
* `--seed <SEED>` — Seed

  Default value: `0`
* `--denoising-strength <DENOISING_STRENGTH>` — Img2img denoising strength

  Default value: `0.5`
* `--txt2img-only` — Only run txt2img; skip the img2img leg
* `--skip-masked-img2img` — Skip the masked img2img leg



## `hipfire admin`

Query the running hipfire admin API for scripts and agents

**Usage:** `hipfire admin [OPTIONS] <COMMAND>`

###### **Subcommands:**

* `status` — Combined status snapshot for scripts and agents
* `chat` — Send one non-streaming chat request through /v1/chat/completions
* `health` — Raw /health payload
* `models` — Local model registry from the admin API
* `config` — Resolved runtime config
* `training` — Training run summaries or one run detail
* `diagnostics` — Filesystem, binary, kernel-cache, lock, and log diagnostics
* `logs` — Tail known hipfire logs
* `get` — GET an arbitrary admin/server path, e.g. /admin/training/runs
* `set-password` — Set the /admin console password (argon2id hash -> ~/.hipfire/admin.passwd)

###### **Options:**

* `--host <HOST>` — Override admin API host. Defaults to config host, with 0.0.0.0 mapped to 127.0.0.1
* `--port <PORT>` — Override admin API port. Defaults to config port



## `hipfire admin status`

Combined status snapshot for scripts and agents

**Usage:** `hipfire admin status`



## `hipfire admin chat`

Send one non-streaming chat request through /v1/chat/completions

**Usage:** `hipfire admin chat [OPTIONS] <PROMPT>...`

###### **Arguments:**

* `<PROMPT>` — User prompt text

###### **Options:**

* `--model <MODEL>` — Model tag/path. Defaults to server config when omitted
* `--system <SYSTEM>` — Optional system message
* `--max-tokens <MAX_TOKENS>` — Max tokens to generate
* `--temperature <TEMPERATURE>` — Sampling temperature
* `--top-p <TOP_P>` — Nucleus sampling top-p
* `--text` — Print only the assistant message text



## `hipfire admin health`

Raw /health payload

**Usage:** `hipfire admin health`



## `hipfire admin models`

Local model registry from the admin API

**Usage:** `hipfire admin models`



## `hipfire admin config`

Resolved runtime config

**Usage:** `hipfire admin config [OPTIONS]`

###### **Options:**

* `--model <MODEL>` — Resolve config for a specific model tag



## `hipfire admin training`

Training run summaries or one run detail

**Usage:** `hipfire admin training [OPTIONS] [ID]`

###### **Arguments:**

* `<ID>` — Optional run ID

###### **Options:**

* `--events` — Return full events for the run ID



## `hipfire admin diagnostics`

Filesystem, binary, kernel-cache, lock, and log diagnostics

**Usage:** `hipfire admin diagnostics`



## `hipfire admin logs`

Tail known hipfire logs

**Usage:** `hipfire admin logs [OPTIONS]`

###### **Options:**

* `--lines <LINES>` — Number of lines per log file

  Default value: `120`



## `hipfire admin get`

GET an arbitrary admin/server path, e.g. /admin/training/runs

**Usage:** `hipfire admin get <PATH>`

###### **Arguments:**

* `<PATH>` — Absolute or relative server path



## `hipfire admin set-password`

Set the /admin console password (argon2id hash -> ~/.hipfire/admin.passwd)

**Usage:** `hipfire admin set-password [PASSWORD]`

###### **Arguments:**

* `<PASSWORD>` — New password. If omitted, read once from stdin (no echo when a TTY)



<hr/>

<small><i>
    This document was generated automatically by
    <a href="https://crates.io/crates/clap-markdown"><code>clap-markdown</code></a>.
</i></small>
