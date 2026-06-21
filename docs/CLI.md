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
* [`hipfire gpu-lock`↴](#hipfire-gpu-lock)
* [`hipfire gpu-lock acquire`↴](#hipfire-gpu-lock-acquire)
* [`hipfire gpu-lock release`↴](#hipfire-gpu-lock-release)
* [`hipfire gpu-lock status`↴](#hipfire-gpu-lock-status)
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
* `gpu-lock` — GPU mutex for multi-agent coordination (acquire/release/status)
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



## `hipfire gpu-lock`

GPU mutex for multi-agent coordination (acquire/release/status)

**Usage:** `hipfire gpu-lock <COMMAND>`

###### **Subcommands:**

* `acquire` — Acquire the GPU lock (blocks until free). A detached holder keeps it until `release` or the calling shell exits
* `release` — Release the GPU lock (SIGTERM the holder recorded in the lockfile)
* `status` — Print lock status: "gpu is free" or "gpu BUSY: <holder>"



## `hipfire gpu-lock acquire`

Acquire the GPU lock (blocks until free). A detached holder keeps it until `release` or the calling shell exits

**Usage:** `hipfire gpu-lock acquire [OPTIONS] <LABEL>`

###### **Arguments:**

* `<LABEL>` — Human label recorded in the lockfile (who/what holds it)

###### **Options:**

* `--watch-pid <WATCH_PID>` — Pid whose death auto-releases the lock (default: the calling shell)
* `--timeout-secs <TIMEOUT_SECS>` — Hard cap in seconds to wait for a busy lock; 0 = wait forever

  Default value: `1800`
* `--poll-secs <POLL_SECS>` — Cadence of "busy" messages while waiting, in seconds

  Default value: `5`



## `hipfire gpu-lock release`

Release the GPU lock (SIGTERM the holder recorded in the lockfile)

**Usage:** `hipfire gpu-lock release`



## `hipfire gpu-lock status`

Print lock status: "gpu is free" or "gpu BUSY: <holder>"

**Usage:** `hipfire gpu-lock status`



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
