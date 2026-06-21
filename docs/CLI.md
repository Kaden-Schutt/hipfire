# Command-Line Help for `hipfire`

This document contains the help content for the `hipfire` command-line program.

**Command Overview:**

* [`hipfire`↴](#hipfire)
* [`hipfire serve`↴](#hipfire-serve)
* [`hipfire run`↴](#hipfire-run)
* [`hipfire list`↴](#hipfire-list)
* [`hipfire eval`↴](#hipfire-eval)
* [`hipfire host-profile`↴](#hipfire-host-profile)
* [`hipfire collect-artifacts`↴](#hipfire-collect-artifacts)
* [`hipfire gpu-lock`↴](#hipfire-gpu-lock)
* [`hipfire gpu-lock acquire`↴](#hipfire-gpu-lock-acquire)
* [`hipfire gpu-lock release`↴](#hipfire-gpu-lock-release)
* [`hipfire gpu-lock status`↴](#hipfire-gpu-lock-status)
* [`hipfire operator`↴](#hipfire-operator)
* [`hipfire operator status`↴](#hipfire-operator-status)
* [`hipfire operator health`↴](#hipfire-operator-health)
* [`hipfire operator models`↴](#hipfire-operator-models)
* [`hipfire operator config`↴](#hipfire-operator-config)
* [`hipfire operator training`↴](#hipfire-operator-training)
* [`hipfire operator diagnostics`↴](#hipfire-operator-diagnostics)
* [`hipfire operator logs`↴](#hipfire-operator-logs)
* [`hipfire operator get`↴](#hipfire-operator-get)

## `hipfire`

hipfire LLM inference CLI

**Usage:** `hipfire <COMMAND>`

###### **Subcommands:**

* `serve` — Start the hipfire HTTP server (OpenAI-compatible)
* `run` — Load a model and generate a response (one-shot)
* `list` — List locally available models
* `eval` — Run the quant admission/model evaluation harness
* `host-profile` — Measure host, GPU-copy, and model storage bandwidth
* `collect-artifacts` — Collect Tier-1 calibration artifacts (Hessian/imatrix/router-histogram) in one model load
* `gpu-lock` — GPU mutex for multi-agent coordination (acquire/release/status)
* `operator` — Query the running hipfire operator API for scripts and agents



## `hipfire serve`

Start the hipfire HTTP server (OpenAI-compatible)

**Usage:** `hipfire serve [OPTIONS]`

###### **Options:**

* `--host <HOST>` — Override bind host
* `-p`, `--port <PORT>` — Override bind port
* `-m`, `--model <MODEL>` — Pre-load a model on startup



## `hipfire run`

Load a model and generate a response (one-shot)

**Usage:** `hipfire run [OPTIONS] <MODEL> <PROMPT>`

###### **Arguments:**

* `<MODEL>` — Model name, alias, or path
* `<PROMPT>` — Prompt text

###### **Options:**

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



## `hipfire operator`

Query the running hipfire operator API for scripts and agents

**Usage:** `hipfire operator [OPTIONS] <COMMAND>`

###### **Subcommands:**

* `status` — Combined status snapshot for scripts and agents
* `health` — Raw /health payload
* `models` — Local model registry from the operator API
* `config` — Resolved runtime config
* `training` — Training run summaries or one run detail
* `diagnostics` — Filesystem, binary, kernel-cache, lock, and log diagnostics
* `logs` — Tail known hipfire logs
* `get` — GET an arbitrary operator/server path, e.g. /operator/training/runs

###### **Options:**

* `--host <HOST>` — Override operator API host. Defaults to config host, with 0.0.0.0 mapped to 127.0.0.1
* `--port <PORT>` — Override operator API port. Defaults to config port



## `hipfire operator status`

Combined status snapshot for scripts and agents

**Usage:** `hipfire operator status`



## `hipfire operator health`

Raw /health payload

**Usage:** `hipfire operator health`



## `hipfire operator models`

Local model registry from the operator API

**Usage:** `hipfire operator models`



## `hipfire operator config`

Resolved runtime config

**Usage:** `hipfire operator config [OPTIONS]`

###### **Options:**

* `--model <MODEL>` — Resolve config for a specific model tag



## `hipfire operator training`

Training run summaries or one run detail

**Usage:** `hipfire operator training [OPTIONS] [ID]`

###### **Arguments:**

* `<ID>` — Optional run ID

###### **Options:**

* `--events` — Return full events for the run ID



## `hipfire operator diagnostics`

Filesystem, binary, kernel-cache, lock, and log diagnostics

**Usage:** `hipfire operator diagnostics`



## `hipfire operator logs`

Tail known hipfire logs

**Usage:** `hipfire operator logs [OPTIONS]`

###### **Options:**

* `--lines <LINES>` — Number of lines per log file

  Default value: `120`



## `hipfire operator get`

GET an arbitrary operator/server path, e.g. /operator/training/runs

**Usage:** `hipfire operator get <PATH>`

###### **Arguments:**

* `<PATH>` — Absolute or relative server path



<hr/>

<small><i>
    This document was generated automatically by
    <a href="https://crates.io/crates/clap-markdown"><code>clap-markdown</code></a>.
</i></small>
