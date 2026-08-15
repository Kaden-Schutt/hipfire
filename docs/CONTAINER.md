# Containerized hipfire

Repo-root multi-stage `Containerfile` (podman-native, Docker-compatible).
Two final targets share a ROCm base:

| Target | Purpose | Image contents |
|---|---|---|
| `runtime` | Deliverable inference image | Wrapped `daemon` + compiled standalone `hipfire` CLI |
| `gate-runner` | Local GPU harness / historical gate runner | Full builder tree + source + in-image entry scripts |

| Field | Value |
|---|---|
| Base image | `docker.io/rocm/dev-ubuntu-24.04:7.2.4` |
| Control plane | Native `hipfire-cli` Rust binary |
| Published port | `11435` (`EXPOSE`) |
| JIT kernel cache env | `HIPFIRE_KERNEL_CACHE=/var/cache/hipfire` |
| Daemon path in runtime | `HIPFIRE_DAEMON_BIN=/opt/hipfire/bin/daemon` |

Truth state: **shipped / ref-pinned** for build/run paths below. There is
**no** GitHub Actions workflow that builds or publishes these images today
(`.github/workflows/` has no container publish job). No-GPU CI does **not**
exercise the image or the GPU.

## Design constraints (from `Containerfile`)

- **ROCm is dlopen'd at runtime** — the image build needs no GPU.
- **gfx1151 needs ROCm 7.2+** — do not downgrade the base; if `hipcc`/HIP
  headers are missing, switch the tag to `7.2.4-complete`.
- **Kernels JIT on first use.** `.hip` sources and helpers are embedded in
  the daemon via `include_str!`; the runtime image needs `hipcc` + HIP
  headers from the base, not `kernels/src/` on disk.
- **CLI is native Rust.** `hipfire-registry` embeds `registry/v1.json`; neither
  the builder nor runtime image installs Bun, Node, or a TypeScript payload.
- **Models are never baked in** — mount a volume.
- **No `HSA_OVERRIDE_GFX_VERSION`** is set in the image; arch comes from HIP
  `gcnArchName` at runtime.

`.dockerignore` keeps the build context small (excludes `target/`, models,
worktrees, etc.).

## Build

```bash
podman build -f Containerfile --target runtime     -t hipfire .
podman build -f Containerfile --target gate-runner -t hipfire-gate .
```

Docker works the same with `docker build …`. Rootful Docker does **not**
implement Podman's `--group-add keep-groups` token — omit that flag under
Docker (see run section).

Daemon build inside the image matches the project default:

```text
cargo build --release --locked --features deltanet --example daemon -p hipfire-runtime
```

## Run the runtime image (GPU required)

GPU device nodes are required for inference. Rootless podman needs
`--group-add keep-groups` so host `render`/`video` gids reach `/dev/kfd` and
`/dev/dri`.

**Unauthenticated HTTP.** Serve has no auth and no TLS ([SERVE.md](SERVE.md)).
Publish the port only on loopback unless a trusted network or authenticated
TLS reverse proxy protects it. Prefer `-p 127.0.0.1:11435:11435` over bare
`-p 11435:11435` (the latter binds all host interfaces).

```bash
podman run --rm -it \
  --device /dev/kfd --device /dev/dri \
  --group-add keep-groups --security-opt seccomp=unconfined \
  -v hipfire-models:/root/.hipfire/models \
  -v hipfire-kcache:/var/cache/hipfire \
  -p 127.0.0.1:11435:11435 \
  hipfire run qwen3.5:4b "2+2="
```

Serve in the **foreground** (ENTRYPOINT is `hipfire`). Keep **host** publication
on loopback (`-p 127.0.0.1:11435:11435`) but bind Hipfire to **`0.0.0.0:11435`
inside the container** so published-port DNAT can reach the listener
(a container-local `127.0.0.1` bind is unreachable through Podman/Docker port publish):

```bash
podman run --rm -it \
  --device /dev/kfd --device /dev/dri \
  --group-add keep-groups --security-opt seccomp=unconfined \
  -v hipfire-models:/root/.hipfire/models \
  -v hipfire-kcache:/var/cache/hipfire \
  -p 127.0.0.1:11435:11435 \
  hipfire serve qwen3.5:4b 0.0.0.0:11435
```

To publish on all host interfaces (no auth, no TLS — trusted network or
authenticated TLS reverse proxy only): `-p 11435:11435` and keep the in-container
bind at `0.0.0.0:11435`.

Do **not** pass `-d` / `--detach` as the container command: that forks a
background child and lets PID 1 exit, which stops the container. Detach with
the container runtime (`podman run -d … hipfire serve …`) if you need a
long-lived container.

Volumes:

| Volume / path | Role |
|---|---|
| `hipfire-models` → `/root/.hipfire/models` | Pulled / mounted model files |
| `hipfire-kcache` → `/var/cache/hipfire` | Persistent JIT kernel cache |

OpenAI HTTP surface after serve is up: [SERVE.md](SERVE.md).

## Local containerized GPU runs (`scripts/container-gate.sh`)

Helper builds `gate-runner` and runs a command inside it with GPU
passthrough and the host models directory bind-mounted.

`Containerfile` gate-runner `CMD` and the wrapper’s no-arg default both name
the legacy base coherence-gate entry (scripts/coherence-gate.sh) — historical/pre-modular/generated and intentionally absent from the checkout (only `scripts/coherence-gate-*.sh` variants remain). Pass an **explicit**
existing command:

```bash
# Preferred current serve smoke (gate ENTRYPOINT is bash — invoke Python explicitly):
scripts/container-gate.sh -lc 'exec python3 scripts/serve_harness.py --model /root/.hipfire/models/<file> --tag qwen3.5:9b'

# Historical reproduction only (retired acceptance — see VALIDATION.md):
scripts/container-gate.sh scripts/coherence-gate-dflash.sh
scripts/container-gate.sh scripts/serve-multiturn-gate.sh
```

Environment:

| Variable | Default | Role |
|---|---|---|
| `HIPFIRE_CONTAINER` | `podman` | `podman` or `docker` |
| `HIPFIRE_MODELS_DIR` | `~/.hipfire/models` | Host models path |
| `HIPFIRE_IMAGE` | `hipfire-gate` | Image tag |
| `HIPFIRE_SKIP_BUILD` | unset | `1` reuses existing tag |
| `HIPFIRE_GPU_LOCKFILE` | `/tmp/hipfire-gpu.lock` | Mounted when present |

**Scope limits (fail closed):**

- `scripts/coherence-gate-*.sh` batteries are **retired as acceptance
  evidence** ([VALIDATION.md](VALIDATION.md)). Historical reproduction only —
  never merge/promotion criteria.
- A successful **image build** proves the multi-stage compile path only. It
  does **not** certify GPU runtime correctness, serve semantics, or perf.
- Bare `scripts/container-gate.sh` with no args will try the missing default
  legacy base coherence-gate entry (scripts/coherence-gate.sh; historical/pre-modular/generated and intentionally absent from the checkout) and fail closed until you pass a real command.
- No-GPU CI (`.github/workflows/no-gpu-ci.yml`) does **not** build this
  Containerfile and does **not** replace a manual GPU route.

For current user-facing serve smoke, prefer `scripts/serve_harness.py` (host
or via `container-gate.sh`) per [VALIDATION.md](VALIDATION.md) and
[SERVE.md](SERVE.md).

## Publishing

Not wired. When added, the natural path is building the `runtime` target
(no GPU required at build time) and pushing tags from CI; GPU validation
remains a local/manual concern.

## Related paths

| Path | Role |
|---|---|
| `Containerfile` | Multi-stage definition |
| `scripts/container-gate.sh` | Local GPU container runner |
| `docker/rocm7-builder.Dockerfile` | Separate helper for pre-compiling gfx12 kernels via `compile-kernels.sh` — **not** part of the runtime/gate-runner deliverable |
| [SERVE.md](SERVE.md) | HTTP API |
| [NIXOS.md](NIXOS.md) | NixOS alternative install |
