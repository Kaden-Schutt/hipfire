# Containerized hipfire (podman)

A single multi-stage `Containerfile` (repo root) produces two images that share a
ROCm base:

| Target        | Purpose                              | Contains                                   |
|---------------|--------------------------------------|--------------------------------------------|
| `runtime`     | Deliverable inference image          | daemon + standalone `hipfire` CLI          |
| `gate-runner` | PR / dev-build GPU-gate validation   | full toolchain + source + gate scripts     |

Everything is podman-native but works unchanged with `docker` (swap the command).

## Why it's built this way

- **ROCm is dlopen'd at runtime** — the build needs no GPU. The base is
  `rocm/dev-ubuntu-24.04:7.2.4` because **gfx1151 requires ROCm 7.2+** (6.4.3
  segfaults on it). If a build reports `hipcc`/HIP headers missing, bump the base
  tag to `7.2.4-complete`.
- **Kernels JIT on first use.** Every `.hip` source and its helper headers
  (`turbo_common.h`, `*.cuh`, …) are embedded into the daemon binary via
  `include_str!` and stitched together in Rust before `hipcc` runs. The runtime
  image therefore needs only `hipcc` + HIP headers (from the base) — **not**
  `kernels/src/` on disk. First run is slower while kernels compile into the
  cache volume; subsequent runs are fast.
- **The CLI is compiled to a single binary** (`bun build --compile`), with
  `registry.json` inlined, so the runtime image carries no Bun runtime or JS.
- **Models are never baked in** — they are large runtime-only downloads mounted
  as a volume.
- **No `HSA_OVERRIDE_GFX_VERSION`** is baked in (project Rule 5); the arch is
  detected at runtime via HIP `gcnArchName`.

## Build

```bash
podman build -f Containerfile --target runtime     -t hipfire .
podman build -f Containerfile --target gate-runner -t hipfire-gate .
```

`.containerignore` prunes the ~160 GB working tree (target/, worktrees, venv,
models) down to a small source context.

## Run the deliverable

GPU passthrough is required. Rootless podman needs `--group-add keep-groups` to
keep the host `render`/`video` gids for `/dev/kfd` + `/dev/dri`.

```bash
podman run --rm -it \
  --device /dev/kfd --device /dev/dri \
  --group-add keep-groups --security-opt seccomp=unconfined \
  -v hipfire-models:/root/.hipfire/models \
  -v hipfire-kcache:/var/cache/hipfire \
  -p 11435:11435 \
  hipfire run qwen3.5:4b "2+2="
```

`hipfire serve qwen3.5:4b 0.0.0.0:11435 -d` exposes the JSON-lines daemon on the
published port. The `hipfire-kcache` volume persists JIT-compiled kernels across
container runs; `hipfire-models` persists pulled models.

## Validate a dev build / PR (GPU gates)

`scripts/container-gate.sh` builds `gate-runner` and runs a gate inside it with
GPU passthrough and the host models bind-mounted:

```bash
scripts/container-gate.sh                              # coherence battery (default)
scripts/container-gate.sh scripts/serve-multiturn-gate.sh
scripts/container-gate.sh scripts/coherence-gate-dflash.sh
```

Environment knobs:

- `HIPFIRE_CONTAINER=docker` — use docker instead of podman.
- `HIPFIRE_MODELS_DIR=<path>` — host models dir (default `~/.hipfire/models`).
- `HIPFIRE_SKIP_BUILD=1` — reuse the existing `hipfire-gate` tag.

The wrapper mounts the host GPU lock file (`/tmp/hipfire-gpu.lock`) when present
so a containerized gate coordinates with host GPU work.

## Publishing to a registry (not yet wired)

There is no CI workflow that publishes the image yet. When one is added, it
should build the `runtime` target (no GPU needed — ROCm is dlopen'd) on a
GitHub-hosted runner and push to a registry (Docker Hub or GHCR), e.g. `latest`
+ short SHA on `master` and a semver tag on release tags. The GPU gates cannot
run in cloud CI and stay local via `scripts/container-gate.sh`.

## Related

The older `docker/rocm7-builder.Dockerfile` is a separate, narrow helper for
pre-compiling gfx12 kernels via `compile-kernels.sh`; it is not part of this
build and is left as-is.
