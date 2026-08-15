# hipfire container — multi-stage build (podman-native; docker-compatible).
#
# Two final targets share a base:
#   base-rocm ──► builder ──► gate-runner   (PR / dev-build validation)
#        └──────────────────► runtime        (deliverable inference image)
#
# Design notes:
#   * ROCm/HIP is dlopen'd at runtime; the build itself needs no GPU.
#   * gfx1151 requires ROCm 7.2+ (6.4.3 segfaults on it) — do NOT downgrade.
#   * Kernels are JIT-compiled on first use: every .hip source AND its helper
#     headers (turbo_common.h, *.cuh, ...) are embedded into the daemon binary
#     via include_str! and stitched in Rust before hipcc runs. So neither
#     kernels/src/ nor registry.json need to exist on disk in the runtime image;
#     the image only needs `hipcc` + HIP headers (from the ROCm base) at runtime.
#   * Models are runtime-only downloads — never baked in; mount as a volume.
#
# Build:
#   podman build -f Containerfile --target runtime     -t hipfire .
#   podman build -f Containerfile --target gate-runner  -t hipfire-gate .
#
# Run (deliverable, needs GPU passthrough):
#   podman run --rm -it \
#     --device /dev/kfd --device /dev/dri \
#     --group-add keep-groups --security-opt seccomp=unconfined \
#     -v hipfire-models:/root/.hipfire/models \
#     -v hipfire-kcache:/var/cache/hipfire \
#     -p 11435:11435 hipfire run qwen3.5:4b "2+2="

# ─────────────────────────────────────────────────────────────────────────────
# base-rocm — ROCm SDK (hipcc + clang + HIP headers + runtime .so set).
# The non-"-complete" dev image (~1.2 GB) ships hipcc and the HIP headers, which
# is all the JIT path needs for gfx1151. If a build ever reports hipcc/headers
# missing, switch the tag to `7.2.4-complete` (adds rocBLAS/MIOpen, ~7.4 GB).
# ─────────────────────────────────────────────────────────────────────────────
FROM docker.io/rocm/dev-ubuntu-24.04:7.2.4 AS base-rocm

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        unzip \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

# HIP install root (compiler.rs defaults HIP_PATH to /opt/rocm) and a writable
# JIT cache dir (the default is CWD-relative .hipfire_kernels/ — unusable in a
# read-only workdir, so pin it to a volume-friendly path).
ENV HIP_PATH=/opt/rocm \
    HIPFIRE_KERNEL_CACHE=/var/cache/hipfire
RUN mkdir -p /var/cache/hipfire && chmod 1777 /var/cache/hipfire

# ─────────────────────────────────────────────────────────────────────────────
# builder — adds Rust, compiles the daemon and the standalone CLI binary.
# ─────────────────────────────────────────────────────────────────────────────
FROM base-rocm AS builder

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain stable --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /hipfire
COPY . .

# Daemon is a standalone binary crate (hipfire-daemon) that bundles every
# architecture unconditionally — no --features or --example needed.
RUN cargo build --release --locked -p hipfire-daemon
RUN cargo build --release --locked -p hipfire-cli

# ─────────────────────────────────────────────────────────────────────────────
# runtime — TARGET A, deliverable. No Rust, no source tree, no gate scripts.
# ─────────────────────────────────────────────────────────────────────────────
FROM base-rocm AS runtime

COPY --from=builder /hipfire/target/release/daemon /opt/hipfire/bin/daemon
COPY --from=builder /hipfire/target/release/hipfire /usr/local/bin/hipfire

ENV HIPFIRE_DAEMON_BIN=/opt/hipfire/bin/daemon \
    HIPFIRE_DIR=/root/.hipfire

# Models and the JIT kernel cache persist across runs via named volumes.
VOLUME ["/root/.hipfire/models", "/var/cache/hipfire"]
EXPOSE 11435

ENTRYPOINT ["hipfire"]
CMD ["--help"]

# ─────────────────────────────────────────────────────────────────────────────
# gate-runner — TARGET B, PR/dev-build validation. Full toolchain + source +
# in-repo GPU gate scripts. Runs with GPU passthrough against mounted models.
# ─────────────────────────────────────────────────────────────────────────────
FROM builder AS gate-runner

# pytest/numpy for the Python-backed detector arms in the gate scripts.
RUN pip install --no-cache-dir --break-system-packages pytest numpy

COPY --from=builder /hipfire/target/release/hipfire /usr/local/bin/hipfire

ENV HIPFIRE_DAEMON_BIN=/hipfire/target/release/daemon \
    HIPFIRE_DIR=/root/.hipfire \
    MODELS_DIR=/root/.hipfire/models

WORKDIR /hipfire
# No default gate: there is no universal correctness battery. The old default
# (scripts/coherence-gate.sh) is retired AND absent from the checkout, so it
# execed as "no such file" (127). Fail closed with guidance instead; pick a
# route from docs/VALIDATION.md and pass it explicitly, e.g.
#   podman run ... hipfire-gate scripts/serve-multiturn-gate.sh
ENTRYPOINT ["/bin/bash"]
CMD ["-c", "echo '[hipfire-gate] no gate command given.' >&2; echo 'There is no universal gate. Select a validation route from docs/VALIDATION.md and pass it explicitly, e.g.:' >&2; echo '  scripts/serve_harness.py battery --model <model>' >&2; echo '  scripts/redline_daemon_harness.py   # kernel/dispatch/graph/Redline changes' >&2; exit 2"]
