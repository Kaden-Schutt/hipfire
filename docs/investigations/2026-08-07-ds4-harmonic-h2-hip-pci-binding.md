<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev> -->

# DS4 harmonic H2 stable HIP PCI binding

Date: 2026-08-07

Branch: `ds4-beta-staging`

Status: source and CPU checks only. No GPU was opened by this checkpoint.

## Problem

The independent Redline ROCr runtime provides a normalized physical
`PciBusId`, but hipfire's `Gpu::init_with_device` accepts a process-local HIP
ordinal. Ordinals can be reordered by visibility filters and are therefore not
a valid persistent owner identity on a multi-GPU host.

## Implementation

`HipRuntime` now loads the public ROCm APIs `hipDeviceGetPCIBusId` and
`hipDeviceGetByPCIBusId`, declared by the installed ROCm 7.14
`hip_runtime_api.h`. It exposes:

- `device_by_pci_bus_id`, which resolves a current process-local ordinal from
  the persistent BDF; and
- `device_pci_bus_id`, which reads the normalized BDF back from an ordinal.

`Gpu::init_with_pci_bus_id` admits a device only when all three identities
agree before any model allocation:

1. HIP resolves the requested BDF;
2. the resolved ordinal round-trips to the same BDF; and
3. the physical architecture and hipfire compilation target both equal the
   caller's exact expected architecture.

This last check rejects `HIPFIRE_TARGET_ARCH` spoofing on the harmonic path.
Ordinary single-device `Gpu::init` and `Gpu::init_with_device` behavior is
unchanged.

`DeepseekV4RoutedWeights::audit_local_owner` includes the stable HIP BDF in its
residency receipt while retaining the ordinal only as diagnostics.

The worker supervisor will additionally open Redline's ROCr agent with
`GpuSelector::PciBusId` and require that its normalized identity equals the HIP
receipt. Failure to resolve under the process visibility environment is a hard
admission failure; it never falls back to ordinal zero.

## Checks

- `cargo check -p hip-bridge -p rdna-compute -p hipfire-arch-deepseek4`
- `cargo test -p hip-bridge --lib` (12 passed)
- `cargo test -p rdna-compute --lib` (125 passed)
- `scripts/fmt-changed.sh` against `origin/ds4-beta-staging`
- `git diff --check`

The exact gfx1100/gfx1151 BDF round-trip and same-device HIP/ROCr proof remain
unrun. They are part of the still-open H2 hardware exit and require the
fault-contained worker process, not an ad hoc product smoke.
