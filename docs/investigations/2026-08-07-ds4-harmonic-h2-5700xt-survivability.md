<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev> -->

# DS4 harmonic H2: PCI-scoped 5700 XT survivability oracle

Status: PASS as hardware de-risking evidence. This does not certify the exact
gfx1100/gfx1151 production pair and therefore does not close H2.

Commit: `f98f768377327da1a24b3c492693be17cfe5158f`

Host: `hipx`

## Scope and identity

The user authorized only the otherwise-unused RX 5700 XT as a fault-injection
guinea pig. Redline's normalized `PciBusId` is the selector; no ROCr ordinal is
used.

| Field | Value |
|---|---|
| PCI BDF | `0000:6e:00.0` |
| PCI vendor/device | `1002:731f` |
| ROCr architecture | exactly `gfx1010` |
| DRM render node | `/dev/dri/renderD130` |
| Peer mappings | none |
| Product-GPU queues | zero |
| Loaded kernel/model | none |
| Held HSA allocation | 65,536 bytes, CPU pool |

The oracle refuses to initialize ROCr unless the BDF, sysfs vendor/device,
amdgpu driver binding, and exact ROCr architecture all agree. The parent also
removes visibility filters from child processes so ordinal remapping cannot
silently turn the requested physical identity into another GPU.

## Fault exercised

Each blocked worker creates one queue on the exact physical device, publishes
one AQL barrier waiting on a process-local signal that remains at one, and
reports its PCI identity and queue ID. The parent then sends SIGKILL. A fresh
process must reopen the same BDF, allocate a new queue, publish a zero-dependency
barrier, and observe its completion within two seconds.

This exercises kernel cleanup of a process that dies with a live queue and a
blocked packet without reproducing the quarantined reciprocal cross-device
wait or opening either production GPU.

## Results

Initial canary:

```text
PASS pci=0000:6e:00.0 name=gfx1010 queue=1 allocation_bytes=65536
{"status":"pass","pci_bus_id":"0000:6e:00.0","expected_name":"gfx1010","cycles":1,"blocked_packet":"process_local_barrier","peer_access":false,"product_gpu_queues":0}
```

Stress follow-up:

```text
10/10 fresh-process recovery dispatches passed
{"status":"pass","pci_bus_id":"0000:6e:00.0","expected_name":"gfx1010","cycles":10,"blocked_packet":"process_local_barrier","peer_access":false,"product_gpu_queues":0}
```

Postconditions after both invocations:

- `fuser -v /dev/dri/renderD130`: no owner;
- `pgrep -af pci_survivability`: no process;
- `journalctl -k -p warning --since=-5min`: no entries.

The device remained usable after eleven forced worker deaths in total.

## What this proves and does not prove

Proved:

1. Redline can select the intended physical GPU without ordinal ambiguity.
2. One-process/one-device AQL ownership survives a worker SIGKILL while its
   queue contains a blocked packet.
3. Recovery is a fresh process and allocation generation, not reuse of poisoned
   user-mode state.
4. The oracle itself has bounded readiness, dispatch, kill, and child-exit
   waits, and leaves no worker behind.

Not yet proved:

1. the exact gfx1100 dense worker and gfx1151 expert worker load receipts;
2. independent KFD/PASID ownership for those two workers;
3. gfx1151 expert service arithmetic or gfx1100 canonical-state execution;
4. unaffected-device survival when one member of the production pair is
   terminated; or
5. 2048/512 byte parity and throughput.

Those remain H2-H7 work. The old same-process reciprocal-wait product route
stays unconditionally quarantined.
