#!/usr/bin/env python3
"""Exercise the ROCm kernels and RCCL transport required by FastMTP training."""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.distributed as dist


def validate_devices(expected_devices: int, arch_prefix: str) -> None:
    if not torch.version.hip:
        raise RuntimeError("installed torch is not a ROCm build")
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm torch cannot see a GPU")
    count = torch.cuda.device_count()
    if count != expected_devices:
        raise RuntimeError(f"expected {expected_devices} GPUs, found {count}")

    devices = []
    for index in range(count):
        props = torch.cuda.get_device_properties(index)
        arch = str(getattr(props, "gcnArchName", "unknown"))
        if not arch.startswith(arch_prefix):
            raise RuntimeError(f"GPU {index} has unexpected architecture {arch}")
        with torch.cuda.device(index):
            lhs = torch.arange(
                64 * 64, device=f"cuda:{index}", dtype=torch.bfloat16
            ).reshape(64, 64)
            result = lhs @ lhs.T
            torch.cuda.synchronize(index)
            if not bool(torch.isfinite(result).all()) or float(result.abs().sum()) == 0:
                raise RuntimeError(f"GPU {index} failed the BF16 matmul smoke")
        devices.append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "arch": arch,
                "bf16_matmul": "pass",
            }
        )
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "hip": torch.version.hip,
                "devices": devices,
            }
        ),
        flush=True,
    )


def validate_distributed(expected_devices: int) -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != expected_devices:
        raise RuntimeError(f"expected {expected_devices} ranks, found {world}")

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    try:
        value = torch.tensor([rank + 1.0], device=f"cuda:{local_rank}")
        dist.all_reduce(value)
        expected = world * (world + 1) / 2
        if float(value.item()) != expected:
            raise RuntimeError(
                f"rank {rank} RCCL all-reduce returned {value.item()}, expected {expected}"
            )
        dist.barrier()
        if rank == 0:
            print(
                json.dumps(
                    {
                        "backend": dist.get_backend(),
                        "ranks": world,
                        "all_reduce": "pass",
                    }
                ),
                flush=True,
            )
    finally:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("devices", "distributed"), default="devices"
    )
    parser.add_argument("--expected-devices", type=int, default=4)
    parser.add_argument("--arch-prefix", default="gfx1201")
    args = parser.parse_args()

    if args.mode == "distributed":
        validate_distributed(args.expected_devices)
    else:
        validate_devices(args.expected_devices, args.arch_prefix)


if __name__ == "__main__":
    main()
