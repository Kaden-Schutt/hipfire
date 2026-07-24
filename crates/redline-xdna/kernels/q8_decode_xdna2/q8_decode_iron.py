# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Copyright (c) 2026 Kaden Schutt

import argparse

import numpy as np
from ml_dtypes import bfloat16

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2


def build_q8_decoder(k: int):
    if k <= 0 or k % 32 != 0:
        raise ValueError(f"K must be a positive multiple of 32, got {k}")

    block_count = k // 32
    packed_type = np.ndarray[(block_count * 34,), np.dtype[np.uint8]]
    decoded_type = np.ndarray[(k,), np.dtype[bfloat16]]
    scratch_type = np.ndarray[(64,), np.dtype[np.uint8]]

    packed_fifo = ObjectFifo(packed_type, name="packed_q8")
    decoded_fifo = ObjectFifo(decoded_type, name="decoded_bf16")
    decoder = Kernel(
        "q8_decode_bf16",
        "q8_decode.cc.o",
        [packed_type, decoded_type, np.int32],
    )

    def decode_body(packed_fifo, decoded_fifo, decoder):
        packed = packed_fifo.acquire(1)
        decoded = decoded_fifo.acquire(1)
        decoder(packed, decoded, block_count)
        packed_fifo.release(1)
        decoded_fifo.release(1)

    worker = Worker(
        decode_body,
        [packed_fifo.cons(), decoded_fifo.prod(), decoder],
    )

    runtime = Runtime()
    with runtime.sequence(
        packed_type, decoded_type, scratch_type
    ) as (packed, decoded, _scratch):
        runtime.start(worker)
        runtime.fill(packed_fifo.prod(), packed)
        runtime.drain(decoded_fifo.cons(), decoded, wait=True)

    return Program(NPU2(), runtime).resolve_program()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-K", type=int, default=2048)
    args = parser.parse_args()
    print(build_q8_decoder(args.K))


if __name__ == "__main__":
    main()
