# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Copyright (c) 2026 Kaden Schutt

import argparse

import numpy as np
from ml_dtypes import bfloat16

from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2


ROWS_PER_CHUNK = 8
K = 64
OUTPUTS = 16
Q8_BLOCK_ELEMENTS = 32
Q8_BLOCK_BYTES = 34


def build_persistent_microtile(chunks: int):
    if chunks not in (2, 8, 16):
        raise ValueError(f"chunks must be one of 2, 8, 16, got {chunks}")

    activation_type = np.ndarray[
        (chunks * ROWS_PER_CHUNK * K,), np.dtype[bfloat16]
    ]
    packed_type = np.ndarray[
        (OUTPUTS * (K // Q8_BLOCK_ELEMENTS) * Q8_BLOCK_BYTES,),
        np.dtype[np.uint8],
    ]
    decoded_type = np.ndarray[(K * OUTPUTS,), np.dtype[bfloat16]]
    output_type = np.ndarray[
        (chunks * ROWS_PER_CHUNK * OUTPUTS,), np.dtype[np.float32]
    ]

    # This proof holds one whole multi-chunk activation object while B is
    # resident. Depth 1 is intentional: default ping-pong would duplicate the
    # 16 KiB A and 8 KiB C objects at CHUNKS=16 and exhaust compute-tile SRAM.
    activation_fifo = ObjectFifo(activation_type, name="activation", depth=1)
    packed_fifo = ObjectFifo(packed_type, name="packed_q8", depth=1)
    output_fifo = ObjectFifo(output_type, name="output", depth=1)
    decoded_buffer = Buffer(decoded_type, name="decoded_weight")
    decode_kernel = Kernel(
        "q8_decode_b_tile_major",
        "q8_persistent_microtile.cc.o",
        [packed_type, decoded_type],
    )
    matmul_kernel = Kernel(
        "bf16_persistent_microtile",
        "q8_persistent_microtile.cc.o",
        [activation_type, decoded_type, output_type],
    )

    def worker_body(
        activation_fifo,
        packed_fifo,
        output_fifo,
        decoded_buffer,
        decode_kernel,
        matmul_kernel,
    ):
        activation = activation_fifo.acquire(1)
        packed = packed_fifo.acquire(1)
        output = output_fifo.acquire(1)
        decode_kernel(packed, decoded_buffer)
        matmul_kernel(activation, decoded_buffer, output)
        activation_fifo.release(1)
        packed_fifo.release(1)
        output_fifo.release(1)

    worker = Worker(
        worker_body,
        [
            activation_fifo.cons(),
            packed_fifo.cons(),
            output_fifo.prod(),
            decoded_buffer,
            decode_kernel,
            matmul_kernel,
        ],
    )

    runtime = Runtime()
    with runtime.sequence(
        activation_type, packed_type, output_type
    ) as (activation, packed, output):
        runtime.start(worker)
        runtime.fill(activation_fifo.prod(), activation)
        runtime.fill(packed_fifo.prod(), packed)
        runtime.drain(output_fifo.cons(), output, wait=True)

    return Program(NPU2(), runtime).resolve_program()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=int, choices=[2, 8, 16], required=True)
    args = parser.parse_args()
    print(build_persistent_microtile(args.chunks))


if __name__ == "__main__":
    main()
