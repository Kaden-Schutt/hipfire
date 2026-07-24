# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Copyright (c) 2026 Kaden Schutt

import numpy as np
from ml_dtypes import bfloat16

from aie.helpers.taplib import TensorTiler2D
from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2, Tile


M = 256
K = 2048
N = 2048
TILE_M = 64
TILE_K = 64
TILE_N = 64
AIE_ROWS = 4
AIE_COLS = 8
MMUL_M = 8
MMUL_K = 8
MMUL_N = 8
Q8_BLOCK_ELEMENTS = 32
Q8_BLOCK_BYTES = 34
Q8_VALUES = 256
LUT_COPIES_PER_PAIR = 2
PANEL_K = 512
PACKED_ROW_BYTES = (K // Q8_BLOCK_ELEMENTS) * Q8_BLOCK_BYTES
PACKED_PANEL_ROW_BYTES = (
    PANEL_K // Q8_BLOCK_ELEMENTS
) * Q8_BLOCK_BYTES
PACKED_TILE_BYTES = (TILE_K // Q8_BLOCK_ELEMENTS) * Q8_BLOCK_BYTES


def build_full_array():
    a_type = np.ndarray[(M * K,), np.dtype[bfloat16]]
    b_type = np.ndarray[(N * PACKED_ROW_BYTES,), np.dtype[np.uint8]]
    c_type = np.ndarray[(M * N,), np.dtype[np.float32]]

    # Retain K=512 panels in each memory tile. This is the largest 64-row BF16
    # panel that fits one XDNA2 64 KiB memory bank.
    a_l2_type = np.ndarray[(TILE_M * PANEL_K,), np.dtype[bfloat16]]
    a_l1_type = np.ndarray[(TILE_M, TILE_K), np.dtype[bfloat16]]
    # Native Q8_0 is row-packed W[N,K]. Retaining K=512 for 64 outputs cuts
    # external 68-byte gathers eightfold while keeping every buffer bank-local.
    b_l2_type = np.ndarray[
        (TILE_N * PACKED_PANEL_ROW_BYTES,), np.dtype[np.uint8]
    ]
    b_l1_type = np.ndarray[
        (TILE_N * PACKED_TILE_BYTES,), np.dtype[np.uint8]
    ]
    b_decoded_type = np.ndarray[(TILE_K * TILE_N,), np.dtype[bfloat16]]
    q8_lut_type = np.ndarray[
        (Q8_VALUES * LUT_COPIES_PER_PAIR,), np.dtype[bfloat16]
    ]
    c_l2_type = np.ndarray[
        (TILE_M * TILE_N * AIE_ROWS,), np.dtype[np.float32]
    ]
    c_l1_type = np.ndarray[(TILE_M, TILE_N), np.dtype[np.float32]]

    zero_kernel = Kernel(
        "zero_f32_64x64", "q8_full_array.cc.o", [c_l1_type]
    )
    decode_kernel = Kernel(
        "q8_decode_64x64_tile",
        "q8_full_array.cc.o",
        [b_l1_type, b_decoded_type, q8_lut_type, q8_lut_type],
    )
    init_lut_kernel = Kernel(
        "init_q8_bf16_lut",
        "q8_full_array.cc.o",
        [q8_lut_type, q8_lut_type],
    )
    matmul_kernel = Kernel(
        "matmul_bf16_f32_64x64",
        "q8_full_array.cc.o",
        [a_l1_type, b_decoded_type, c_l1_type],
    )

    a_l3l2 = [None] * AIE_ROWS
    a_l2l1 = [None] * AIE_ROWS
    for row in range(AIE_ROWS):
        a_l3l2[row] = ObjectFifo(
            a_l2_type, name=f"A_L3L2_{row}", depth=2
        )
        a_l2l1[row] = (
            a_l3l2[row]
            .cons()
            .forward(
                obj_type=a_l1_type,
                name=f"A_L2L1_{row}",
                depth=2,
                dims_to_stream=[
                    (PANEL_K // MMUL_K, MMUL_K),
                    (TILE_M // MMUL_M, MMUL_M * PANEL_K),
                    (MMUL_M, PANEL_K),
                    (MMUL_K, 1),
                ],
                tile=Tile(2 * row, 1),
            )
        )

    b_l3l2 = [None] * AIE_COLS
    b_l2l1 = [None] * AIE_COLS
    c_l1l2 = [[None] * AIE_COLS for _ in range(AIE_ROWS)]
    c_l2l3 = [None] * AIE_COLS
    for col in range(AIE_COLS):
        b_l3l2[col] = ObjectFifo(
            b_l2_type, name=f"B_L3L2_{col}", depth=2
        )
        b_l2l1[col] = (
            b_l3l2[col]
            .cons()
            .forward(
                obj_type=b_l1_type,
                name=f"B_L2L1_{col}",
                depth=1,
                dims_to_stream=[
                    (PANEL_K // TILE_K, PACKED_TILE_BYTES),
                    (TILE_N, PACKED_PANEL_ROW_BYTES),
                    (PACKED_TILE_BYTES, 1),
                ],
                tile=Tile(col, 1),
            )
        )

        c_l2l3[col] = ObjectFifo(
            c_l2_type,
            name=f"C_L2L3_{col}",
            depth=1,
            dims_to_stream=[
                (TILE_M // MMUL_M, MMUL_M * TILE_N),
                (MMUL_M, MMUL_N),
                (TILE_N // MMUL_N, MMUL_M * MMUL_N),
                (MMUL_N, 1),
            ],
        )
        c_children = (
            c_l2l3[col]
            .prod()
            .join(
                [TILE_M * TILE_N * row for row in range(AIE_ROWS)],
                obj_types=[c_l1_type] * AIE_ROWS,
                names=[f"C_L1L2_{col}_{row}" for row in range(AIE_ROWS)],
                depths=[1] * AIE_ROWS,
                tile=Tile(col, 1),
            )
        )
        for row in range(AIE_ROWS):
            c_l1l2[row][col] = c_children[row]

    def core_body(
        a_fifo,
        b_fifo,
        c_fifo,
        decoded,
        lut_ab,
        lut_cd,
        zero,
        init_lut,
        decode,
        matmul,
    ):
        init_lut(lut_ab, lut_cd)
        for _ in range_(4):
            c_tile = c_fifo.acquire(1)
            zero(c_tile)
            for _ in range_(K // TILE_K):
                a_tile = a_fifo.acquire(1)
                b_tile = b_fifo.acquire(1)
                decode(b_tile, decoded, lut_ab, lut_cd)
                matmul(a_tile, decoded, c_tile)
                a_fifo.release(1)
                b_fifo.release(1)
            c_fifo.release(1)

    workers = []
    for row in range(AIE_ROWS):
        for col in range(AIE_COLS):
            decoded = Buffer(
                b_decoded_type, name=f"decoded_B_{row}_{col}"
            )
            lut_ab = Buffer(q8_lut_type, name=f"q8_lut_ab_{row}_{col}")
            lut_cd = Buffer(q8_lut_type, name=f"q8_lut_cd_{row}_{col}")
            workers.append(
                Worker(
                    core_body,
                    [
                        a_l2l1[row].cons(),
                        b_l2l1[col].cons(),
                        c_l1l2[row][col].prod(),
                        decoded,
                        lut_ab,
                        lut_cd,
                        zero_kernel,
                        init_lut_kernel,
                        decode_kernel,
                        matmul_kernel,
                    ],
                    tile=Tile(col, row + 2),
                    stack_size=0xD00,
                )
            )

    a_tiles = TensorTiler2D.group_tiler(
        (M, K),
        (TILE_M, PANEL_K),
        (1, K // PANEL_K),
        pattern_repeat=N // TILE_N // AIE_COLS,
        prune_step=False,
    )
    b_tiles = TensorTiler2D.step_tiler(
        (N, PACKED_ROW_BYTES),
        (TILE_N, PACKED_PANEL_ROW_BYTES),
        tile_group_repeats=(
            N // TILE_N // AIE_COLS,
            K // PANEL_K,
        ),
        tile_group_steps=(AIE_COLS, 1),
        prune_step=False,
    )
    c_tiles = TensorTiler2D.step_tiler(
        (M, N),
        (TILE_M * AIE_ROWS, TILE_N),
        tile_group_repeats=(1, N // TILE_N // AIE_COLS),
        tile_group_steps=(1, AIE_COLS),
        prune_step=False,
    )
    if not (
        len(a_tiles) == AIE_ROWS
        and len(b_tiles) == AIE_COLS
        and len(c_tiles) == AIE_COLS
    ):
        raise ValueError(
            "unexpected full-array tap count: "
            f"A={len(a_tiles)} B={len(b_tiles)} C={len(c_tiles)}"
        )

    runtime = Runtime()
    with runtime.sequence(a_type, b_type, c_type) as (a, b, c):
        runtime.start(*workers)
        group = runtime.task_group()
        for col in range(AIE_COLS):
            runtime.drain(
                c_l2l3[col].cons(),
                c,
                tap=c_tiles[col],
                wait=True,
                task_group=group,
                tile=Tile(col, 0),
            )
            if col < AIE_ROWS:
                runtime.fill(
                    a_l3l2[col].prod(),
                    a,
                    tap=a_tiles[col],
                    task_group=group,
                    tile=Tile(2 * col, 0),
                )
            runtime.fill(
                b_l3l2[col].prod(),
                b,
                tap=b_tiles[col],
                task_group=group,
                tile=Tile(col, 0),
            )
        runtime.finish_task_group(group)

    return Program(NPU2(), runtime).resolve_program()


if __name__ == "__main__":
    print(build_full_array())
