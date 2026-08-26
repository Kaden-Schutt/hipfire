#!/usr/bin/env python3
"""Focused regression tests for tools/quant-design/mq4c_repack.py."""
from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools" / "quant-design"))

import mq4c_repack as repack  # noqa: E402


HFQM_MAGIC = b"HFQM"
QT_V1 = 13
QT_V15 = 45
GROUP_BYTES = 136


def _pack_v1_group(scale: float, zero: float, nibble_fill: int = 0x12) -> bytes:
    s = np.float32(scale).tobytes()
    z = np.float32(zero).tobytes()
    nib = bytes([nibble_fill & 0xFF]) * 128
    return s + z + nib


def _pack_entry(name: str, qt: int, shape: list[int], group_size: int, data_len: int) -> bytes:
    nb = name.encode("utf-8")
    out = bytearray()
    out += struct.pack("<H", len(nb))
    out += nb
    out += bytes([qt & 0xFF, len(shape) & 0xFF])
    for d in shape:
        out += struct.pack("<I", d)
    out += struct.pack("<I", group_size)
    out += struct.pack("<Q", data_len)
    return bytes(out)


def write_hfqm(
    path: Path,
    tensors: list[tuple[str, int, list[int], int, bytes]],
    *,
    magic: bytes = HFQM_MAGIC,
    version: int = 1,
    arch: int = 7,
    metadata: bytes = b'{"format":"test"}',
    align: int = 64,
) -> None:
    """Write a minimal canonical HFQM file for tests."""
    index = bytearray()
    index += struct.pack("<I", len(tensors))
    for name, qt, shape, gs, data in tensors:
        index += _pack_entry(name, qt, shape, gs, len(data))

    metadata_offset = 32
    unaligned = metadata_offset + len(metadata) + len(index)
    if align > 1:
        data_offset = (unaligned + (align - 1)) & ~(align - 1)
    else:
        data_offset = unaligned
    pad = data_offset - unaligned

    with path.open("wb") as f:
        f.write(magic)
        f.write(struct.pack("<I", version))
        f.write(struct.pack("<I", arch))
        f.write(struct.pack("<I", len(tensors)))
        f.write(struct.pack("<Q", metadata_offset))
        f.write(struct.pack("<Q", data_offset))
        f.write(metadata)
        f.write(index)
        if pad:
            f.write(b"\x00" * pad)
        for *_, data in tensors:
            f.write(data)


def read_header(path: Path) -> tuple[bytes, int, int, int, int, int]:
    raw = path.read_bytes()
    magic = raw[0:4]
    version, arch, n = struct.unpack_from("<III", raw, 4)
    meta_off, data_off = struct.unpack_from("<QQ", raw, 16)
    return magic, version, arch, n, meta_off, data_off


def read_index_qts(path: Path) -> list[tuple[str, int, int]]:
    """Return (name, qt, data_len) from the canonical packed index."""
    idx = repack.parse_hfqm_index(path)
    return [(e.name, e.qt, e.data_len) for e in idx.entries]


def read_tensor_payload(path: Path, name: str) -> bytes:
    idx = repack.parse_hfqm_index(path)
    entry = next(e for e in idx.entries if e.name == name)
    with path.open("rb") as f:
        f.seek(entry.data_off)
        return f.read(entry.data_len)


class TestMq4cRepack(unittest.TestCase):
    def test_full_conversion_patches_qt_and_payload(self) -> None:
        g0 = _pack_v1_group(0.25, -1.5, 0xAB)
        g1 = _pack_v1_group(1.0, 0.0, 0x34)
        other = b"\xDE\xAD" * 8  # non-qt13 payload preserved byte-for-byte

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            src = td_path / "in.mq4"
            dst = td_path / "out.mq4c"
            write_hfqm(
                src,
                [
                    ("w_a", QT_V1, [256, 256], 256, g0 + g1),
                    ("bias", 0, [4], 0, other),
                    ("w_b", QT_V1, [256], 256, g0),
                ],
            )

            rc = repack.repack(src, dst, limit=None)
            self.assertEqual(rc, 0)
            self.assertEqual(dst.stat().st_size, src.stat().st_size)

            magic, version, arch, n, meta_off, data_off = read_header(dst)
            self.assertEqual(magic, HFQM_MAGIC)
            self.assertEqual(version, 1)
            self.assertEqual(arch, 7)
            self.assertEqual(n, 3)
            self.assertEqual(meta_off, 32)
            self.assertGreaterEqual(data_off, 32)

            qts = read_index_qts(dst)
            self.assertEqual(qts, [("w_a", QT_V15, 272), ("bias", 0, 16), ("w_b", QT_V15, 136)])

            # Unselected tensor payload identical.
            self.assertEqual(read_tensor_payload(dst, "bias"), other)

            # Pad layout: fp16 scale/zero, 4 zero pad, same nibbles.
            pa = read_tensor_payload(dst, "w_a")
            self.assertEqual(len(pa), 272)
            for base, scale, zero, fill in (
                (0, 0.25, -1.5, 0xAB),
                (136, 1.0, 0.0, 0x34),
            ):
                blk = pa[base : base + 136]
                s16 = np.frombuffer(blk[0:2], dtype="<f2")[0]
                z16 = np.frombuffer(blk[2:4], dtype="<f2")[0]
                self.assertEqual(float(s16), np.float16(scale).astype(np.float32))
                self.assertEqual(float(z16), np.float16(zero).astype(np.float32))
                self.assertEqual(blk[4:8], b"\x00\x00\x00\x00")
                self.assertEqual(blk[8:], bytes([fill]) * 128)

            # Source header/index qt bytes untouched.
            src_qts = read_index_qts(src)
            self.assertEqual(src_qts, [("w_a", QT_V1, 272), ("bias", 0, 16), ("w_b", QT_V1, 136)])

    def test_limit_tensors_patches_only_converted_entries(self) -> None:
        g = _pack_v1_group(0.5, 0.25, 0x11)
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            src = td_path / "in.mq4"
            dst = td_path / "out.mq4c"
            write_hfqm(
                src,
                [
                    ("t0", QT_V1, [256], 256, g),
                    ("t1", QT_V1, [256], 256, g),
                    ("t2", QT_V1, [256], 256, g),
                ],
            )

            rc = repack.repack(src, dst, limit=1)
            self.assertEqual(rc, 0)

            qts = read_index_qts(dst)
            self.assertEqual(
                qts,
                [
                    ("t0", QT_V15, 136),
                    ("t1", QT_V1, 136),  # not converted
                    ("t2", QT_V1, 136),
                ],
            )

            # Only t0 payload transformed; t1 remains v1 f32 header.
            p0 = read_tensor_payload(dst, "t0")
            self.assertEqual(p0[4:8], b"\x00\x00\x00\x00")
            s16 = float(np.frombuffer(p0[0:2], dtype="<f2")[0])
            self.assertAlmostEqual(s16, float(np.float16(0.5)), places=5)

            p1 = read_tensor_payload(dst, "t1")
            s32 = float(np.frombuffer(p1[0:4], dtype="<f4")[0])
            z32 = float(np.frombuffer(p1[4:8], dtype="<f4")[0])
            self.assertEqual(s32, 0.5)
            self.assertEqual(z32, 0.25)
            self.assertEqual(p1[8:], bytes([0x11]) * 128)

    def test_bad_magic_rejected(self) -> None:
        g = _pack_v1_group(1.0, 0.0, 0x00)
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            src = td_path / "bad.mq4"
            dst = td_path / "out.mq4c"
            write_hfqm(src, [("t0", QT_V1, [256], 256, g)], magic=b"HFQ\x00")
            with self.assertRaises(repack.HfqmError) as ctx:
                repack.repack(src, dst)
            self.assertIn("bad magic", str(ctx.exception).lower())
            self.assertFalse(dst.exists())

    def test_drift_refusal_preserves_destination_sentinel(self) -> None:
        # Drift = |(z32-z16) + q*(s32-s16)| / s16. Tiny scale + large f32 zero
        # that collapses under f16 exceeds the 0.5 boundary even at q=0.
        scale = 1e-3
        zero = 1000.3  # finite in f16, but rounds far enough to cross the drift guard
        s16 = float(np.float16(scale))
        z16 = float(np.float16(zero))
        drift0 = abs(zero - z16) / s16
        self.assertGreaterEqual(drift0, 0.5, f"synthetic drift too small: {drift0}")
        g = _pack_v1_group(scale, zero, 0x00)

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            src = td_path / "drift.mq4"
            dst = td_path / "out.mq4c"
            sentinel = b"SENTINEL-DEST-UNCHANGED\n"
            dst.write_bytes(sentinel)
            write_hfqm(src, [("bad", QT_V1, [256], 256, g)])

            with self.assertRaises(repack.HfqmError) as ctx:
                repack.repack(src, dst)
            self.assertIn("drift", str(ctx.exception).lower())
            # Refusal must not truncate or replace the pre-existing destination.
            self.assertEqual(dst.read_bytes(), sentinel)
            # No leftover temp partials next to destination.
            leftovers = list(td_path.glob(".out.mq4c.*.tmp"))
            self.assertEqual(leftovers, [])

    def test_cli_limit_tensors_equals_and_space(self) -> None:
        g = _pack_v1_group(0.5, 0.0, 0x22)
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            src = td_path / "in.mq4"
            dst_a = td_path / "a.mq4c"
            dst_b = td_path / "b.mq4c"
            write_hfqm(
                src,
                [
                    ("t0", QT_V1, [256], 256, g),
                    ("t1", QT_V1, [256], 256, g),
                ],
            )
            self.assertEqual(repack.main([str(src), str(dst_a), "--limit-tensors=1"]), 0)
            self.assertEqual(repack.main([str(src), str(dst_b), "--limit-tensors", "1"]), 0)
            self.assertEqual(read_index_qts(dst_a), [("t0", QT_V15, 136), ("t1", QT_V1, 136)])
            self.assertEqual(read_index_qts(dst_b), [("t0", QT_V15, 136), ("t1", QT_V1, 136)])


if __name__ == "__main__":
    unittest.main()
