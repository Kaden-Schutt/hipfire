#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

import hashlib
import json
import os
import stat
import tarfile
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
LMX = REPO / "scripts" / "lmx_corpus.py"

# Import lmx_corpus as module without polluting sys path
import importlib.util
spec = importlib.util.spec_from_file_location("lmx_corpus", str(LMX))
assert spec and spec.loader
lmx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lmx)  # type: ignore


def _make_case_dir(root: Path, family: str, case_id: str, content_map: dict | None = None) -> Path:
    """Create a minimal sealed case directory under root/family/case_id.
    Returns case path. Creates required canonical files and raw/decoded/validation subdirs.
    content_map overrides default content for specific rel paths.
    """
    case_path = root / family / case_id
    case_path.mkdir(parents=True, exist_ok=True)
    for sub in ["raw", "decoded", "validation"]:
        (case_path / sub).mkdir(parents=True, exist_ok=True)

    # default content
    defaults = {
        "metadata.json": json.dumps({"route": "pm4", "valid": True, "host": "testhost", "measurements": {"median_tok_s": 100}}, sort_keys=True),
        "summary.json": json.dumps({"median_tok_s": 100, "route": "plain"}, sort_keys=True),
        "command.txt": "hipfire run --test\n",
        "environment.txt": "PATH=/usr/bin\n",
        "raw/a.txt": "raw content\n",
        "decoded/b.txt": "decoded\n",
        "validation/c.txt": "validation\n",
    }
    if content_map:
        defaults.update(content_map)

    for rel, data in defaults.items():
        p = case_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, str):
            p.write_text(data, encoding="utf-8")
        else:
            p.write_bytes(data if isinstance(data, bytes) else str(data).encode())

    # compute checksums
    checksums = lmx.compute_case_checksums(str(case_path))
    lmx.write_checksums_file(str(case_path / "checksums.sha256"), checksums)
    return case_path


def _init_root(root: Path):
    """Create minimal corpus root with machine.json and required top dirs."""
    for d in lmx.REQUIRED_TOP_DIRS:
        (root / d).mkdir(parents=True, exist_ok=True)
    machine = {
        "hostname": "testhost",
        "uname": {"node": "testhost", "system": "Linux"},
        "platform": "Linux-test",
    }
    # also include what capture_machine_info would generate
    lmx.atomic_write_json(str(root / "machine.json"), machine)


class TestDeterministicHelpers(unittest.TestCase):
    def test_normalize_tarinfo_deterministic(self):
        ti = tarfile.TarInfo(name="foo/bar.txt")
        ti.mtime = 123456
        ti.uid = 1000
        ti.gid = 1000
        ti.uname = "user"
        ti.gname = "group"
        ti.mode = 0o777
        ti.size = 123
        ti.type = tarfile.REGTYPE
        out = lmx._normalize_tarinfo(ti)
        self.assertEqual(out.mtime, 0)
        self.assertEqual(out.uid, 0)
        self.assertEqual(out.gid, 0)
        self.assertEqual(out.uname, "")
        self.assertEqual(out.gname, "")
        self.assertEqual(out.mode, 0o644)
        self.assertEqual(out.pax_headers, {})

        ti2 = tarfile.TarInfo(name="mydir")
        ti2.type = tarfile.DIRTYPE
        ti2.mode = 0o777
        out2 = lmx._normalize_tarinfo(ti2)
        self.assertEqual(out2.mode, 0o755)

    def test_build_package_index_without_inventing(self):
        # metadata missing identity fields -> identity not invented
        idx = lmx._build_package_index(
            family="qwen-0.8b-mq4",
            case_id="case1",
            hostname="host1",
            machine_identity_hash="abc",
            source_manifest_hash="def",
            case_file_checksums={"a": "b"},
            metadata={},  # empty
            summary={},
        )
        self.assertNotIn("identity", idx)
        self.assertNotIn("route", idx)
        self.assertNotIn("metrics", idx)
        # when present, they appear
        idx2 = lmx._build_package_index(
            family="qwen-0.8b-mq4",
            case_id="case1",
            hostname="host1",
            machine_identity_hash="abc",
            source_manifest_hash="def",
            case_file_checksums={"a": "b"},
            metadata={"host": "h1", "route_proof": {"transport": "pm4"}, "measurements": {"median_tok_s": 1}},
            summary={},
        )
        self.assertIn("identity", idx2)
        self.assertIn("route", idx2)
        self.assertIn("metrics", idx2)

    def test_sanitize_absolute_paths(self):
        idx = lmx._build_package_index(
            family="qwen-0.8b-mq4",
            case_id="c1",
            hostname="host1",
            machine_identity_hash=None,
            source_manifest_hash=None,
            case_file_checksums={},
            metadata={"host": "h1", "hardware": {"path": "/absolute/secret/path"}, "model": {"path": "/models/foo.gguf", "sha256": "abc"}},
            summary={},
        )
        # absolute paths should be sanitized to basename, not remain absolute
        if "identity" in idx:
            ident = idx["identity"]
            # check no value starts with "/"
            def check_no_absolute(v):
                if isinstance(v, str):
                    self.assertFalse(v.startswith("/"), f"absolute path leaked: {v}")
                elif isinstance(v, dict):
                    for vv in v.values():
                        check_no_absolute(vv)
                elif isinstance(v, list):
                    for vv in v:
                        check_no_absolute(vv)
            check_no_absolute(ident)

    def test_per_case_tar_deterministic_byte_identical(self):
        with tempfile.TemporaryDirectory() as td:
            td1 = Path(td) / "root1"
            td1.mkdir()
            _init_root(td1)
            case1 = _make_case_dir(td1, "qwen-0.8b-mq4", "case-a")
            # also need manifest for packaging
            # create minimal manifest via lmx.cmd_manifest logic: just write a manifest with no packages
            # Instead, create MANIFEST.json manually for source hash
            manifest_hash = hashlib.sha256(b"dummy").hexdigest()
            (td1 / "MANIFEST.json").write_text(json.dumps({"schema": "lmx-corpus/1", "cases": []}), encoding="utf-8")
            (td1 / "MANIFEST.sha256").write_text(f"{lmx.sha256_file(str(td1 / 'MANIFEST.json'))}  MANIFEST.json\n", encoding="utf-8")

            idx = lmx._build_package_index(
                family="qwen-0.8b-mq4", case_id="case-a", hostname="testhost",
                machine_identity_hash=lmx.sha256_file(str(td1 / "machine.json")),
                source_manifest_hash=lmx.sha256_file(str(td1 / "MANIFEST.json")),
                case_file_checksums=dict(lmx.compute_case_checksums(str(case1))),
                metadata=json.loads((case1 / "metadata.json").read_text()),
                summary=json.loads((case1 / "summary.json").read_text()),
            )
            tmp1 = Path(td) / "a1.tar"
            tmp2 = Path(td) / "a2.tar"
            lmx._write_deterministic_case_tar(str(case1), str(tmp1), idx)
            lmx._write_deterministic_case_tar(str(case1), str(tmp2), idx)
            self.assertEqual(tmp1.read_bytes(), tmp2.read_bytes())
            self.assertEqual(hashlib.sha256(tmp1.read_bytes()).hexdigest(), hashlib.sha256(tmp2.read_bytes()).hexdigest())

    def test_mtime_invariance(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            root = td / "root"
            root.mkdir()
            _init_root(root)
            case = _make_case_dir(root, "qwen-0.8b-mq4", "case-b")
            idx = lmx._build_package_index(
                family="qwen-0.8b-mq4", case_id="case-b", hostname="testhost",
                machine_identity_hash="abc", source_manifest_hash="def",
                case_file_checksums=dict(lmx.compute_case_checksums(str(case))),
                metadata={}, summary={},
            )
            tmp1 = td / "t1.tar"
            lmx._write_deterministic_case_tar(str(case), str(tmp1), idx)
            h1 = hashlib.sha256(tmp1.read_bytes()).hexdigest()
            # touch source mtimes to future
            future = time.time() + 10000
            for p in case.rglob("*"):
                if p.is_file():
                    os.utime(p, (future, future))
            tmp2 = td / "t2.tar"
            lmx._write_deterministic_case_tar(str(case), str(tmp2), idx)
            h2 = hashlib.sha256(tmp2.read_bytes()).hexdigest()
            self.assertEqual(h1, h2)
            self.assertEqual(tmp1.read_bytes(), tmp2.read_bytes())

    def test_per_case_package_excludes_other_cases_and_host_raw(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            root = td / "root"
            root.mkdir()
            _init_root(root)
            # top-level raw/validation that must not appear in per-case tar
            (root / "raw").mkdir(exist_ok=True)
            (root / "raw" / "top-secret.txt").write_text("secret")
            (root / "validation" / "top-level.txt").write_text("x")
            # Create two cases
            case_a = _make_case_dir(root, "qwen-0.8b-mq4", "case-a", {"raw/secret.txt": "a"})
            case_b = _make_case_dir(root, "qwen-0.8b-mq4", "case-b", {"raw/secret.txt": "b"})
            # Create manifest so package-cases can use source hash
            # Run manifest generation via _regenerate_manifest
            lmx._regenerate_manifest(str(root))
            idx_a = lmx._build_package_index(
                family="qwen-0.8b-mq4", case_id="case-a", hostname="testhost",
                machine_identity_hash=lmx.sha256_file(str(root / "machine.json")),
                source_manifest_hash=lmx.sha256_file(str(root / "MANIFEST.json")),
                case_file_checksums=dict(lmx.compute_case_checksums(str(case_a))),
                metadata={}, summary={},
            )
            tmp = td / "case-a.tar"
            lmx._write_deterministic_case_tar(str(case_a), str(tmp), idx_a)
            with tarfile.open(str(tmp), "r") as tar:
                names = [m.name for m in tar.getmembers()]
                # must not contain other case's files, top-level raw, or lmx-packages
                for n in names:
                    self.assertFalse(n.startswith("lmx-packages"), f"per-case tar leaked lmx-packages: {n}")
                    self.assertFalse(n.startswith("raw/top-secret"), f"leaked top-level raw: {n}")
                    self.assertFalse("case-b" in n, f"leaked other case: {n}")
                self.assertIn("PACKAGE.json", names)
                self.assertIn("raw/secret.txt", names)

    def test_host_archive_excludes_self(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            root = td / "root"
            root.mkdir()
            _init_root(root)
            case = _make_case_dir(root, "qwen-0.8b-mq4", "case-x")
            lmx._regenerate_manifest(str(root))
            # Create a per-case package first (simulate package-cases)
            pkg_dir = root / "lmx-packages" / "cases" / "qwen-0.8b-mq4"
            pkg_dir.mkdir(parents=True, exist_ok=True)
            idx = lmx._build_package_index(
                family="qwen-0.8b-mq4", case_id="case-x", hostname="testhost",
                machine_identity_hash=lmx.sha256_file(str(root / "machine.json")),
                source_manifest_hash=lmx.sha256_file(str(root / "MANIFEST.json")),
                case_file_checksums=dict(lmx.compute_case_checksums(str(case))),
                metadata={}, summary={},
            )
            tmp_pkg = pkg_dir / "case-x.tar"
            lmx._write_deterministic_case_tar(str(case), str(tmp_pkg), idx)
            (pkg_dir / "case-x.tar.sha256").write_text(f"{lmx.sha256_file(str(tmp_pkg))}  lmx-packages/cases/qwen-0.8b-mq4/case-x.tar\n", encoding="utf-8")
            # need to regenerate manifest to include package hash
            lmx._regenerate_manifest(str(root))
            # Now write host tar
            tmp_host = td / "host.tar"
            lmx._write_deterministic_host_tar(str(root), str(tmp_host))
            with tarfile.open(str(tmp_host), "r") as tar:
                names = [m.name for m in tar.getmembers()]
                for n in names:
                    self.assertFalse(n.startswith("lmx-packages/host-archives"), f"host archive includes itself: {n}")
                self.assertIn("machine.json", names)
                self.assertIn("MANIFEST.json", names)
                self.assertIn("qwen-0.8b-mq4/case-x/metadata.json", names)
                self.assertIn("lmx-packages/cases/qwen-0.8b-mq4/case-x.tar", names)

    def test_manifest_package_hashes_excludes_host_archives(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            root = td / "root"
            root.mkdir()
            _init_root(root)
            case = _make_case_dir(root, "qwen-0.8b-mq4", "case-h")
            # create host-archives file that would previously have been included
            host_dir = root / "lmx-packages" / "host-archives"
            host_dir.mkdir(parents=True, exist_ok=True)
            (host_dir / "dummy.tar").write_text("dummy")
            lmx._regenerate_manifest(str(root))
            manifest = json.loads((root / "MANIFEST.json").read_text())
            for rel in manifest.get("package_hashes", {}):
                self.assertFalse(rel.startswith("lmx-packages/host-archives"), f"manifest leaked host-archives: {rel}")

    def test_corrupt_checksum_aborts(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            root = td / "root"
            root.mkdir()
            _init_root(root)
            case = _make_case_dir(root, "qwen-0.8b-mq4", "case-corrupt")
            # corrupt checksums
            (case / "checksums.sha256").write_text("badhash  metadata.json\n", encoding="utf-8")
            # _verify_all_cases should sys.exit
            with self.assertRaises(SystemExit):
                lmx._verify_all_cases(str(root), lmx._discover_cases(str(root)))

    def test_corrupt_package_sidecar_aborts(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            root = td / "root"
            root.mkdir()
            _init_root(root)
            case = _make_case_dir(root, "qwen-0.8b-mq4", "case-s1")
            lmx._regenerate_manifest(str(root))
            # create package-cases first (use helper directly to simulate success)
            # then corrupt sidecar and attempt package-host verification
            pkg_dir = root / "lmx-packages" / "cases" / "qwen-0.8b-mq4"
            pkg_dir.mkdir(parents=True, exist_ok=True)
            idx = lmx._build_package_index("qwen-0.8b-mq4", "case-s1", "testhost", "abc", "def", dict(lmx.compute_case_checksums(str(case))), {}, {})
            tmp_pkg = pkg_dir / "case-s1.tar"
            lmx._write_deterministic_case_tar(str(case), str(tmp_pkg), idx)
            # write correct sidecar then corrupt
            (pkg_dir / "case-s1.tar.sha256").write_text(f"{lmx.sha256_file(str(tmp_pkg))}  lmx-packages/cases/qwen-0.8b-mq4/case-s1.tar\n", encoding="utf-8")
            lmx._regenerate_manifest(str(root))
            # corrupt
            (pkg_dir / "case-s1.tar.sha256").write_text("0" * 64 + "  lmx-packages/cases/qwen-0.8b-mq4/case-s1.tar\n", encoding="utf-8")
            with self.assertRaises(SystemExit):
                lmx.cmd_package_host(SimpleNamespace(root=str(root)))

    def test_validate_archive_names_rejects_symlink(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            root = td / "root"
            root.mkdir()
            _init_root(root)
            case = _make_case_dir(root, "qwen-0.8b-mq4", "case-sym")
            # create symlink inside case
            (case / "raw" / "link.txt").symlink_to(case / "raw" / "a.txt")
            idx = lmx._build_package_index("qwen-0.8b-mq4", "case-sym", "testhost", None, None, {}, {}, {})
            tmp = td / "out.tar"
            with self.assertRaises(SystemExit):
                lmx._write_deterministic_case_tar(str(case), str(tmp), idx)


if __name__ == "__main__":
    unittest.main()
