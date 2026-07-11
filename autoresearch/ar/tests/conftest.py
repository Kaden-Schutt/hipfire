# Copyright (c) Kaden Schutt
"""Shared pytest scaffolding for the migrated autoresearch harness tests.

Two jobs:

1. **Package imports.** Ensure the repo root is on ``sys.path`` so
   ``from autoresearch.ar.config import ...`` resolves when the suite is run
   as ``pytest autoresearch/ar/tests/``. The ``__init__.py`` chain
   (``autoresearch`` → ``ar`` → ``tests``) already makes pytest's default
   ``prepend`` import mode insert the repo root, but we belt-and-brace it here
   so the imports resolve regardless of the caller's CWD or import mode.
   This conftest lives *inside* ``autoresearch/ar/tests/`` on purpose: it is
   NOT loaded by ``pytest tests scripts/test_astrea.py`` (the no-GPU CI line),
   so it cannot perturb the existing suites.

2. **Shared fixtures.** ``mini_ledger`` and ``bod_gfx1201`` expose the static
   fixture files (derived from the real ledger + ``state/bod_gfx1201.json``)
   as absolute paths for later phases. The Phase-1 plan tests reference the
   fixtures by their repo-root-relative literal paths, so run the suite from
   the repo root.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# .../autoresearch/ar/tests -> repo root is three levels up.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_FIXTURES = os.path.join(_HERE, "fixtures")

import pytest


@pytest.fixture
def repo_root():
    """Absolute path to the repository root."""
    return _REPO_ROOT


@pytest.fixture
def mini_ledger():
    """Absolute path to the 3-row mini ledger fixture dir (1 WIN, 1 DEAD, 1 noise)."""
    return os.path.join(_FIXTURES, "mini_ledger")


@pytest.fixture
def bod_gfx1201():
    """Absolute path to the gfx1201 BOD census fixture (derived from state/bod_gfx1201.json)."""
    return os.path.join(_FIXTURES, "bod_gfx1201.json")


@pytest.fixture
def loop_toml():
    """Absolute path to the stable 3-worker loop-config fixture. Decouples
    loader/config-verb tests from the campaign-mutable loop_gfx1201.toml."""
    return os.path.join(_FIXTURES, "loop_test.toml")
