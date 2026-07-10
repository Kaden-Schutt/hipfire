# Copyright (c) Kaden Schutt
"""autoresearch.ar.certify — the three-arm A/B certify gate (parity → perf → coherence).

Decision logic is pure-stdlib and no-GPU-unit-testable; the GPU work lives behind
the ``ServeRunner`` seam in ``orchestrator``/``serve_runner``.
"""
