# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""CLI entry: python3 -m tools.redline {golden|bench|serve-diff|lower} ..."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(
            "usage: python3 -m tools.redline {golden|bench|serve-diff|lower} ...",
            file=sys.stderr,
        )
        return 0 if args and args[0] in {"-h", "--help"} else 2

    command, rest = args[0], args[1:]
    if command == "golden":
        from tools.redline.golden import GoldenError, main as golden_main

        try:
            return golden_main(rest)
        except GoldenError as exc:
            print(f"golden-redline: {exc}", file=sys.stderr)
            return 2
    if command == "bench":
        from tools.redline.product_bench import main as bench_main

        bench_main(rest)
        return 0

    if command == "serve-diff":
        from tools.redline.serve_diff import main as serve_diff_main

        return serve_diff_main(rest)

    if command == "lower":
        from tools.redline.lower import main as lower_main

        return lower_main(rest)

    print(
        f"tools.redline: unknown subcommand {command!r} "
        f"(expected golden, bench, serve-diff, or lower)",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
