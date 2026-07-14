# Copyright (c) Kaden Schutt
"""``python -m autoresearch.ar`` → the role-scoped ``ar`` CLI."""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
