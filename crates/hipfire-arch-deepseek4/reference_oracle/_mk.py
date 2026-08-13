# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt <kaden@hipfire.dev>
from pathlib import Path
HERE = Path('.').resolve()
print('HERE', HERE)

# Fix empty stubs: symlink model.py and config.json to research tree
import os
research = HERE
for p in HERE.parents:
    cand = p / '.codeinsight+research' / 'ds4-parent-ref' / 'inference'
    if (cand / 'model.py').is_file():
        research = cand
        break
print('research', research)
for name in ('model.py', 'config.json'):
    dst = HERE / name
    src = research / name
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)
    print('linked', name, '->', src, 'size', src.stat().st_size)
print('DONE_LINKS')
