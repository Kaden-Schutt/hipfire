# Crate maps

Each crate may carry a `map.md` at its crate root (`crates/<name>/map.md`) so
humans and agents can navigate the workspace without reading source first.
The first screen of every map is summary plus table of contents; the
mechanical half is **generated**, because hand-written structure rots (the
pre-saddle daemon path survived in docs for weeks after the file moved).

## What a map owns — and what it does not

A map answers "what is in this crate and who touches it." It does **not**
restate what existing docs already own:

- [`ARCHITECTURE.md`](ARCHITECTURE.md) owns the workspace tree, the layering
  DAG, and the request lifecycle. A map names its layer and links; it never
  re-draws the DAG.
- [`GLOSSARY.md`](GLOSSARY.md) owns named subsystems and the status
  vocabulary `production` / `research` / `legacy`. Maps reuse exactly that
  vocabulary for crate status; they do not redefine DFlash, PFlash, etc.
- [`INDEX.md`](INDEX.md) owns doc lifecycle truth states
  (shipped / branch-implemented / measured / planned). Those are doc states,
  not crate maturity — keep the two vocabularies separate.
- The crate's own `//!` docs own design rationale. Prefer linking
  `src/lib.rs` over copying prose into the map.

## Marker convention

The mechanical section is fenced by explicit markers:

```text
<!-- crate-map:generated:begin -->
...modules, public API, dependencies, reverse dependencies, totals...
<!-- crate-map:generated:end -->
```

Regeneration replaces **only** the text between the markers; everything
outside (purpose, status, gotchas) is hand-written judgement and is preserved
byte-for-byte. Never hand-edit inside the markers — rerun the generator.

The generated section carries, all measured from the tree:

- module inventory: every `src/**/*.rs` with line counts, public-item counts,
  and per-file test counts (doubles as the TOC, with relative links);
- public API surface: top-level `pub` items per file (capped, `+N more`);
- direct dependencies from `Cargo.toml`, split path / external / dev / build;
- reverse dependencies: workspace crates with a path dependency on this one;
- totals, including `tests/` and `examples/` counts.

## Running it

```bash
# Generate or refresh one or more maps (safe to rerun; touches only the
# generated block):
scripts/check-crate-maps.py saddle-core

# Drift check — every map present under crates/, or a named subset:
scripts/check-crate-maps.py --check
scripts/check-crate-maps.py --check saddle-core
```

Exit codes mirror [`scripts/check-env-docs.py`](../scripts/check-env-docs.py):
`0` clean, `1` drift found (one line per finding), `2` usage error or a named
crate without a map.

`--check` fails when:

- a `src/**/*.rs` file exists with no row in the module table;
- the module table lists a file that no longer exists;
- a declared dependency (or reverse dependency) in the map is stale or
  missing relative to `Cargo.toml`;
- any generated count has drifted (line counts, public items, tests).

Fix drift by rerunning the generator for that crate — never by editing inside
the markers. Crates without a `map.md` are reported as a count but do not
fail the check; adoption is incremental, exemplar
[`crates/saddle-core/map.md`](../crates/saddle-core/map.md).
