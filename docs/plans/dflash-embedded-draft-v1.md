# DFlash Drafts Inside Target HFQ: v1 Design

Date: 2026-05-17

## Goal

Make a target `.mq4`/HFQ artifact optionally carry the paired DFlash draft so
install/pull can become single-file while existing external draft
auto-discovery keeps working. This note is design-only; no draft tensors are
embedded in this pass.

## Container Constraints

HFQ currently has a fixed header, one JSON metadata blob, one tensor index, and
a contiguous tensor-data region. Older loaders compute tensor offsets by walking
the index and expect `data_end == file_size`. Any v1 that appends opaque bytes
after tensor data will look corrupt to strict tools and will fail the existing
Astrea `data_end_matches_file_size` check.

Tensor names are arbitrary strings and quant type IDs are already versioned by
convention, so the safest compatibility surface is still "metadata plus normal
tensor records" rather than changing the header layout.

## Options

### 1. External Reference Metadata

Add metadata like:

```json
"dflash": {
  "mode": "external",
  "draft_filename": "qwen35-27b-dflash-mq4.hfq",
  "draft_md5": "...",
  "target_family": "qwen3.5",
  "target_size": "27b"
}
```

Pros: zero container risk, works with current loader model, easy for pull/install
to validate. Cons: still two files, still vulnerable to rename/path drift, and
not a real single-file package.

### 2. Namespaced Embedded Tensors

Store draft tensors as ordinary HFQ tensor records under a reserved namespace:

```text
__dflash_draft__/model.layers.0.self_attn.q_proj.weight
__dflash_draft__/lm_head.weight
```

Metadata declares the embedded draft index:

```json
"dflash": {
  "mode": "embedded_tensors",
  "schema": "hipfire.dflash.embedded.v1",
  "arch_id": 5,
  "tensor_prefix": "__dflash_draft__/",
  "quant_format": "mq4",
  "source_md5": "..."
}
```

Pros: compatible with the current physical container, older loaders ignore
unknown tensors by name, no trailing payload ambiguity, tensor data remains
memory-mappable and page-droppable, and pull/install can still checksum one
file. Cons: increases tensor count substantially and requires loader code to
build a second `HfqFile`-like view over prefixed records.

### 3. Appended Nested HFQ Payload

Append a complete draft HFQ after the target tensor data and point to it from
metadata with offset/length/md5.

Pros: preserves the draft as an independent byte-identical HFQ blob and allows
easy extraction. Cons: breaks strict `data_end == file_size` assumptions, needs
new footer/offset validation, complicates mmap lifetime, and is the most likely
to confuse old tooling.

## Recommendation

Use **namespaced embedded tensors** for v1, with external reference metadata as
the migration bridge. Do not append nested HFQ payloads unless the HFQ container
gets an explicit v2 footer and all strict readers are updated first.

Implementation path:

1. Writer: add an optional embed step that opens the draft HFQ and copies its
   tensor records into the target writer with `__dflash_draft__/` prefixed
   names. Preserve draft quant types, shapes, group sizes, and data bytes.
2. Metadata: add `dflash.mode = "embedded_tensors"`, the prefix, draft arch id,
   tensor count, original draft md5, and the target compatibility key currently
   inferred from filenames.
3. Loader: when `dflash_mode` is `auto`/`on`, prefer embedded draft metadata,
   construct a draft tensor map from the reserved prefix, and fall back to the
   current external filename auto-discovery when metadata is absent.
4. Pull/install: continue publishing external drafts, then add single-file
   packages once loader support lands. Keep filename auto-discovery so old
   models and existing user caches remain valid.
5. Compatibility: old runtimes should still load the target weights because the
   draft tensors are just unknown records. They will pay metadata/index parse
   cost and file size only; no DFlash activation changes unless the new metadata
   is understood.

Open risk: a few tools treat unexpected `data_end != file_size` as corruption,
which is why v1 avoids appended payloads. Tools that enumerate every tensor may
need filters to hide `__dflash_draft__/` records from quality summaries.
