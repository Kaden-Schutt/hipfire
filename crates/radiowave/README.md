# Radiowave

Radiowave is Redline's architecture-neutral, policy-driven HIP compiler
boundary. Applications give Radiowave HIP source rather than invoking hipcc
directly. Radiowave injects reviewed lowering helpers, calls the installed ROCm
compiler for the requested target, inspects the resulting AMDGPU code object,
and writes a reproducible JSON manifest.

The compiler and recipe schema do not special-case a GPU family. A recipe
describes a semantic source or launch transformation; separate evidence records
certify it for a concrete architecture. A proven gfx1201 recipe is therefore a
candidate on gfx942, gfx11, or a future target, but it is not selected there
until that target produces its own correctness-passing performance WIN.

The first promoted lowering family is the buffer-resource load/store layer
derived from the ROCm issue 6409 benchmark. It lets reviewed HIP source request
the same 32-bit-offset buffer instruction class that Vulkan SSBOs naturally
expose to RADV/ACO. A live HIP probe now validates its B32 load/store contract
on gfx1010, gfx1030, gfx1100, and gfx1151 in addition to the gfx1201 benchmark.
Those portability checks are correctness evidence, not performance promotion.

```bash
cargo run -p radiowave -- compile \
  --source kernel.hip \
  --output kernel.hsaco \
  --arch gfx1201 \
  --wave32 \
  --scheduler-profile default
```

Radiowave force-includes `radiowave/hip.h`. Kernels can use:

```cpp
const auto resource = radiowave::buffer_resource(words);
uint32_t value = radiowave::buffer_load_u32(resource, word_offset);
auto four_values = radiowave::buffer_load_u32x4(resource, aligned_word_offset);
radiowave::buffer_store_u32(resource, word_offset, value);
```

These helpers default to AMDGPU cache policy zero: the ordinary temporal path
that retains L0/L1/L2 reuse. Radiowave does not set per-load coherence or cache
bypass bits by default; Redline owns invalidation at the dependency boundary.
Every library or CLI compile also writes an inspected
`*.radiowave.json` manifest beside the code object unless an explicit manifest
path is supplied, so Redline can prove when the lean VMEM-only boundary is
safe. Disabling inspection remains an explicit diagnostic mode and cannot
produce a cache certification accepted by Redline.

For experiments which need adjacent scalar VMEM operations to remain distinct,
`buffer_load_bytes_u32` accepts a byte offset and `opaque_vgpr_u32x4` prevents
LLVM from folding four reviewed offsets back into one B128 load. These helpers
are opt-in: inspection proves the requested instruction shape, while benchmark
correctness and timing decide whether it may be promoted.

## Architecture-neutral recipe catalog

The built-in catalog lowers the accepted Hipfire microbenchmark decisions into
typed actions: wave size, workgroup size, scheduler profile, compiler defines,
named source variant, B32/B128 buffer access, output RMW, unroll/chunk size, and
paired integer scheduling. Kernel selectors and semantic tags say when an
action is relevant; they do not grant cross-architecture trust.

```bash
# Show the built-in reference library.
radiowave recipes builtin --output radiowave-recipes.json

# Consume only transformations certified on this exact target.
radiowave recipes select \
  --arch gfx1201 --kernel memory_interleave4 --family memory-waitcnt \
  --tag timing:independent_throughput

# Ask for candidates to benchmark on a target with no prior evidence.
radiowave recipes select \
  --arch gfx942 --kernel memory_interleave4 --family memory-waitcnt \
  --tag timing:independent_throughput --candidates

# Promote only labeled, correctness-gated WIN rows from autoresearch.
radiowave recipes ingest \
  --catalog radiowave-recipes.json --ledger wins.jsonl \
  --output radiowave-recipes.json
```

An ingestible WIN row names `radiowave_recipe` or `radiowave_recipes` and
contains `arch`/`gpu_arch`, `verdict: "WIN"`, and a measurement or variant hash.
All other rows are ignored. Hipfire exposes this bridge as `ar radiowave`; it
delegates validation and promotion to this CLI so the loop does not acquire a
second recipe implementation.

The catalog reports source lowerings; it does not rewrite an arbitrary C++ AST
behind the caller's back. Today the Hipfire harness maps those requests to its
correctness-gated named kernel variants. A future source transformer can
implement the same typed actions without changing the evidence or selection
format.

The promoted gfx1201 benchmark rules include scalar B32 resource access,
aligned B128 tile loads, and correctness-gated variant selection for wave size,
unroll factor, and workgroup shape. Radiowave does not replace LLVM: it makes
the source and launch-policy decisions explicit, compiles them with upstream
hipcc, and rejects resource or spill regressions using the emitted manifest.

## Scheduler profiles

Radiowave exposes reviewed AMDGPU scheduler configurations as typed profiles,
not arbitrary hidden compiler arguments:

| Profile | LLVM machine-scheduler policy |
|---|---|
| `default` | upstream hipcc defaults |
| `max-ilp` | `gcn-max-ilp` |
| `iterative-ilp` | `gcn-iterative-ilp` |
| `memory-clause` | `gcn-max-memory-clause`, maximum clause length 4 |
| `pipeline-ilp` | max-ILP plus relaxed occupancy, exact igrouplp solving, and the machine pipeliner |

Each profile is compiled into a separate code object. A certified recipe may
select a profile per kernel; an explicit harness `--scheduler-profile` remains
a whole-run override. HIP, HipGraph, and Redline then load the identical object
recorded for each row. Radiowave does not promote a profile merely because its
disassembly is shorter: the candidate must also pass the CPU oracle and a
counterbalanced GPU timing comparison.

## ACO and LLPC compiler oracles

Radiowave can normalize a HIP code-object manifest, one isolated RADV/ACO
shader dump, and amdllpc assembly into schema-versioned reports. The comparison
is diagnostic: it identifies missing memory clauses, extra waits, instruction
count, and register-pressure differences without treating a shorter shader as
a benchmark win.

Resolve Vulkan specialization constants before generating both Vulkan reports.
For example, use `spirv-opt --set-spec-const-default-value` followed by
`--freeze-spec-const`, then pass that same SPIR-V file as `--input-artifact` to
both ACO and LLPC. Radiowave hashes the artifact and calls the comparison
`exact` only when its target, workgroup, wavefront, and input hash all match.
HIP source is recorded as `semantic` evidence because its ABI and input IR are
different even when the kernel implements the same algorithm.

```bash
radiowave oracle aco \
  --input shader.aco.dump --input-artifact shader.wg64.spv \
  --kernel shader --arch gfx1201 --wavefront 64 \
  --output shader.aco.json

radiowave oracle llpc \
  --input shader.llpc.s --input-artifact shader.wg64.spv \
  --kernel shader --arch gfx1201 \
  --output shader.llpc.json

radiowave oracle hip \
  --manifest kernel.radiowave.json --kernel kernel --workgroup 64 \
  --output kernel.hip.json

radiowave oracle compare \
  --baseline shader.aco.json \
  --candidate shader.llpc.json --candidate kernel.hip.json
```

An ACO dump must contain exactly one shader statistics block. This fail-closed
rule avoids silently comparing the wrong pipeline from an application-wide
`RADV_DEBUG=shaders,shaderstats` capture. ACO's allocated-register statistics
and HIP/LLPC code-object metadata use different count bases, so cross-basis
register deltas are emitted as `null`; the raw values and ACO pre-scheduler
statistics remain available in each report.

Instruction accounting stops at `s_endpgm`. Alignment NOPs after program
termination are not executable shader work and are excluded from manifests and
oracle reports.

Each schema-3 compile manifest records source, injected-header, and output
hashes; the complete compiler command and selected scheduler profile; ROCm
version; kernel VGPR/SGPR/private-memory metadata; counts of
buffer/global/flat/scalar memory instructions; static instruction, wait,
`s_delay_alu`, and `s_clause` counts; the longest consecutive VMEM run; and a
fail-closed mutable-read cache classification. Scalar loads from the live
kernarg pointer, or from a proven constant-offset SGPR pair derived from it,
are classified as immutable prologue reads. Either pair stops being trusted as
soon as LLVM overwrites one of its registers. A kernel is `vmem_only` only when
every other resource read is VMEM; scalar-buffer loads and unknown
scalar-memory forms classify as `scalar_or_unknown`.

Redline verifies the embedded manifest schema, wavefront, and code-object
SHA-256 before using that classification. A `vmem_only` consumer can use the
gfx12 same-agent `CS_PARTIAL_FLUSH + GLV/GL1` boundary (`GCR=0x00300`). Missing,
stale, or scalar/unknown evidence falls back to `CS_PARTIAL_FLUSH + GLK/GLV/GL1`
(`0x00380`); certification never weakens the completion edge.

Runtime adapters should construct `CodeObjectCertification::from_json` from
the exact bytes they will load. The C and PyO3 bindings use this same reusable
validator; neither binding independently interprets or trusts manifest JSON.

## Exact architecture campaigns

Architecture-specific tuning uses `ArchProfile`, not a family prefix.  The
initial exact profile is `ArchProfile::Gfx1151`: its resource contract requires
the gfx1151 bundle target, ELF machine id `0x04a`, ISA `{11,5,1}`, wave32, no
register spills or scratch, no lower VGPR occupancy class than the incumbent,
and no more than 32 static memory-clause instructions.  In particular, a
gfx1100 or generic gfx11 object cannot satisfy the gfx1151 contract.
The default remains wave32. A target whose incumbent is already wave64 may use
`ResourceContract::require_wavefront(Wavefront::Wave64)` for that assessment;
the override fails closed when the incumbent is not wave64 and does not alter
the default used by any other gfx1151 target.

```bash
radiowave assess --input router.hsaco --arch gfx1151 --kernel router \
  --incumbent-vgprs 80 --incumbent-wavefront 64 --required-wavefront 64 \
  --out router-resources.json
```

`CampaignLedger` persists a campaign as append-only JSONL.  The default policy
allows three completed, distinct candidate batteries per target, requires an
eight-turn battery, and makes promotion require at least a 0.5% median gain and
five paired wins. By default the exact object SHA is the de-duplication and
retry identity. When replay geometry or another runtime configuration changes
without changing the HSACO (for example tile32 versus tile64), record a
`configuration_sha256`/`--configuration-sha` over the canonical complete
candidate configuration. The object and configuration SHA pair becomes the
campaign identity while the source and object SHAs remain required provenance.
Exact duplicate identities are returned as
`RecordDisposition::DuplicateSkipped`; distinct configurations of the same
object and recompiled objects using the same configuration consume distinct
rounds. Infrastructure failures do not consume a GPU round and receive one
retry. Completed batteries require an accepted resource assessment plus
correctness and timing artifacts before they can be recorded or promoted. A
configured promotion supplies both `--object-sha` and the same
`--configuration-sha`, so equal objects with different geometry remain
unambiguous. Promotion advances the cumulative whole-product incumbent used by
the next target.

```rust
use radiowave::{
    ArchProfile, CampaignLedger, CampaignStarted, ResourceContract,
};

let ledger = CampaignLedger::create(
    "gfx1151-campaign.jsonl",
    CampaignStarted::new("gfx1151-tg128", ArchProfile::Gfx1151, baseline_sha),
)?;
let assessment = ResourceContract::new(ArchProfile::Gfx1151)
    .assess(&candidate_inspection, kernel_symbol, incumbent_kernel);
```

## Current safety boundary

- Radiowave accepts any architecture understood by the installed hipcc; callers
  must pass `--arch` (or `RADIOWAVE_ARCH`) rather than inheriting a gfx1201
  default.
- The injected buffer descriptor configuration is correctness-checked across
  gfx1010, gfx1030, gfx1100, gfx1151, and gfx1201. Only gfx1201 has performance
  promotion; every other target remains candidate-only until it wins its own
  benchmark. The [hipx portability artifact](tests/artifacts/hipx-portability-2026-07-14.json)
  records the live outputs and inspected code-object hashes.
- Buffer byte offsets are 32-bit. Allocations larger than 4 GiB must rebase the
  descriptor instead of allowing an offset to wrap.
- Radiowave does not silently rewrite arbitrary pointer accesses yet. Kernels
  opt into the helper where bounds, alignment, aliasing, and coherence are
  known.
- Cache certification is intentionally conservative and instruction-class
  based. It proves which shader read caches a same-agent consumer may use; it
  does not prove resource ownership or remove the required producer-completion
  wait.
- Inspection is evidence, not binary rewriting. Retuning happens by compiling
  another source/IR variant and correctness-gating it.

The next stages are typed buffer views, bounded source transformations, and
cache-policy variants. LLVM/hipcc remains the backend; Radiowave owns the
architecture-neutral policy and architecture-specific proof.
