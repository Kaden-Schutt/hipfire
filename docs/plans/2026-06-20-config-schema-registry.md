# Plan: config schema registry and layered overrides

Status: **proposed** - 2026-06-20.

## Goal

Let the code that owns a config need declare that need near the implementation,
while still producing one global, machine-readable config schema for CLI, daemon
operator APIs, TUI, WebUI, docs, validation, and config drift diagnostics.

The target is similar in spirit to `clap-markdown`: declarations live in Rust,
and a generator emits JSON/TOML/Markdown views of every known config option.

Example declaration shape:

```rust
config_field! {
    key: "vision.max_cores",
    ty: u8,
    required: when("vision.enabled"),
    scope: [runtime, host, model],
    owner: "hipfire-arch-qwen35-vl",
    description: "Maximum number of CPU cores to allocate to the vision encoder.",
}
```

## Requirements

- Field declarations can live near the crate/module that consumes them.
- The assembled schema is deterministic and exportable as JSON, TOML, and
  Markdown.
- Required fields support conditions, not just a boolean.
- Values can be overridden by layer: global defaults, host/node profiles,
  pool policy, per-model policy, environment variables, CLI flags, and request
  scoped overrides.
- The schema captures enough metadata for safe UI editing: type, default,
  allowed values, requirement, mutability, scope, owner, description, validation,
  secrecy, and restart/reload impact.
- The runtime can explain a resolved value: final value plus which layer supplied
  it.

## Schema Model

Each field should carry:

- `key`: stable dotted name, for example `vision.max_cores`;
- `ty`: `bool`, integer widths, float, string, path, enum, list, object, or
  structured custom type;
- `requirement`: optional, required, or required when a predicate is true;
- `default`: optional literal or computed default descriptor;
- `scopes`: one or more of global, host, node, pool, model, runtime, eval,
  training, request;
- `mutability`: static, load-time, runtime-reloadable, request-only;
- `owner`: crate/module that declares the field;
- `description`: one-line user-facing help text;
- `validation`: range, enum set, regex/path policy, or custom validator name;
- `secret`: whether UI/docs should redact the value;
- `restart_impact`: none, reload model, restart daemon, restart service, or
  reconnect clients.

Required predicates should be data, not prose:

```text
required = true
required_when = "vision.enabled == true"
required_when = "model.family in ['qwen3.5-vl', 'gemma3-vl']"
required_when = "pool.mode == 'distributed'"
```

## Declaration Strategy

Start with explicit per-crate schema functions and a small macro:

```rust
pub fn config_schema() -> &'static [ConfigField] {
    &[VISION_MAX_CORES]
}
```

The top-level schema aggregator imports known crate schema functions and sorts by
key before export. This keeps the first implementation simple and auditable.

If the explicit aggregator becomes painful, move to distributed registration
with `linkme` or `inventory`, but keep the exported schema format unchanged.

## Override Layers

Config resolution should be layered and explainable. Proposed precedence, low to
high:

1. compiled default from the schema;
2. global base config, for example `~/.hipfire/config.json`;
3. named profile, for example `profile=workstation` or `profile=bench`;
4. host config, keyed by stable host identity;
5. node config, keyed by daemon/node identity in a future local-network pool;
6. pool policy, selected by scheduler or operator;
7. per-model override;
8. per-model-per-host override, only when needed for heterogeneous fleets;
9. environment variables;
10. CLI flags;
11. request-scoped overrides.

Every resolved config value should retain provenance:

```json
{
  "key": "vision.max_cores",
  "value": 6,
  "source": "host",
  "source_id": "strix-halo-01",
  "overrode": ["default", "global", "model"]
}
```

This provenance is what the TUI/WebUI should display instead of guessing whether
a value is defaulted or overridden.

## Host, Node, and Pool Scope

Dynamic local-network pooling needs config that is not purely global or
per-model:

- **Host profile:** machine-local facts and preferences, such as CPU core budget,
  ROCm path, NPU visibility, default resource-lock wait, log location, and
  service-management policy.
- **Node profile:** daemon-advertised runtime identity, GPUs/NPUs, VRAM, arch,
  health, active leases, and capabilities.
- **Pool policy:** scheduler preferences across nodes, such as placement,
  max concurrent loads, replication limits, preferred quant format, failover,
  and admission threshold.
- **Model policy:** model-specific defaults and restrictions, such as max context,
  KV mode, sidecars, draft compatibility, and arch-specific constraints.

The schema should describe which layers may set a field. For example,
`vision.max_cores` may be set globally, per host, or per model, but a request
override might be rejected if changing it would invalidate an already-loaded
vision encoder.

## Generated Artifacts

Add a maintenance command analogous to CLI doc generation:

```bash
cargo run -p hipfire-cli -- gen-config-schema --format json
cargo run -p hipfire-cli -- gen-config-schema --format toml
cargo run -p hipfire-cli -- gen-config-schema --format markdown
```

Candidate committed outputs:

- `docs/config-schema.md` for humans;
- `docs/config-schema.json` or `docs/config-schema.toml` for tools;
- daemon `/operator/config/schema` response for TUI/WebUI.

## Open Questions

- Whether env-var declarations should be first-class schema fields or a separate
  compatibility layer mapped onto schema keys.
- Whether per-model overrides should stay embedded in `config.json` or move to a
  separate sparse file once host/node/pool scopes exist.
- How to name stable host and node identities when the same machine can run
  several daemons or expose several GPU partitions.
- Whether distributed schema registration is worth the dependency and link-time
  behavior before plugin-like crates exist.
