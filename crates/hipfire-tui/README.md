# hipfire-tui

Prototype 1 of the Wick-shaped hipfire terminal home.

Run from the repository root:

```bash
cargo run -p hipfire-tui
```

For the Chat tab to stream, start the native Rust serve process in a separate
terminal first:

```bash
hipfire serve
```

Current scope:

- Ratatui shell with Home, Chat, Models, Settings, and System tabs.
- Reads real `~/.hipfire` config, per-model overlays, local model files, and
  the bundled `registry/v1.json`.
- Probes the existing `/health` endpoint and streams chat through
  `/v1/chat/completions`.
- Chat can ask the native CLI to start `serve -d` when the endpoint is
  offline; the typed prompt stays in place so you can retry after health comes
  online.
- The Models tab groups registry entries and local-only model files by family.
  `Enter` expands/collapses a family, and `Enter` on a model selects it for
  this TUI session without mutating config.
- Settings edit and persist global typed TOML values. Per-model overlays remain
  available through `hipfire config <tag> set ...`. Tools such as quantizer,
  AWQ import, and TriAttention sidecar workflows are intentionally outside the
  default TUI path.

Out of scope for this spike:

- Agent profiles, skills, plugins, or `/slash` generation.
- Pull/install progress UI.
