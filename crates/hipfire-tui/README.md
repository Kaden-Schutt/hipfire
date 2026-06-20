# hipfire-tui

Prototype 1 of the Wick-shaped hipfire terminal home.

Longer-term direction: `hipfire-tui` should remain optional and act as one
operator client over daemon/server APIs. A future WebUI should use the same
model/server/config/eval/training-feedback state and action vocabulary. See
`../../docs/plans/2026-06-20-operator-ui-clients.md`.

Run from the repository root:

```bash
cargo run -p hipfire-tui
```

For the Chat tab to stream, start the Rust serve process in a separate terminal
first:

```bash
hipfire serve
```

Current scope:

- Ratatui shell with Home, Chat, Models, Settings, and System tabs.
- Reads real `~/.hipfire` config, per-model overlays, and local model files.
- Probes the existing `/health` endpoint and streams chat through
  `/v1/chat/completions`.
- Chat can ask the Rust CLI to start `serve` when the endpoint is offline; the
  typed prompt stays in place so you can retry after health comes online.
- The Models tab groups local model files by family.
  `Enter` expands/collapses a family, and `Enter` on a model selects it for
  this TUI session without mutating config.
- Settings are read-only in this spike. Tools such as quantizer, AWQ import,
  and TriAttention sidecar workflows are intentionally visible only as runway.

Out of scope for this spike:

- Replacing the `hipfire` command.
- Mutating config values from the TUI.
- Agent profiles, skills, plugins, or `/slash` generation.
- Pull/install progress UI.
