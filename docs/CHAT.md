# `hipfire chat` - Interactive Chat TUI

## Usage

```bash
hipfire chat <model-tag>
```

Example:

```bash
hipfire chat qwen3.5:9b
```

`hipfire chat` reuses a running `hipfire serve` if one is detected on the
configured port. Otherwise it spawns a dedicated daemon and tears it down on
exit.

## Keybindings

| Key | Action |
|-----|--------|
| `Ctrl+O` | Insert newline for multi-line input |
| `Enter` | Submit message |
| `Ctrl+C` | Abort stream; press twice from idle to exit |
| `Ctrl+L` | Clear screen |
| `Ctrl+D` | Exit when input is empty |
| `Up` / `Down` | Navigate input history |
| `Left` / `Right` | Move cursor |
| `Home` / `End` | Jump to start or end of line |
| `Backspace` / `Delete` | Delete characters |

## Slash Commands

| Command | Description |
|---------|-------------|
| `/help`, `/?` | Show help |
| `/clear` | Clear conversation history |
| `/stats` | Show model stats |
| `/trim [pct]` | Drop oldest turns to free context; defaults to 50% |
| `/set <key> <val>` | Adjust `temperature`, `top_p`, `max_tokens`, or `repeat_penalty` for this session |
| `/exit`, `/quit` | Exit chat |

## Color Output

Colors and OSC 8 hyperlinks track the terminal palette using 16-color ANSI
styling. To disable styling:

| Trigger | Effect |
|---|---|
| `hipfire chat <tag> --no-color` | Disable styling for this session |
| `NO_COLOR=1 hipfire chat <tag>` | Honor the no-color.org convention |
| `CLICOLOR=0 hipfire chat <tag>` | Honor the common CLI color fallback |
