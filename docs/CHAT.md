# `hipfire chat` — Interactive Chat TUI

## Usage

```
hipfire chat <model-tag>
```

Example:
```
hipfire chat qwen3.5:9b
```

Requires a running `hipfire serve` or auto-spawns a dedicated daemon.

## Keybindings

| Key | Action |
|-----|--------|
| CTRL+O | Insert newline (multi-line input) |
| Enter | Submit message |
| CTRL+C | Abort stream (press twice to exit) |
| CTRL+L | Clear screen |
| CTRL+D | Exit (when input empty) |
| Up / Down | Navigate input history |
| Left / Right | Move cursor |
| Home / End | Jump to start / end of line |
| Backspace / Delete | Delete characters |

## Slash Commands

| Command | Description |
|---------|-------------|
| `/help`, `/?` | Show help |
| `/clear` | Clear conversation history |
| `/stats` | Show model stats |
| `/trim` | Drop old messages to free context |
| `/exit`, `/quit` | Exit chat |

## Features

- **Multi-line input:** Use CTRL+O to insert newlines. Bracketed paste works automatically.
- **Markdown rendering:** Code blocks, inline code, bold, and italic are rendered with ANSI styling.
- **Streaming:** See tokens appear in real-time. Abort with CTRL+C.
- **Input history:** Use up/down arrows to recall previous messages.
- **Context awareness:** Warning when approaching the model's context limit.

## Color output

Colors and OSC 8 hyperlinks track your terminal palette automatically (we
use 16-color ANSI throughout, no truecolor). To disable styling entirely:

| Trigger | Effect |
|---|---|
| `hipfire chat <tag> --no-color` | One-shot: this session only |
| `NO_COLOR=1 hipfire chat <tag>` | Honors the [no-color.org](https://no-color.org) standard |
| `CLICOLOR=0 hipfire chat <tag>` | De-facto-standard fallback also honored |

When disabled, all SGR (bold/dim/italic/color) and OSC 8 hyperlink
sequences are stripped at write-time — markdown still renders to
plain text but loses styling.

## Daemon Lifecycle

`hipfire chat` reuses a running `hipfire serve` if one is detected on the
configured port. Otherwise it spawns a dedicated daemon and tears it down on
exit.
