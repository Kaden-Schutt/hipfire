---
name: ratatui
description: Build terminal UIs in Rust with Ratatui. Use when creating TUI applications, immediate-mode rendering, high-performance terminal interfaces, or production Rust CLIs.
---

# Ratatui (Rust TUI)

Immediate-mode terminal UI framework for Rust using Crossterm backend.

## Dependencies

```toml
[dependencies]
ratatui = "0.28"
crossterm = "0.28"
```

## Basic Application

```rust
use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, widgets::{Block, Borders, Paragraph}, Terminal};
use std::io;

struct App {
    counter: i32,
}

impl App {
    fn new() -> App {
        App { counter: 0 }
    }

    fn on_key(&mut self, key: KeyCode) {
        match key {
            KeyCode::Up => self.counter += 1,
            KeyCode::Down => self.counter -= 1,
            _ => {}
        }
    }
}

fn main() -> Result<(), io::Error> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;

    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = App::new();

    loop {
        terminal.draw(|f| {
            let block = Block::default().title("Counter").borders(Borders::ALL);
            let paragraph = Paragraph::new(format!("Count: {}", app.counter)).block(block);
            f.render_widget(paragraph, f.area());
        })?;

        if let Event::Key(key) = event::read()? {
            match key.code {
                KeyCode::Char('q') => break,
                code => app.on_key(code),
            }
        }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    Ok(())
}
```

## Event Loop with Polling

```rust
use std::time::Duration;

fn run_app<B: Backend>(terminal: &mut Terminal<B>, app: &mut App) -> io::Result<()> {
    loop {
        terminal.draw(|f| ui(f, app))?;

        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('q') => return Ok(()),
                    KeyCode::Up => app.increment(),
                    KeyCode::Down => app.decrement(),
                    _ => {}
                }
            }
        }
    }
}
```

## Layout

```rust
use ratatui::layout::{Constraint, Direction, Layout};

let chunks = Layout::default()
    .direction(Direction::Vertical)
    .constraints([
        Constraint::Length(3),  // Fixed height
        Constraint::Min(0),     // Fill remaining
        Constraint::Length(1),  // Status bar
    ])
    .split(f.area());
```

## References

- **Widgets**: See [references/widgets.md](references/widgets.md) for List, Table, Paragraph, Gauge
- **Best Practices**: See [references/best-practices.md](references/best-practices.md) for keyboard, accessibility, performance
