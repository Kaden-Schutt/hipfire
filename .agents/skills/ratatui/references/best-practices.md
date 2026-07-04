# TUI Best Practices

Design guidelines for terminal user interfaces.

## Keyboard Shortcuts

Use consistent, discoverable shortcuts:

| Key | Action |
|-----|--------|
| `q` / `Ctrl+C` | Quit |
| `h/j/k/l` or arrows | Navigate |
| `Enter` | Select/Confirm |
| `Esc` | Cancel/Back |
| `/` | Search |
| `?` | Help |
| `Tab` | Next field |
| `Shift+Tab` | Previous field |
| `g` / `G` | Top/Bottom |

## Application Structure

```rust
struct App {
    // State
    items: Vec<Item>,
    selected: usize,
    mode: Mode,

    // Flags
    should_quit: bool,
}

enum Mode {
    Normal,
    Insert,
    Search,
}

impl App {
    fn handle_key(&mut self, key: KeyCode) {
        match self.mode {
            Mode::Normal => self.handle_normal_key(key),
            Mode::Insert => self.handle_insert_key(key),
            Mode::Search => self.handle_search_key(key),
        }
    }
}
```

## Performance

- **Limit redraws**: Only redraw when state changes
- **Use polling**: `event::poll(Duration::from_millis(100))` instead of blocking
- **Batch updates**: Terminal uses double buffering automatically
- **Cache content**: Precompute expensive renders

```rust
// Good: poll with timeout
if event::poll(Duration::from_millis(100))? {
    if let Event::Key(key) = event::read()? {
        app.handle_key(key.code);
    }
}

// Bad: blocking read (can't update async state)
if let Event::Key(key) = event::read()? {
    // ...
}
```

## Async Integration

```rust
use tokio::sync::mpsc;

enum AppEvent {
    Key(KeyCode),
    Tick,
    Data(Vec<Item>),
}

async fn run_app(mut rx: mpsc::Receiver<AppEvent>) {
    loop {
        terminal.draw(|f| ui(f, &app))?;

        if let Some(event) = rx.recv().await {
            match event {
                AppEvent::Key(KeyCode::Char('q')) => break,
                AppEvent::Key(code) => app.handle_key(code),
                AppEvent::Tick => app.tick(),
                AppEvent::Data(items) => app.items = items,
            }
        }
    }
}
```

## Error Handling

```rust
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Setup
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Run with panic recovery
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_app(&mut terminal)
    }));

    // Always cleanup
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;

    // Re-raise panic after cleanup
    match result {
        Ok(Ok(())) => Ok(()),
        Ok(Err(e)) => Err(e.into()),
        Err(panic) => std::panic::resume_unwind(panic),
    }
}
```

## Accessibility

- High contrast colors (avoid light gray on white)
- Keyboard-only navigation (never require mouse)
- Clear focus indicators (highlight selected item)
- Don't rely on color alone (use symbols: ✓, ✗, ▶)
- Support terminal themes (use semantic colors)

## Responsive Layout

```rust
fn ui(f: &mut Frame, app: &App) {
    let area = f.area();

    // Adapt layout based on terminal size
    if area.width >= 80 {
        // Wide: side-by-side
        let chunks = Layout::horizontal([
            Constraint::Percentage(30),
            Constraint::Percentage(70),
        ]).split(area);
        render_sidebar(f, chunks[0], app);
        render_main(f, chunks[1], app);
    } else {
        // Narrow: stacked
        let chunks = Layout::vertical([
            Constraint::Length(10),
            Constraint::Min(0),
        ]).split(area);
        render_sidebar(f, chunks[0], app);
        render_main(f, chunks[1], app);
    }
}
```

## Status Bar

```rust
fn render_status_bar(f: &mut Frame, area: Rect, app: &App) {
    let status = format!(
        " {} | {} items | Press ? for help",
        app.mode,
        app.items.len()
    );

    let paragraph = Paragraph::new(status)
        .style(Style::default().bg(Color::Blue).fg(Color::White));

    f.render_widget(paragraph, area);
}
```
