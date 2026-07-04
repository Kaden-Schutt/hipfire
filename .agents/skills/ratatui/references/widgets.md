# Ratatui Widgets

Built-in widgets for common UI patterns.

## Paragraph

```rust
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

let paragraph = Paragraph::new("Hello, world!")
    .block(Block::default().title("Title").borders(Borders::ALL))
    .wrap(Wrap { trim: true });

f.render_widget(paragraph, area);
```

## List

```rust
use ratatui::widgets::{Block, Borders, List, ListItem, ListState};
use ratatui::style::{Color, Style};

let items: Vec<ListItem> = app.items
    .iter()
    .map(|i| ListItem::new(i.as_str()))
    .collect();

let list = List::new(items)
    .block(Block::default().title("List").borders(Borders::ALL))
    .highlight_style(Style::default().bg(Color::Gray))
    .highlight_symbol("> ");

// Stateful rendering for selection
let mut state = ListState::default();
state.select(Some(app.selected));
f.render_stateful_widget(list, area, &mut state);
```

## Table

```rust
use ratatui::widgets::{Block, Borders, Table, Row, Cell};
use ratatui::layout::Constraint;

let header = Row::new(vec!["Name", "Age", "City"])
    .style(Style::default().fg(Color::Yellow));

let rows = vec![
    Row::new(vec!["Alice", "30", "NYC"]),
    Row::new(vec!["Bob", "25", "SF"]),
];

let table = Table::new(rows, [
    Constraint::Percentage(40),
    Constraint::Percentage(20),
    Constraint::Percentage(40),
])
.header(header)
.block(Block::default().title("Users").borders(Borders::ALL))
.highlight_style(Style::default().bg(Color::DarkGray));

f.render_widget(table, area);
```

## Gauge

```rust
use ratatui::widgets::{Block, Borders, Gauge};
use ratatui::style::{Color, Style};

let gauge = Gauge::default()
    .block(Block::default().title("Progress").borders(Borders::ALL))
    .gauge_style(Style::default().fg(Color::Green))
    .percent(app.progress);

f.render_widget(gauge, area);
```

## Tabs

```rust
use ratatui::widgets::{Block, Borders, Tabs};
use ratatui::text::Line;

let titles = vec!["Tab1", "Tab2", "Tab3"];
let tabs = Tabs::new(titles)
    .block(Block::default().title("Tabs").borders(Borders::ALL))
    .select(app.current_tab)
    .highlight_style(Style::default().fg(Color::Yellow));

f.render_widget(tabs, area);
```

## Block Styles

```rust
use ratatui::widgets::{Block, Borders, BorderType};
use ratatui::style::{Color, Style};

let block = Block::default()
    .title("Styled Block")
    .title_alignment(Alignment::Center)
    .borders(Borders::ALL)
    .border_type(BorderType::Rounded)
    .border_style(Style::default().fg(Color::Cyan));
```

## Scrollbar

```rust
use ratatui::widgets::{Scrollbar, ScrollbarOrientation, ScrollbarState};

let scrollbar = Scrollbar::new(ScrollbarOrientation::VerticalRight);
let mut scrollbar_state = ScrollbarState::new(total_items).position(current_position);

f.render_stateful_widget(scrollbar, area, &mut scrollbar_state);
```

## Sparkline

```rust
use ratatui::widgets::{Block, Sparkline};

let sparkline = Sparkline::default()
    .block(Block::default().title("Data"))
    .data(&app.data_points)
    .style(Style::default().fg(Color::Green));

f.render_widget(sparkline, area);
```
