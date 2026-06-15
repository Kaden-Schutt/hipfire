// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

mod app;
mod hipfire;
mod ui;

use std::{io, panic};

use anyhow::Result;
use app::App;
use crossterm::{
    event::{self, Event, KeyCode, KeyEvent, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};

fn main() -> Result<()> {
    let mut terminal = setup_terminal()?;
    let result = run(&mut terminal);
    restore_terminal(&mut terminal)?;
    result
}

fn setup_terminal() -> Result<Terminal<CrosstermBackend<io::Stdout>>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;

    let hook = panic::take_hook();
    panic::set_hook(Box::new(move |info| {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen);
        hook(info);
    }));

    let backend = CrosstermBackend::new(stdout);
    Ok(Terminal::new(backend)?)
}

fn restore_terminal(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> Result<()> {
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    Ok(())
}

fn run(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> Result<()> {
    let mut app = App::load()?;

    loop {
        terminal.draw(|frame| ui::draw(frame, &mut app))?;
        app.drain_chat_events();

        if event::poll(std::time::Duration::from_millis(80))? {
            match event::read()? {
                Event::Key(key) => {
                    if handle_key(&mut app, key) {
                        break;
                    }
                }
                Event::Resize(_, _) => {}
                _ => {}
            }
        }
    }

    Ok(())
}

fn handle_key(app: &mut App, key: KeyEvent) -> bool {
    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
        if app.chat.sending {
            app.chat.status =
                "stream is still running; wait for this spike build to finish it".into();
            return false;
        }
        return true;
    }

    match key.code {
        KeyCode::Char('q') if !app.chat.is_input_focused() => return true,
        KeyCode::Esc => {
            if app.chat.sending {
                app.chat.status = "stream abort is not wired in prototype 1".into();
            } else if app.chat.is_input_focused() {
                app.chat.blur_input();
            } else {
                return true;
            }
        }
        KeyCode::Tab => app.next_tab(),
        KeyCode::BackTab => app.prev_tab(),
        KeyCode::Char('r') if !app.chat.is_input_focused() => app.reload(),
        KeyCode::Char('e') if app.tab == app::Tab::Settings => app.settings_easy = true,
        KeyCode::Char('a') if app.tab == app::Tab::Settings => app.settings_easy = false,
        _ => app.handle_tab_key(key),
    }

    false
}
