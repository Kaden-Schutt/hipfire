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
        KeyCode::Char('q') if !app.text_input_active() => return true,
        KeyCode::Esc => {
            if app.chat.sending {
                app.chat.status = "stream abort is not wired in prototype 1".into();
            } else if app.text_input_active() {
                app.chat.blur_input();
            } else {
                return true;
            }
        }
        KeyCode::Tab => app.next_tab(),
        KeyCode::BackTab => app.prev_tab(),
        KeyCode::Char('r') if !app.text_input_active() => app.reload(),
        KeyCode::Char('e') if app.tab == app::Tab::Settings => app.settings_easy = true,
        KeyCode::Char('a') if app.tab == app::Tab::Settings => app.settings_easy = false,
        _ => app.handle_tab_key(key),
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use app::Tab;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    #[test]
    fn q_quits_from_launch_on_every_non_chat_tab() {
        for tab in [Tab::Home, Tab::Models, Tab::Settings, Tab::System] {
            let mut app = App::test_app();
            app.tab = tab;
            assert!(
                app.chat.is_input_focused(),
                "chat input starts focused by default"
            );
            assert!(
                handle_key(&mut app, key(KeyCode::Char('q'))),
                "q must quit on {tab:?} even though chat input is focused"
            );
        }
    }

    #[test]
    fn q_and_r_type_into_focused_chat_input() {
        let mut app = App::test_app();
        app.tab = Tab::Chat;
        assert!(!handle_key(&mut app, key(KeyCode::Char('q'))));
        assert!(!handle_key(&mut app, key(KeyCode::Char('r'))));
        assert_eq!(app.chat.input, "qr");
    }

    #[test]
    fn esc_blurs_focused_chat_then_quits() {
        let mut app = App::test_app();
        app.tab = Tab::Chat;
        assert!(!handle_key(&mut app, key(KeyCode::Esc)));
        assert!(!app.chat.is_input_focused());
        assert!(handle_key(&mut app, key(KeyCode::Esc)));
    }

    #[test]
    fn blurred_chat_is_navigation_mode() {
        let mut app = App::test_app();
        app.tab = Tab::Chat;
        app.chat.blur_input();
        // characters do not leak into the input buffer
        let mut probe = App::test_app();
        probe.tab = Tab::Chat;
        probe.chat.blur_input();
        assert!(!handle_key(&mut probe, key(KeyCode::Char('x'))));
        assert_eq!(probe.chat.input, "");
        // q quits, i refocuses
        assert!(handle_key(&mut app, key(KeyCode::Char('q'))));
        assert!(!handle_key(&mut probe, key(KeyCode::Char('i'))));
        assert!(probe.chat.is_input_focused());
    }

    #[test]
    fn esc_quits_directly_on_non_chat_tabs() {
        let mut app = App::test_app();
        app.tab = Tab::Models;
        assert!(handle_key(&mut app, key(KeyCode::Esc)));
    }
}
