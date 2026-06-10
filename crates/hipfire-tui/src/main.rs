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
                app.dismiss_text_input();
            } else {
                return true;
            }
        }
        KeyCode::Tab => app.next_tab(),
        KeyCode::BackTab => app.prev_tab(),
        KeyCode::Char('r') if !app.text_input_active() => app.reload(),
        KeyCode::Char('e') if app.tab == app::Tab::Settings && !app.text_input_active() => {
            app.settings_easy = true;
            app.settings_selected = 0;
        }
        KeyCode::Char('a') if app.tab == app::Tab::Settings && !app.text_input_active() => {
            app.settings_easy = false;
            app.settings_selected = 0;
        }
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
    fn settings_edit_captures_keys_and_esc_cancels() {
        let mut app = App::test_app();
        app.tab = Tab::Settings;
        app.settings_easy = false;
        app.config.values.insert("kv_cache".into(), "auto".into());
        app.settings_selected = 0;
        // Enter begins the edit prefilled with the current value
        assert!(!handle_key(&mut app, key(KeyCode::Enter)));
        assert!(app.text_input_active());
        assert_eq!(app.settings_edit.as_ref().unwrap().key, "kv_cache");
        assert_eq!(app.settings_edit.as_ref().unwrap().buffer, "auto");
        // q now types into the buffer instead of quitting
        assert!(!handle_key(&mut app, key(KeyCode::Char('q'))));
        assert_eq!(app.settings_edit.as_ref().unwrap().buffer, "autoq");
        // Esc cancels without applying, second Esc quits
        assert!(!handle_key(&mut app, key(KeyCode::Esc)));
        assert!(app.settings_edit.is_none());
        assert!(handle_key(&mut app, key(KeyCode::Esc)));
    }

    #[test]
    fn settings_scope_toggle_targets_active_model() {
        let mut app = App::test_app();
        app.tab = Tab::Settings;
        app.active_model = "qwen3.5:9b".into();
        assert!(!app.settings_scope_model);
        assert!(!handle_key(&mut app, key(KeyCode::Char('m'))));
        assert!(app.settings_scope_model);
        assert!(app.last_reload.contains("qwen3.5:9b"));
        assert!(!handle_key(&mut app, key(KeyCode::Char('m'))));
        assert!(!app.settings_scope_model);
    }

    #[test]
    fn informational_easy_row_is_not_editable() {
        let mut app = App::test_app();
        app.tab = Tab::Settings;
        app.settings_easy = true;
        // last easy row is the Serve endpoint (key: None)
        app.settings_selected = app.config.easy_rows().len() - 1;
        assert!(!handle_key(&mut app, key(KeyCode::Enter)));
        assert!(app.settings_edit.is_none());
    }

    #[test]
    fn esc_quits_directly_on_non_chat_tabs() {
        let mut app = App::test_app();
        app.tab = Tab::Models;
        assert!(handle_key(&mut app, key(KeyCode::Esc)));
    }
}
