// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Chat message body rendering — the one fiddly bit, isolated and pure.
//!
//! Turns a message's text into styled ratatui lines, giving fenced ``` code
//! blocks a distinct background, dimming trailing comments, and tinting inline
//! `code` spans in prose. Deliberately light (no per-language grammar): a code
//! TUI wants legible structure, not a full highlighter. Pure + color-injected so
//! it is unit-testable without a terminal.

use ratatui::{
    style::{Color, Modifier, Style},
    text::{Line, Span},
};

/// Colors injected by the caller (from the UI theme) so this module stays pure.
#[derive(Clone, Copy)]
pub struct CodeTheme {
    pub text: Color,
    pub code_fg: Color,
    pub code_bg: Color,
    pub comment: Color,
    pub fence: Color,
}

/// Render a message body into styled lines. Fenced code blocks (delimited by a
/// line whose first non-space chars are ```` ``` ````) get the code background;
/// inside them, a trailing `//` or `#` comment is dimmed. Outside code, inline
/// `` `code` `` spans are tinted.
pub fn render_body(content: &str, theme: &CodeTheme) -> Vec<Line<'static>> {
    let mut out = Vec::new();
    let mut in_code = false;
    for raw in content.lines() {
        if raw.trim_start().starts_with("```") {
            in_code = !in_code;
            out.push(fence_line(raw, in_code, theme));
        } else if in_code {
            out.push(code_line(raw, theme));
        } else {
            out.push(prose_line(raw, theme));
        }
    }
    out
}

fn fence_line(raw: &str, opening: bool, theme: &CodeTheme) -> Line<'static> {
    let lang = raw.trim_start().trim_start_matches('`').trim();
    let label = if opening && !lang.is_empty() {
        format!("``` {lang}")
    } else {
        "```".to_string()
    };
    Line::from(Span::styled(
        label,
        Style::default().fg(theme.fence).add_modifier(Modifier::DIM),
    ))
}

fn code_line(raw: &str, theme: &CodeTheme) -> Line<'static> {
    let code = Style::default().fg(theme.code_fg).bg(theme.code_bg);
    match find_comment(raw) {
        Some(idx) => Line::from(vec![
            Span::styled(raw[..idx].to_string(), code),
            Span::styled(
                raw[idx..].to_string(),
                Style::default().fg(theme.comment).bg(theme.code_bg),
            ),
        ]),
        None => Line::from(Span::styled(raw.to_string(), code)),
    }
}

fn prose_line(raw: &str, theme: &CodeTheme) -> Line<'static> {
    let text = Style::default().fg(theme.text);
    if !raw.contains('`') {
        return Line::from(Span::styled(raw.to_string(), text));
    }
    let code = Style::default().fg(theme.code_fg).bg(theme.code_bg);
    // Only PAIRED backticks form an inline code span; an unmatched backtick is
    // rendered literally so the transcript isn't misrepresented.
    let chars: Vec<char> = raw.chars().collect();
    let mut spans = Vec::new();
    let mut buf = String::new();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '`' {
            if let Some(off) = chars[i + 1..].iter().position(|&c| c == '`') {
                if !buf.is_empty() {
                    spans.push(Span::styled(std::mem::take(&mut buf), text));
                }
                let inner: String = chars[i + 1..i + 1 + off].iter().collect();
                spans.push(Span::styled(inner, code));
                i = i + 1 + off + 1; // skip past the closing backtick
                continue;
            }
            buf.push('`'); // unmatched — literal
        } else {
            buf.push(chars[i]);
        }
        i += 1;
    }
    if !buf.is_empty() {
        spans.push(Span::styled(buf, text));
    }
    if spans.is_empty() {
        spans.push(Span::styled(String::new(), text));
    }
    Line::from(spans)
}

/// Byte index of a line-trailing `//` or `#` comment, skipping matches inside
/// double-quoted strings. None if the line has no comment. Deliberately
/// grammar-free: only `"` opens a string (so lifetimes / apostrophes in code
/// aren't mistaken for char literals), with backslash-parity escape handling.
fn find_comment(line: &str) -> Option<usize> {
    let bytes = line.as_bytes();
    let mut in_str = false;
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if in_str {
            if c == b'"' && even_preceding_backslashes(bytes, i) {
                in_str = false;
            }
        } else if c == b'"' {
            in_str = true;
        } else if c == b'#' {
            return Some(i);
        } else if c == b'/' && bytes.get(i + 1) == Some(&b'/') {
            return Some(i);
        }
        i += 1;
    }
    None
}

/// Whether the `"` at byte `i` is unescaped — i.e. preceded by an even number of
/// backslashes (`\\"` closes the string; `\"` does not).
fn even_preceding_backslashes(bytes: &[u8], i: usize) -> bool {
    let mut count = 0;
    let mut k = i;
    while k > 0 && bytes[k - 1] == b'\\' {
        count += 1;
        k -= 1;
    }
    count % 2 == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn theme() -> CodeTheme {
        CodeTheme {
            text: Color::White,
            code_fg: Color::Cyan,
            code_bg: Color::Black,
            comment: Color::DarkGray,
            fence: Color::Blue,
        }
    }

    #[test]
    fn code_block_lines_get_code_background() {
        let body = "before\n```rust\nlet x = 1;\n```\nafter";
        let lines = render_body(body, &theme());
        // before(0) fence(1) code(2) fence(3) after(4)
        assert_eq!(lines.len(), 5);
        let prose_bg = lines[0].spans[0].style.bg;
        let code_bg = lines[2].spans[0].style.bg;
        assert_eq!(code_bg, Some(Color::Black), "code line carries the code bg");
        assert_ne!(prose_bg, Some(Color::Black), "prose line does not");
    }

    #[test]
    fn trailing_comment_is_dimmed_but_hash_in_string_is_not() {
        let theme = theme();
        // `let s = "a#b"; // note` -> code span + comment span split at '//'
        let lines = render_body("```\nlet s = \"a#b\"; // note\n```", &theme);
        let code = &lines[1];
        assert_eq!(code.spans.len(), 2, "split into code + comment");
        assert!(code.spans[1].content.contains("// note"));
        assert_eq!(code.spans[1].style.fg, Some(Color::DarkGray));
        // The '#' inside the string must NOT have started a comment.
        assert!(code.spans[0].content.contains("a#b"));
    }

    #[test]
    fn inline_code_span_is_tinted_in_prose() {
        let lines = render_body("use the `foo` function", &theme());
        assert_eq!(lines.len(), 1);
        let spans = &lines[0].spans;
        let foo = spans
            .iter()
            .find(|s| s.content == "foo")
            .expect("inline code span");
        assert_eq!(foo.style.bg, Some(Color::Black));
    }

    #[test]
    fn find_comment_ignores_strings() {
        assert_eq!(find_comment("x = 1 // c"), Some(6));
        assert_eq!(find_comment("x = 1 # c"), Some(6));
        assert_eq!(find_comment("s = \"# not a comment\""), None);
        assert_eq!(find_comment("plain code"), None);
        // Lifetimes / apostrophes are NOT strings (only `"` opens one).
        assert_eq!(find_comment("let x: &'a str = y; // c"), Some(20));
        // Escaped quote inside a string doesn't close it early (backslash parity).
        assert_eq!(find_comment(r#"s = "a\"b"; // c"#), Some(12));
    }

    #[test]
    fn code_line_with_multibyte_before_comment_does_not_panic() {
        // The byte index from find_comment must land on a char boundary (# is
        // ASCII) even when earlier chars are multibyte.
        let lines = render_body("```\ncafé = 1 # x\n```", &theme());
        let code = &lines[1];
        assert_eq!(code.spans.len(), 2);
        assert!(code.spans[0].content.contains("café = 1"));
        assert!(code.spans[1].content.contains("# x"));
    }

    #[test]
    fn unbalanced_fence_keeps_trailing_lines_as_code() {
        // A code block that is never closed: every following line stays code.
        let lines = render_body("intro\n```\ncode one\ncode two", &theme());
        assert_ne!(
            lines[0].spans[0].style.bg,
            Some(Color::Black),
            "intro is prose"
        );
        assert_eq!(lines[2].spans[0].style.bg, Some(Color::Black), "code one");
        assert_eq!(lines[3].spans[0].style.bg, Some(Color::Black), "code two");
    }

    #[test]
    fn odd_inline_backtick_is_kept_literal() {
        // An unmatched backtick is rendered, not swallowed.
        let lines = render_body("a `b c", &theme());
        let joined: String = lines[0].spans.iter().map(|s| s.content.as_ref()).collect();
        assert_eq!(joined, "a `b c");
    }
}
