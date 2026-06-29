//! hipfire browser chat console — Leptos CSR app.
//!
//! Browser-heavy by design: streaming chat over the OpenAI-compatible API,
//! drag-drop image attachments, a settings drawer for sampler overrides +
//! model selection, markdown rendering with code blocks, cancellable
//! generation, live token-usage, and a collapsible reasoning panel.

use std::rc::Rc;

use gloo_file::{futures::read_as_data_url, File};
use hipfire_web_ui::{get_json, post_json, sse_post};
use leptos::prelude::*;
use serde::Deserialize;
use serde_json::{json, Value};
use wasm_bindgen::JsCast;
use web_sys::{AbortController, DragEvent, HtmlTextAreaElement, KeyboardEvent, SubmitEvent};

#[derive(Clone, Debug, PartialEq)]
struct UiMessage {
    role: String,
    content: String,
    /// Streamed `reasoning_content`, shown in a collapsible panel.
    reasoning: String,
    images: Vec<AttachedImage>,
}

impl UiMessage {
    fn new(role: &str) -> Self {
        Self {
            role: role.to_string(),
            content: String::new(),
            reasoning: String::new(),
            images: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct AttachedImage {
    name: String,
    data_url: String,
}

/// Token usage reported by the final stream event (`include_usage`).
#[derive(Clone, Copy, Debug, PartialEq, Default)]
struct Usage {
    prompt: u32,
    completion: u32,
}

/// Which server API the UI drives. Both are OpenAI-compatible and stream
/// `data:` JSON, but differ in request body and response shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ApiKind {
    /// `/v1/chat/completions` — supports image inputs.
    Chat,
    /// `/v1/responses` — text-only in hipfire today (images are dropped).
    Responses,
}

impl ApiKind {
    fn as_str(self) -> &'static str {
        match self {
            ApiKind::Chat => "chat",
            ApiKind::Responses => "responses",
        }
    }
    fn parse(s: &str) -> Self {
        if s == "responses" {
            ApiKind::Responses
        } else {
            ApiKind::Chat
        }
    }
    fn url(self) -> &'static str {
        match self {
            ApiKind::Chat => "/v1/chat/completions",
            ApiKind::Responses => "/v1/responses",
        }
    }
}

/// User-overridable request settings (the settings drawer).
#[derive(Clone, Debug, PartialEq)]
struct Settings {
    api: ApiKind,
    model: String,
    max_tokens: String,
    temperature: String,
    top_p: String,
    repeat_penalty: String,
    presence_penalty: String,
    frequency_penalty: String,
    reasoning_effort: String,
    stream: bool,
    system: String,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            api: ApiKind::Chat,
            model: String::new(),
            max_tokens: "512".to_string(),
            temperature: "0.7".to_string(),
            top_p: String::new(),
            repeat_penalty: String::new(),
            presence_penalty: String::new(),
            frequency_penalty: String::new(),
            reasoning_effort: String::new(),
            stream: true,
            system: String::new(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct ModelsEnvelope {
    data: Vec<ModelItem>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
struct ModelItem {
    id: String,
    #[serde(
        default,
        alias = "context_length",
        alias = "max_context",
        alias = "max_seq"
    )]
    context: Option<u32>,
}

fn main() {
    console_error_panic_hook::set_once();
    leptos::mount::mount_to_body(App);
}

#[component]
fn App() -> impl IntoView {
    let (messages, set_messages) = signal(Vec::<UiMessage>::new());
    let (models, set_models) = signal(Vec::<ModelItem>::new());
    let (settings, set_settings) = signal(Settings::default());
    let (prompt, set_prompt) = signal(String::new());
    let (attachments, set_attachments) = signal(Vec::<AttachedImage>::new());
    let (drag_active, set_drag_active) = signal(false);
    let (busy, set_busy) = signal(false);
    let (status, set_status) = signal("Ready".to_string());
    let (settings_open, set_settings_open) = signal(false);
    let (usage, set_usage) = signal(None::<Usage>);
    // Responses API server-side conversation state: the last response id, sent
    // as previous_response_id so the server reconstructs prior context.
    let (last_response_id, set_last_response_id) = signal(None::<String>);
    // Running server build identity (from /health), shown in the settings footer.
    let (server_version, set_server_version) = signal(None::<String>);
    // In-flight abort handle, so the Stop button can cancel a stream.
    let abort = StoredValue::new(None::<AbortController>);
    let messages_ref = NodeRef::<leptos::html::Main>::new();

    leptos::task::spawn_local(async move {
        if let Ok(items) = fetch_models().await {
            if settings.get_untracked().model.is_empty() && items.len() == 1 {
                set_settings.update(|s| s.model = items[0].id.clone());
            }
            set_models.set(items);
        }
    });

    leptos::task::spawn_local(async move {
        if let Ok(health) = get_json::<Value>("/health").await {
            if let Some(version) = health["version"].as_str() {
                set_server_version.set(Some(version.to_string()));
            }
        }
    });

    // Keep the transcript pinned to the newest content as it streams in.
    Effect::new(move |_| {
        messages.track();
        if let Some(el) = messages_ref.get() {
            el.set_scroll_top(el.scroll_height());
        }
    });

    let attach_files = Rc::new(move |files: Vec<web_sys::File>| {
        if busy.get_untracked() || files.is_empty() {
            return;
        }
        set_status.set("Reading images".to_string());
        leptos::task::spawn_local(async move {
            let mut accepted = Vec::new();
            let mut rejected = 0usize;
            for file in files {
                match read_attached_image(file).await {
                    Ok(image) => accepted.push(image),
                    Err(_) => rejected += 1,
                }
            }
            let added = accepted.len();
            if added > 0 {
                set_attachments.update(|items| items.extend(accepted));
            }
            set_status.set(match (added, rejected) {
                (0, 0) => "Ready".to_string(),
                (0, _) => "Drop PNG, JPEG, or WebP images".to_string(),
                (n, 0) => format!("Attached {n} image(s)"),
                (n, r) => format!("Attached {n}; skipped {r} unsupported"),
            });
        });
    });

    let send_action = Rc::new(move || {
        if busy.get_untracked() {
            return;
        }
        let text = prompt.get_untracked().trim().to_string();
        let image_attachments = attachments.get_untracked();
        if text.is_empty() && image_attachments.is_empty() {
            return;
        }

        let mut user_message = UiMessage::new("user");
        user_message.content = text;
        user_message.images = image_attachments;

        let mut request_messages = messages.get_untracked();
        request_messages.push(user_message.clone());
        let assistant_index = request_messages.len();

        set_messages.update(move |items| {
            items.push(user_message);
            items.push(UiMessage::new("assistant"));
        });
        set_prompt.set(String::new());
        set_attachments.set(Vec::new());
        set_usage.set(None);
        set_busy.set(true);
        set_status.set("Generating".to_string());

        let cfg = settings.get_untracked();
        // Responses API chains via previous_response_id; chat replays history.
        let prev = if cfg.api == ApiKind::Responses {
            last_response_id.get_untracked()
        } else {
            None
        };
        let (url, body) = build_request(&request_messages, &cfg, prev.as_deref());
        let use_stream = cfg.stream;
        let api = cfg.api;

        leptos::task::spawn_local(async move {
            let result = if use_stream {
                let controller = AbortController::new().ok();
                abort.set_value(controller.clone());
                stream_completion(url, body, controller, set_messages, assistant_index, set_status, set_usage, set_last_response_id).await
            } else {
                fetch_completion(url, body, api, set_usage, set_last_response_id).await.map(|content| {
                    set_messages.update(|items| {
                        if let Some(message) = items.get_mut(assistant_index) {
                            message.content = content;
                        }
                    });
                })
            };

            match result {
                Ok(()) => set_status.set("Ready".to_string()),
                Err(error) => {
                    let aborted = abort
                        .get_value()
                        .map(|c| c.signal().aborted())
                        .unwrap_or(false);
                    if aborted {
                        set_status.set("Stopped".to_string());
                    } else {
                        // Drop stale conversation state so the next turn restarts
                        // fresh rather than chaining off a failed/expired id.
                        if api == ApiKind::Responses {
                            set_last_response_id.set(None);
                        }
                        set_messages.update(|items| {
                            if let Some(message) = items.get_mut(assistant_index) {
                                message.role = "error".to_string();
                                message.content = error;
                            }
                        });
                        set_status.set("Ready".to_string());
                    }
                }
            }
            abort.set_value(None);
            set_busy.set(false);
        });
    });

    let stop_action = move |_| {
        abort.with_value(|c| {
            if let Some(controller) = c {
                controller.abort();
            }
        });
        set_status.set("Stopping…".to_string());
    };

    let submit_action = send_action.clone();
    let key_action = send_action.clone();
    let drop_action = attach_files.clone();
    let context_feedback = move || {
        context_summary(
            &messages.get(),
            &prompt.get(),
            &attachments.get(),
            &settings.get(),
            &models.get(),
            usage.get(),
        )
    };

    view! {
        <div
            class=move || if drag_active.get() { "shell dragging" } else { "shell" }
            on:dragenter=move |ev: DragEvent| { ev.prevent_default(); if !busy.get_untracked() { set_drag_active.set(true); } }
            on:dragover=move |ev: DragEvent| { ev.prevent_default(); if !busy.get_untracked() { set_drag_active.set(true); } }
            on:dragleave=move |ev: DragEvent| { ev.prevent_default(); set_drag_active.set(false); }
            on:drop=move |ev: DragEvent| {
                ev.prevent_default();
                set_drag_active.set(false);
                if let Some(data) = ev.data_transfer() {
                    if let Some(files) = data.files() {
                        drop_action(files_to_vec(files));
                    }
                }
            }
        >
            <div class="drop-target" aria-hidden="true"></div>
            <header>
                <h1>"Hipfire Chat"</h1>
                <div class="status">{move || status.get()}</div>
                <div class="header-actions">
                    <button type="button" title="Settings" on:click=move |_| set_settings_open.update(|v| *v = !*v)>"⚙"</button>
                    <button type="button" title="Clear conversation" on:click=move |_| {
                        set_messages.set(Vec::new());
                        set_attachments.set(Vec::new());
                        set_usage.set(None);
                        set_last_response_id.set(None);
                        set_status.set("Ready".to_string());
                    }>"Clear"</button>
                </div>
            </header>

            <SettingsDrawer settings=settings set_settings=set_settings models=models
                open=settings_open set_open=set_settings_open version=server_version/>

            <main class="messages" node_ref=messages_ref aria-live="polite">
                {move || messages.get().into_iter().map(message_view).collect_view()}
            </main>

            <form class="composer" on:submit=move |ev: SubmitEvent| { ev.prevent_default(); submit_action(); }>
                <div class="context-feedback">{context_feedback}</div>
                <div class="attachment-strip">
                    {move || attachments.get().into_iter().enumerate().map(|(idx, image)| {
                        view! {
                            <figure class="attachment">
                                <img src=image.data_url alt=image.name.clone() />
                                <figcaption>{image.name}</figcaption>
                                <button type="button" aria-label="Remove image" prop:disabled=move || busy.get()
                                    on:click=move |_| set_attachments.update(|items| { if idx < items.len() { items.remove(idx); } })>
                                    "×"
                                </button>
                            </figure>
                        }
                    }).collect_view()}
                </div>
                <textarea
                    class="prompt"
                    autocomplete="off"
                    spellcheck="true"
                    placeholder="Message hipfire"
                    prop:value=move || prompt.get()
                    on:input=move |ev| {
                        set_prompt.set(event_target_value(&ev));
                        if let Some(ta) = ev.target().and_then(|t| t.dyn_into::<HtmlTextAreaElement>().ok()) {
                            autosize(&ta);
                        }
                    }
                    on:keydown=move |ev: KeyboardEvent| {
                        if ev.key() == "Enter" && !ev.shift_key() { ev.prevent_default(); key_action(); }
                    }
                ></textarea>
                {move || if busy.get() {
                    view! { <button class="primary stop" type="button" on:click=stop_action>"Stop"</button> }.into_any()
                } else {
                    view! { <button class="primary" type="submit">"Send"</button> }.into_any()
                }}
            </form>
        </div>
    }
}

/// Render one transcript message: role tag, optional reasoning panel, markdown
/// body (assistant) or plain text (user/error), images, and a copy button.
fn message_view(message: UiMessage) -> impl IntoView {
    let class = format!("message {}", message.role);
    let is_assistant = message.role == "assistant";
    let role = message.role.clone();
    let content = message.content.clone();
    let copy_text = message.content.clone();
    let reasoning = message.reasoning;
    let images = message.images;

    let reasoning_view = if reasoning.is_empty() {
        ().into_any()
    } else {
        view! {
            <details class="reasoning"><summary>"Reasoning"</summary><div>{reasoning}</div></details>
        }
        .into_any()
    };

    let body_view = if content.is_empty() {
        ().into_any()
    } else if is_assistant {
        view! { <div class="md" inner_html=render_markdown(&content)></div> }.into_any()
    } else {
        view! { <div class="text">{content}</div> }.into_any()
    };

    let images_view = if images.is_empty() {
        ().into_any()
    } else {
        view! {
            <div class="message-images">
                {images.into_iter().map(|image| view! { <img src=image.data_url alt=image.name /> }).collect_view()}
            </div>
        }
        .into_any()
    };

    let copy_view = if is_assistant && !copy_text.is_empty() {
        view! {
            <button class="copy" type="button" title="Copy" on:click=move |_| copy_to_clipboard(&copy_text)>"Copy"</button>
        }
        .into_any()
    } else {
        ().into_any()
    };

    view! {
        <article class=class>
            <span class="role">{role}</span>
            {reasoning_view}
            {body_view}
            {images_view}
            {copy_view}
        </article>
    }
}

/// Settings drawer: model selection + sampler overrides + system prompt. Only
/// exposes parameters the server's ChatRequest actually honors.
#[component]
fn SettingsDrawer(
    settings: ReadSignal<Settings>,
    set_settings: WriteSignal<Settings>,
    models: ReadSignal<Vec<ModelItem>>,
    open: ReadSignal<bool>,
    set_open: WriteSignal<bool>,
    version: ReadSignal<Option<String>>,
) -> impl IntoView {
    // Helper to build a labeled numeric override input bound to one field.
    macro_rules! num_field {
        ($label:expr, $get:expr, $set:expr, $ph:expr) => {
            view! {
                <label>{$label}
                    <input type="text" inputmode="decimal" placeholder=$ph
                        prop:value=move || ($get)(&settings.get())
                        on:input=move |ev| { let v = event_target_value(&ev); set_settings.update(|s| ($set)(s, v)); } />
                </label>
            }
        };
    }

    view! {
        <aside class=move || if open.get() { "drawer open" } else { "drawer" }>
            <div class="drawer-head">
                <strong>"Settings"</strong>
                <button type="button" title="Close" on:click=move |_| set_open.set(false)>"×"</button>
            </div>
            <label>"API"
                <select prop:value=move || settings.get().api.as_str().to_string()
                    on:change=move |ev| { let v = event_target_value(&ev); set_settings.update(|s| s.api = ApiKind::parse(&v)); }>
                    <option value="chat">"Chat Completions"</option>
                    <option value="responses">"Responses"</option>
                </select>
            </label>
            {move || (settings.get().api == ApiKind::Responses)
                .then(|| view! { <p class="hint">"Responses keeps conversation state server-side (previous_response_id)."</p> })}
            <label>"Model"
                <input list="model-list" autocomplete="off" placeholder="default"
                    prop:value=move || settings.get().model
                    on:input=move |ev| { let v = event_target_value(&ev); set_settings.update(|s| s.model = v); } />
                <datalist id="model-list">
                    {move || models.get().into_iter().map(|m| view! { <option value=m.id></option> }).collect_view()}
                </datalist>
            </label>
            <div class="grid2">
                {num_field!("Max tokens", |s: &Settings| s.max_tokens.clone(), |s: &mut Settings, v| s.max_tokens = v, "512")}
                {num_field!("Temperature", |s: &Settings| s.temperature.clone(), |s: &mut Settings, v| s.temperature = v, "0.7")}
                {num_field!("Top-p", |s: &Settings| s.top_p.clone(), |s: &mut Settings, v| s.top_p = v, "default")}
                {num_field!("Repeat penalty", |s: &Settings| s.repeat_penalty.clone(), |s: &mut Settings, v| s.repeat_penalty = v, "default")}
                {num_field!("Presence penalty", |s: &Settings| s.presence_penalty.clone(), |s: &mut Settings, v| s.presence_penalty = v, "default")}
                {num_field!("Frequency penalty", |s: &Settings| s.frequency_penalty.clone(), |s: &mut Settings, v| s.frequency_penalty = v, "default")}
            </div>
            <div class="grid2">
                <label>"Reasoning"
                    <select prop:value=move || settings.get().reasoning_effort
                        on:change=move |ev| { let v = event_target_value(&ev); set_settings.update(|s| s.reasoning_effort = v); }>
                        <option value="">"off"</option>
                        <option value="low">"low"</option>
                        <option value="medium">"medium"</option>
                        <option value="high">"high"</option>
                    </select>
                </label>
                <label class="toggle">
                    <input type="checkbox" prop:checked=move || settings.get().stream
                        on:change=move |ev| { let v = event_target_checked(&ev); set_settings.update(|s| s.stream = v); } />
                    "Stream"
                </label>
            </div>
            <label>"System prompt"
                <textarea class="system" placeholder="optional system message"
                    prop:value=move || settings.get().system
                    on:input=move |ev| { let v = event_target_value(&ev); set_settings.update(|s| s.system = v); }></textarea>
            </label>
            <div class="drawer-foot">
                {move || match version.get() {
                    Some(v) => format!("hipfire {v}"),
                    None => "hipfire".to_string(),
                }}
            </div>
        </aside>
    }
}

/// Pick the endpoint and build the matching request body for the active API.
/// `prev_id` (Responses only) chains server-side state via previous_response_id.
fn build_request(messages: &[UiMessage], cfg: &Settings, prev_id: Option<&str>) -> (&'static str, Value) {
    let body = match cfg.api {
        ApiKind::Chat => chat_request_body(messages, cfg),
        ApiKind::Responses => responses_request_body(messages, cfg, prev_id),
    };
    (cfg.api.url(), body)
}

/// Build a `/v1/responses` body. Uses `input` (role/content items) instead of
/// `messages` and `max_output_tokens` instead of `max_tokens`. Content goes
/// through `chat_message_content`, so the last user turn's images ride along as
/// `image_url` parts (vision). When `prev_id` is set the server already holds
/// prior context, so only the newest user turn is sent.
fn responses_request_body(messages: &[UiMessage], cfg: &Settings, prev_id: Option<&str>) -> Value {
    let last_user_index = messages.iter().rposition(|m| m.role == "user");
    let mut input: Vec<Value> = Vec::new();
    if prev_id.is_none() {
        let system = cfg.system.trim();
        if !system.is_empty() {
            input.push(json!({"role": "system", "content": system}));
        }
        input.extend(
            messages
                .iter()
                .enumerate()
                .filter(|(_, m)| m.role == "user" || m.role == "assistant")
                .map(|(idx, m)| {
                    json!({"role": m.role, "content": chat_message_content(m, Some(idx) == last_user_index)})
                }),
        );
    } else if let Some(idx) = last_user_index {
        // Continuing: server reconstructs history; send only the new user turn.
        input.push(json!({"role": "user", "content": chat_message_content(&messages[idx], true)}));
    }
    let mut body = json!({
        "input": input,
        "max_output_tokens": cfg.max_tokens.parse::<u32>().unwrap_or(512).max(1),
        "stream": cfg.stream,
        "chat_template_kwargs": {"enable_thinking": !cfg.reasoning_effort.is_empty()},
    });
    if let Some(id) = prev_id {
        body["previous_response_id"] = json!(id);
    }
    set_if_num(&mut body, "temperature", &cfg.temperature);
    set_if_num(&mut body, "top_p", &cfg.top_p);
    set_if_num(&mut body, "repeat_penalty", &cfg.repeat_penalty);
    set_if_num(&mut body, "presence_penalty", &cfg.presence_penalty);
    set_if_num(&mut body, "frequency_penalty", &cfg.frequency_penalty);
    let model = cfg.model.trim();
    if !model.is_empty() {
        body["model"] = json!(model);
    }
    if !cfg.reasoning_effort.is_empty() {
        body["reasoning_effort"] = json!(cfg.reasoning_effort);
    }
    body
}

fn chat_request_body(messages: &[UiMessage], cfg: &Settings) -> Value {
    let last_user_index = messages.iter().rposition(|message| message.role == "user");
    let mut body = json!({
        "messages": messages
            .iter()
            .enumerate()
            .filter(|(_, m)| m.role == "user" || m.role == "assistant")
            .map(|(idx, m)| json!({
                "role": m.role,
                "content": chat_message_content(m, Some(idx) == last_user_index),
            }))
            .collect::<Vec<_>>(),
        "max_tokens": cfg.max_tokens.parse::<u32>().unwrap_or(512).max(1),
        "stream": cfg.stream,
        "chat_template_kwargs": {"enable_thinking": !cfg.reasoning_effort.is_empty()},
    });
    set_if_num(&mut body, "temperature", &cfg.temperature);
    set_if_num(&mut body, "top_p", &cfg.top_p);
    set_if_num(&mut body, "repeat_penalty", &cfg.repeat_penalty);
    set_if_num(&mut body, "presence_penalty", &cfg.presence_penalty);
    set_if_num(&mut body, "frequency_penalty", &cfg.frequency_penalty);
    let model = cfg.model.trim();
    if !model.is_empty() {
        body["model"] = json!(model);
    }
    let system = cfg.system.trim();
    if !system.is_empty() {
        body["system"] = json!(system);
    }
    if !cfg.reasoning_effort.is_empty() {
        body["reasoning_effort"] = json!(cfg.reasoning_effort);
    }
    if cfg.stream {
        body["stream_options"] = json!({"include_usage": true});
    }
    body
}

/// Set a JSON number field from a string only when it parses as f64.
fn set_if_num(body: &mut Value, key: &str, raw: &str) {
    if let Ok(v) = raw.trim().parse::<f64>() {
        body[key] = json!(v);
    }
}

fn context_summary(
    messages: &[UiMessage],
    prompt: &str,
    attachments: &[AttachedImage],
    cfg: &Settings,
    models: &[ModelItem],
    usage: Option<Usage>,
) -> String {
    if let Some(u) = usage {
        return format!("Tokens: {} in + {} out = {}", u.prompt, u.completion, u.prompt + u.completion);
    }
    let input_tokens = estimate_input_tokens(messages, prompt, attachments);
    let output_tokens = cfg.max_tokens.parse::<u32>().unwrap_or(512).max(1);
    let total = input_tokens.saturating_add(output_tokens);
    let limit = models
        .iter()
        .find(|item| item.id == cfg.model)
        .and_then(|item| item.context);
    match limit {
        Some(limit) if limit > 0 => {
            let pct = ((total as f64 / limit as f64) * 100.0).ceil() as u32;
            format!("Context: ~{input_tokens} in + {output_tokens} out / {limit} ({pct}%)")
        }
        _ => format!("Context: ~{input_tokens} in + {output_tokens} out; model limit unknown"),
    }
}

fn estimate_input_tokens(messages: &[UiMessage], prompt: &str, attachments: &[AttachedImage]) -> u32 {
    let message_chars = messages
        .iter()
        .map(|m| m.content.chars().count() as u32)
        .sum::<u32>();
    let prompt_chars = prompt.chars().count() as u32;
    let text_tokens = message_chars.saturating_add(prompt_chars).saturating_add(3) / 4;
    let image_headroom = messages
        .iter()
        .map(|m| m.images.len() as u32)
        .sum::<u32>()
        .saturating_add(attachments.len() as u32)
        .saturating_mul(1024);
    text_tokens.saturating_add(image_headroom)
}

fn chat_message_content(message: &UiMessage, include_images: bool) -> Value {
    if !include_images || message.images.is_empty() {
        return json!(message.content);
    }
    let mut parts = Vec::new();
    if !message.content.is_empty() {
        parts.push(json!({"type": "text", "text": message.content}));
    }
    parts.extend(
        message
            .images
            .iter()
            .map(|image| json!({"type": "image_url", "image_url": {"url": image.data_url}})),
    );
    Value::Array(parts)
}

/// Render markdown to HTML, neutralizing raw HTML (model output is untrusted):
/// inline/block HTML events are downgraded to escaped text by `push_html`.
fn render_markdown(src: &str) -> String {
    use pulldown_cmark::{html, Event, Options, Parser};
    let mut opts = Options::empty();
    opts.insert(Options::ENABLE_TABLES);
    opts.insert(Options::ENABLE_STRIKETHROUGH);
    let parser = Parser::new_ext(src, opts).map(|event| match event {
        Event::Html(h) | Event::InlineHtml(h) => Event::Text(h),
        other => other,
    });
    let mut out = String::new();
    html::push_html(&mut out, parser);
    out
}

fn copy_to_clipboard(text: &str) {
    if let Some(win) = web_sys::window() {
        let _ = win.navigator().clipboard().write_text(text);
    }
}

/// Grow a textarea to fit its content (auto-resize), capped by CSS max-height.
fn autosize(ta: &HtmlTextAreaElement) {
    let el: &web_sys::HtmlElement = ta.as_ref();
    let style = el.style();
    let _ = style.set_property("height", "auto");
    let _ = style.set_property("height", &format!("{}px", ta.scroll_height()));
}

async fn read_attached_image(file: web_sys::File) -> Result<AttachedImage, String> {
    let name = file.name();
    let mime = file.type_();
    if mime != "image/png" && mime != "image/jpeg" && mime != "image/webp" {
        return Err("unsupported image format".to_string());
    }
    let file = File::from(file);
    let data_url = read_as_data_url(&file).await.map_err(|e| e.to_string())?;
    Ok(AttachedImage { name, data_url })
}

fn files_to_vec(files: web_sys::FileList) -> Vec<web_sys::File> {
    (0..files.length()).filter_map(|idx| files.get(idx)).collect()
}

async fn fetch_models() -> Result<Vec<ModelItem>, String> {
    get_json::<ModelsEnvelope>("/v1/models").await.map(|e| e.data)
}

async fn fetch_completion(
    url: &str,
    body: Value,
    api: ApiKind,
    set_usage: WriteSignal<Option<Usage>>,
    set_response_id: WriteSignal<Option<String>>,
) -> Result<String, String> {
    let result = post_json(url, &body).await?;
    let payload = result.payload;
    if result.status >= 400 || payload.get("error").is_some() {
        return Err(error_message(&payload).unwrap_or_else(|| format!("HTTP {}", result.status)));
    }
    if let Some(u) = parse_usage(&payload) {
        set_usage.set(Some(u));
    }
    if api == ApiKind::Responses {
        if let Some(id) = payload["id"].as_str() {
            set_response_id.set(Some(id.to_string()));
        }
    }
    let content = match api {
        // Responses exposes the full text via the `output_text` convenience field.
        ApiKind::Responses => payload["output_text"].as_str().unwrap_or_default().to_string(),
        ApiKind::Chat => payload["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or_default()
            .to_string(),
    };
    Ok(content)
}

#[allow(clippy::too_many_arguments)]
async fn stream_completion(
    url: &str,
    body: Value,
    controller: Option<AbortController>,
    set_messages: WriteSignal<Vec<UiMessage>>,
    assistant_index: usize,
    set_status: WriteSignal<String>,
    set_usage: WriteSignal<Option<Usage>>,
    set_response_id: WriteSignal<Option<String>>,
) -> Result<(), String> {
    let signal = controller.as_ref().map(|c| c.signal());
    sse_post(url, &body, signal.as_ref(), |data| {
        handle_sse_data(data, set_messages, assistant_index, set_status, set_usage, set_response_id)
    })
    .await
}

/// Interpret one SSE `data:` payload — handles both the Chat Completions and
/// Responses event shapes and routes content/reasoning/usage into the signals.
fn handle_sse_data(
    data: &str,
    set_messages: WriteSignal<Vec<UiMessage>>,
    assistant_index: usize,
    set_status: WriteSignal<String>,
    set_usage: WriteSignal<Option<Usage>>,
    set_response_id: WriteSignal<Option<String>>,
) -> Result<(), String> {
    let payload: Value = serde_json::from_str(data).map_err(|e| e.to_string())?;
    if payload.get("error").is_some() {
        return Err(error_message(&payload).unwrap_or_else(|| "request failed".to_string()));
    }
    if let Some(u) = parse_usage(&payload) {
        set_usage.set(Some(u));
    }
    // Responses API: typed events carry the token in `delta`; events also carry
    // the response id for server-side state chaining.
    if let Some(kind) = payload["type"].as_str() {
        if kind.starts_with("response.") {
            if let Some(id) = payload["response_id"]
                .as_str()
                .or_else(|| payload["response"]["id"].as_str())
            {
                set_response_id.set(Some(id.to_string()));
            }
            if kind == "response.output_text.delta" {
                if let Some(content) = payload["delta"].as_str() {
                    append_content(set_messages, assistant_index, content);
                }
            }
            return Ok(());
        }
    }
    // Chat Completions API: token in `choices[0].delta.{content,reasoning_content}`.
    let delta = &payload["choices"][0]["delta"];
    if let Some(content) = delta["content"].as_str() {
        append_content(set_messages, assistant_index, content);
    }
    if let Some(reasoning) = delta["reasoning_content"].as_str() {
        set_status.set("Thinking".to_string());
        set_messages.update(|items| {
            if let Some(message) = items.get_mut(assistant_index) {
                message.reasoning.push_str(reasoning);
            }
        });
    }
    Ok(())
}

fn append_content(set_messages: WriteSignal<Vec<UiMessage>>, index: usize, content: &str) {
    set_messages.update(|items| {
        if let Some(message) = items.get_mut(index) {
            message.content.push_str(content);
        }
    });
}

/// Token usage, tolerant of both naming schemes: chat uses
/// `prompt_tokens`/`completion_tokens`; responses uses `input_tokens`/
/// `output_tokens` and may nest it under a `response` object.
fn parse_usage(payload: &Value) -> Option<Usage> {
    let usage = payload
        .get("usage")
        .or_else(|| payload.get("response").and_then(|r| r.get("usage")))?;
    let prompt = usage["prompt_tokens"]
        .as_u64()
        .or_else(|| usage["input_tokens"].as_u64())
        .unwrap_or(0);
    let completion = usage["completion_tokens"]
        .as_u64()
        .or_else(|| usage["output_tokens"].as_u64())
        .unwrap_or(0);
    Some(Usage {
        prompt: prompt as u32,
        completion: completion as u32,
    })
}

fn error_message(payload: &Value) -> Option<String> {
    payload["error"]["message"].as_str().map(str::to_string)
}
