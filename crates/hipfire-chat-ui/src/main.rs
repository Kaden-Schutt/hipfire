//! hipfire browser chat console — Leptos CSR app.
//!
//! This stays browser-heavy by design: media capture, local WASM/WebNN models,
//! and terminal emulation can land here without coupling the server API to a
//! frontend framework.

use std::rc::Rc;

use gloo_net::http::Request as HttpRequest;
use js_sys::{Reflect, Uint8Array};
use leptos::prelude::*;
use serde::Deserialize;
use serde_json::{json, Value};
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;
use web_sys::{
    KeyboardEvent, ReadableStreamDefaultReader, Request, RequestInit, RequestMode, Response,
    SubmitEvent, TextDecodeOptions,
};

#[derive(Clone, Debug, PartialEq)]
struct UiMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct ModelsEnvelope {
    data: Vec<ModelItem>,
}

#[derive(Debug, Deserialize)]
struct ModelItem {
    id: String,
}

fn main() {
    console_error_panic_hook::set_once();
    leptos::mount::mount_to_body(App);
}

#[component]
fn App() -> impl IntoView {
    let (messages, set_messages) = signal(Vec::<UiMessage>::new());
    let (models, set_models) = signal(Vec::<String>::new());
    let (model, set_model) = signal(String::new());
    let (max_tokens, set_max_tokens) = signal("512".to_string());
    let (temperature, set_temperature) = signal("0.7".to_string());
    let (stream, set_stream) = signal(true);
    let (prompt, set_prompt) = signal(String::new());
    let (busy, set_busy) = signal(false);
    let (status, set_status) = signal("Ready".to_string());

    leptos::task::spawn_local(async move {
        if let Ok(items) = fetch_models().await {
            if model.get_untracked().is_empty() && items.len() == 1 {
                set_model.set(items[0].clone());
            }
            set_models.set(items);
        }
    });

    let send_action = Rc::new(move || {
        if busy.get_untracked() {
            return;
        }
        let text = prompt.get_untracked().trim().to_string();
        if text.is_empty() {
            return;
        }

        let user_message = UiMessage {
            role: "user".to_string(),
            content: text,
        };
        let mut request_messages = messages.get_untracked();
        request_messages.push(user_message.clone());
        let assistant_index = request_messages.len();

        set_messages.update(move |items| {
            items.push(user_message);
            items.push(UiMessage {
                role: "assistant".to_string(),
                content: String::new(),
            });
        });
        set_prompt.set(String::new());
        set_busy.set(true);
        set_status.set("Generating".to_string());

        let body = chat_request_body(
            &request_messages,
            &model.get_untracked(),
            &max_tokens.get_untracked(),
            &temperature.get_untracked(),
            stream.get_untracked(),
        );
        let use_stream = stream.get_untracked();
        leptos::task::spawn_local(async move {
            let result = if use_stream {
                stream_completion(body, set_messages, assistant_index, set_status).await
            } else {
                fetch_completion(body).await.map(|content| {
                    set_messages.update(|items| {
                        if let Some(message) = items.get_mut(assistant_index) {
                            message.content = content;
                        }
                    });
                })
            };

            if let Err(error) = result {
                set_messages.update(|items| {
                    if let Some(message) = items.get_mut(assistant_index) {
                        message.role = "error".to_string();
                        message.content = error;
                    }
                });
            }
            set_status.set("Ready".to_string());
            set_busy.set(false);
        });
    });

    let submit_action = send_action.clone();
    let key_action = send_action.clone();

    view! {
        <div class="shell">
            <header>
                <h1>"Hipfire Chat"</h1>
                <div class="status">{move || status.get()}</div>
            </header>
            <section class="controls" aria-label="Chat controls">
                <label>
                    "Model"
                    <input
                        list="model-list"
                        autocomplete="off"
                        placeholder="default"
                        prop:value=move || model.get()
                        on:input=move |ev| set_model.set(event_target_value(&ev))
                    />
                    <datalist id="model-list">
                        {move || models.get().into_iter().map(|id| view! { <option value=id></option> }).collect_view()}
                    </datalist>
                </label>
                <label>
                    "Tokens"
                    <input
                        type="number"
                        min="1"
                        max="131072"
                        prop:value=move || max_tokens.get()
                        on:input=move |ev| set_max_tokens.set(event_target_value(&ev))
                    />
                </label>
                <label>
                    "Temp"
                    <input
                        type="number"
                        min="0"
                        max="2"
                        step="0.05"
                        prop:value=move || temperature.get()
                        on:input=move |ev| set_temperature.set(event_target_value(&ev))
                    />
                </label>
                <label class="toggle">
                    <input
                        type="checkbox"
                        prop:checked=move || stream.get()
                        on:change=move |ev| set_stream.set(event_target_checked(&ev))
                    />
                    "Stream"
                </label>
                <button
                    type="button"
                    prop:disabled=move || busy.get()
                    on:click=move |_| {
                        set_messages.set(Vec::new());
                        set_status.set("Ready".to_string());
                    }
                >
                    "Clear"
                </button>
            </section>
            <main class="messages" aria-live="polite">
                {move || messages.get().into_iter().map(|message| {
                    let class = format!("message {}", message.role);
                    let role = message.role;
                    let content = message.content;
                    view! {
                        <article class=class>
                            <span class="role">{role}</span>
                            <div>{content}</div>
                        </article>
                    }
                }).collect_view()}
            </main>
            <form on:submit=move |ev: SubmitEvent| {
                ev.prevent_default();
                submit_action();
            }>
                <textarea
                    autocomplete="off"
                    spellcheck="true"
                    placeholder="Message hipfire"
                    prop:value=move || prompt.get()
                    on:input=move |ev| set_prompt.set(event_target_value(&ev))
                    on:keydown=move |ev: KeyboardEvent| {
                        if ev.key() == "Enter" && !ev.shift_key() {
                            ev.prevent_default();
                            key_action();
                        }
                    }
                ></textarea>
                <button class="primary" type="submit" prop:disabled=move || busy.get()>
                    "Send"
                </button>
            </form>
        </div>
    }
}

fn chat_request_body(
    messages: &[UiMessage],
    model: &str,
    max_tokens: &str,
    temperature: &str,
    stream: bool,
) -> Value {
    let mut body = json!({
        "messages": messages
            .iter()
            .filter(|message| message.role == "user" || message.role == "assistant")
            .map(|message| json!({"role": message.role, "content": message.content}))
            .collect::<Vec<_>>(),
        "max_tokens": max_tokens.parse::<u32>().unwrap_or(512).max(1),
        "temperature": temperature.parse::<f64>().unwrap_or(0.7),
        "stream": stream,
        "chat_template_kwargs": {"enable_thinking": false},
    });
    let model = model.trim();
    if !model.is_empty() {
        body["model"] = json!(model);
    }
    if stream {
        body["stream_options"] = json!({"include_usage": true});
    }
    body
}

async fn fetch_models() -> Result<Vec<String>, String> {
    let resp = HttpRequest::get("/v1/models")
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !resp.ok() {
        return Err(format!("/v1/models returned HTTP {}", resp.status()));
    }
    let envelope = resp
        .json::<ModelsEnvelope>()
        .await
        .map_err(|e| e.to_string())?;
    Ok(envelope.data.into_iter().map(|model| model.id).collect())
}

async fn fetch_completion(body: Value) -> Result<String, String> {
    let resp = HttpRequest::post("/v1/chat/completions")
        .json(&body)
        .map_err(|e| e.to_string())?
        .send()
        .await
        .map_err(|e| e.to_string())?;
    let status = resp.status();
    let payload = resp.json::<Value>().await.map_err(|e| e.to_string())?;
    if status >= 400 || payload.get("error").is_some() {
        return Err(error_message(&payload).unwrap_or_else(|| format!("HTTP {status}")));
    }
    Ok(payload["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default()
        .to_string())
}

async fn stream_completion(
    body: Value,
    set_messages: WriteSignal<Vec<UiMessage>>,
    assistant_index: usize,
    set_status: WriteSignal<String>,
) -> Result<(), String> {
    let opts = RequestInit::new();
    opts.set_method("POST");
    opts.set_mode(RequestMode::SameOrigin);
    let serialized = serde_json::to_string(&body).map_err(|e| e.to_string())?;
    opts.set_body(&JsValue::from_str(&serialized));
    let request =
        Request::new_with_str_and_init("/v1/chat/completions", &opts).map_err(js_error)?;
    request
        .headers()
        .set("Content-Type", "application/json")
        .map_err(js_error)?;

    let window = web_sys::window().ok_or_else(|| "window unavailable".to_string())?;
    let resp = JsFuture::from(window.fetch_with_request(&request))
        .await
        .map_err(js_error)?
        .dyn_into::<Response>()
        .map_err(js_error)?;
    if !resp.ok() {
        return Err(format!("HTTP {}", resp.status()));
    }
    let stream = resp
        .body()
        .ok_or_else(|| "response body stream unavailable".to_string())?;
    let reader = stream
        .get_reader()
        .dyn_into::<ReadableStreamDefaultReader>()
        .map_err(|e| js_error(e.into()))?;
    let decoder = web_sys::TextDecoder::new().map_err(js_error)?;
    let decode_opts = TextDecodeOptions::new();
    decode_opts.set_stream(true);
    let mut buffer = String::new();

    loop {
        let chunk = JsFuture::from(reader.read()).await.map_err(js_error)?;
        let done = Reflect::get(&chunk, &JsValue::from_str("done"))
            .map_err(js_error)?
            .as_bool()
            .unwrap_or(false);
        if done {
            break;
        }
        let value = Reflect::get(&chunk, &JsValue::from_str("value")).map_err(js_error)?;
        let bytes = Uint8Array::new(&value);
        let mut bytes_vec = vec![0; bytes.length() as usize];
        bytes.copy_to(&mut bytes_vec);
        let text = decoder
            .decode_with_u8_array_and_options(&bytes_vec, &decode_opts)
            .map_err(js_error)?;
        buffer.push_str(&text);
        consume_sse_buffer(&mut buffer, set_messages, assistant_index, set_status)?;
    }
    if !buffer.trim().is_empty() {
        let event = std::mem::take(&mut buffer);
        consume_sse_event(&event, set_messages, assistant_index, set_status)?;
    }
    Ok(())
}

fn consume_sse_buffer(
    buffer: &mut String,
    set_messages: WriteSignal<Vec<UiMessage>>,
    assistant_index: usize,
    set_status: WriteSignal<String>,
) -> Result<(), String> {
    while let Some(split) = buffer.find("\n\n") {
        let event = buffer[..split].to_string();
        buffer.drain(..split + 2);
        consume_sse_event(&event, set_messages, assistant_index, set_status)?;
    }
    Ok(())
}

fn consume_sse_event(
    event: &str,
    set_messages: WriteSignal<Vec<UiMessage>>,
    assistant_index: usize,
    set_status: WriteSignal<String>,
) -> Result<(), String> {
    for line in event.lines() {
        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        let data = data.trim_start();
        if data.is_empty() || data == "[DONE]" {
            continue;
        }
        let payload: Value = serde_json::from_str(data).map_err(|e| e.to_string())?;
        if payload.get("error").is_some() {
            return Err(error_message(&payload).unwrap_or_else(|| "request failed".to_string()));
        }
        if let Some(content) = payload["choices"][0]["delta"]["content"].as_str() {
            set_messages.update(|items| {
                if let Some(message) = items.get_mut(assistant_index) {
                    message.content.push_str(content);
                }
            });
        }
        if payload["choices"][0]["delta"]["reasoning_content"].is_string() {
            set_status.set("Thinking".to_string());
        }
    }
    Ok(())
}

fn error_message(payload: &Value) -> Option<String> {
    payload["error"]["message"].as_str().map(str::to_string)
}

fn js_error(value: JsValue) -> String {
    value
        .as_string()
        .or_else(|| {
            js_sys::JSON::stringify(&value)
                .ok()
                .and_then(|s| s.as_string())
        })
        .unwrap_or_else(|| "JavaScript error".to_string())
}
