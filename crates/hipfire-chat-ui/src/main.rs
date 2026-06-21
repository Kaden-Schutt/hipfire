//! hipfire browser chat console — Leptos CSR app.
//!
//! This stays browser-heavy by design: media capture, local WASM/WebNN models,
//! and terminal emulation can land here without coupling the server API to a
//! frontend framework.

use std::rc::Rc;

use gloo_file::{futures::read_as_data_url, File};
use gloo_net::http::Request as HttpRequest;
use js_sys::{Reflect, Uint8Array};
use leptos::prelude::*;
use serde::Deserialize;
use serde_json::{json, Value};
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;
use web_sys::{
    DragEvent, KeyboardEvent, ReadableStreamDefaultReader, Request, RequestInit, RequestMode,
    Response, SubmitEvent, TextDecodeOptions,
};

#[derive(Clone, Debug, PartialEq)]
struct UiMessage {
    role: String,
    content: String,
    images: Vec<AttachedImage>,
}

#[derive(Clone, Debug, PartialEq)]
struct AttachedImage {
    name: String,
    mime: String,
    data_url: String,
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
    let (model, set_model) = signal(String::new());
    let (max_tokens, set_max_tokens) = signal("512".to_string());
    let (temperature, set_temperature) = signal("0.7".to_string());
    let (stream, set_stream) = signal(true);
    let (prompt, set_prompt) = signal(String::new());
    let (attachments, set_attachments) = signal(Vec::<AttachedImage>::new());
    let (drag_active, set_drag_active) = signal(false);
    let (busy, set_busy) = signal(false);
    let (status, set_status) = signal("Ready".to_string());

    leptos::task::spawn_local(async move {
        if let Ok(items) = fetch_models().await {
            if model.get_untracked().is_empty() && items.len() == 1 {
                set_model.set(items[0].id.clone());
            }
            set_models.set(items);
        }
    });

    let attach_files = Rc::new(move |files: Vec<web_sys::File>| {
        if busy.get_untracked() || files.is_empty() {
            return;
        }
        set_status.set("Reading image".to_string());
        leptos::task::spawn_local(async move {
            let mut accepted = Vec::new();
            let mut rejected = 0usize;
            for file in files {
                match read_attached_image(file).await {
                    Ok(image) => accepted.push(image),
                    Err(_) => rejected += 1,
                }
                if !accepted.is_empty() {
                    break;
                }
            }

            if let Some(image) = accepted.into_iter().next() {
                set_attachments.set(vec![image]);
                if rejected > 0 {
                    set_status.set("Attached one image; skipped unsupported files".to_string());
                } else {
                    set_status.set("Image attached".to_string());
                }
            } else if rejected > 0 {
                set_status.set("Drop a PNG or JPEG image".to_string());
            } else {
                set_status.set("Ready".to_string());
            }
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

        let user_message = UiMessage {
            role: "user".to_string(),
            content: text,
            images: image_attachments,
        };
        let mut request_messages = messages.get_untracked();
        request_messages.push(user_message.clone());
        let assistant_index = request_messages.len();

        set_messages.update(move |items| {
            items.push(user_message);
            items.push(UiMessage {
                role: "assistant".to_string(),
                content: String::new(),
                images: Vec::new(),
            });
        });
        set_prompt.set(String::new());
        set_attachments.set(Vec::new());
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
    let drop_action = attach_files.clone();
    let context_feedback = move || {
        context_summary(
            &messages.get(),
            &prompt.get(),
            &attachments.get(),
            &max_tokens.get(),
            &model.get(),
            &models.get(),
        )
    };

    view! {
        <div
            class=move || if drag_active.get() { "shell dragging" } else { "shell" }
            on:dragenter=move |ev: DragEvent| {
                ev.prevent_default();
                if !busy.get_untracked() {
                    set_drag_active.set(true);
                }
            }
            on:dragover=move |ev: DragEvent| {
                ev.prevent_default();
                if !busy.get_untracked() {
                    set_drag_active.set(true);
                }
            }
            on:dragleave=move |ev: DragEvent| {
                ev.prevent_default();
                set_drag_active.set(false);
            }
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
                        {move || models.get().into_iter().map(|model| view! { <option value=model.id></option> }).collect_view()}
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
                        set_attachments.set(Vec::new());
                        set_status.set("Ready".to_string());
                    }
                >
                    "Clear"
                </button>
                <div class="context-feedback">{context_feedback}</div>
            </section>
            <main class="messages" aria-live="polite">
                {move || messages.get().into_iter().map(|message| {
                    let class = format!("message {}", message.role);
                    let role = message.role;
                    let content = message.content;
                    let images = message.images;
                    view! {
                        <article class=class>
                            <span class="role">{role}</span>
                            {if content.is_empty() {
                                view! { <div></div> }.into_any()
                            } else {
                                view! { <div>{content}</div> }.into_any()
                            }}
                            {if images.is_empty() {
                                view! { <div></div> }.into_any()
                            } else {
                                view! {
                                    <div class="message-images">
                                        {images.into_iter().map(|image| {
                                            view! {
                                                <img src=image.data_url alt=image.name />
                                            }
                                        }).collect_view()}
                                    </div>
                                }.into_any()
                            }}
                        </article>
                    }
                }).collect_view()}
            </main>
            <form class="composer" on:submit=move |ev: SubmitEvent| {
                ev.prevent_default();
                submit_action();
            }>
                <div class="attachment-strip">
                    {move || attachments.get().into_iter().enumerate().map(|(idx, image)| {
                        view! {
                            <figure class="attachment">
                                <img src=image.data_url alt=image.name.clone() />
                                <figcaption>{image.name}</figcaption>
                                <button
                                    type="button"
                                    aria-label="Remove image"
                                    prop:disabled=move || busy.get()
                                    on:click=move |_| {
                                        set_attachments.update(|items| {
                                            if idx < items.len() {
                                                items.remove(idx);
                                            }
                                        });
                                    }
                                >
                                    "x"
                                </button>
                            </figure>
                        }
                    }).collect_view()}
                </div>
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
    let last_user_index = messages.iter().rposition(|message| message.role == "user");
    let mut body = json!({
        "messages": messages
            .iter()
            .enumerate()
            .filter(|(_, message)| message.role == "user" || message.role == "assistant")
            .map(|(idx, message)| json!({
                "role": message.role,
                "content": chat_message_content(message, Some(idx) == last_user_index),
            }))
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

fn context_summary(
    messages: &[UiMessage],
    prompt: &str,
    attachments: &[AttachedImage],
    max_tokens: &str,
    selected_model: &str,
    models: &[ModelItem],
) -> String {
    let input_tokens = estimate_input_tokens(messages, prompt, attachments);
    let output_tokens = max_tokens.parse::<u32>().unwrap_or(512).max(1);
    let total = input_tokens.saturating_add(output_tokens);
    let selected_context = models
        .iter()
        .find(|item| item.id == selected_model)
        .and_then(|item| item.context);

    match selected_context {
        Some(limit) if limit > 0 => {
            let pct = ((total as f64 / limit as f64) * 100.0).ceil() as u32;
            format!("Context: ~{input_tokens} in + {output_tokens} out / {limit} ({pct}%)")
        }
        _ => format!("Context: ~{input_tokens} in + {output_tokens} out; model limit unknown"),
    }
}

fn estimate_input_tokens(
    messages: &[UiMessage],
    prompt: &str,
    attachments: &[AttachedImage],
) -> u32 {
    let message_chars = messages
        .iter()
        .map(|message| message.content.chars().count() as u32)
        .sum::<u32>();
    let prompt_chars = prompt.chars().count() as u32;
    let text_tokens = message_chars.saturating_add(prompt_chars).saturating_add(3) / 4;
    let image_headroom = messages
        .iter()
        .map(|message| message.images.len() as u32)
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

async fn read_attached_image(file: web_sys::File) -> Result<AttachedImage, String> {
    let name = file.name();
    let mime = file.type_();
    if mime != "image/png" && mime != "image/jpeg" {
        return Err("unsupported image format".to_string());
    }
    let file = File::from(file);
    let data_url = read_as_data_url(&file).await.map_err(|e| e.to_string())?;
    Ok(AttachedImage {
        name,
        mime,
        data_url,
    })
}

fn files_to_vec(files: web_sys::FileList) -> Vec<web_sys::File> {
    (0..files.length())
        .filter_map(|idx| files.get(idx))
        .collect()
}

async fn fetch_models() -> Result<Vec<ModelItem>, String> {
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
    Ok(envelope.data)
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
