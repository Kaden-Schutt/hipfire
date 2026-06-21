use axum::response::Html;

pub async fn get_chat_index() -> Html<&'static str> {
    Html(CHAT_INDEX_HTML)
}

const CHAT_INDEX_HTML: &str = r#"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Hipfire Chat</title>
  <style>
    :root {
      color-scheme: light dark;
      --bg: #f7f8f5;
      --panel: #ffffff;
      --text: #151716;
      --muted: #5f6761;
      --line: #d9ded8;
      --accent: #126b5f;
      --accent-2: #2f5e9e;
      --danger: #9b2d30;
      --user: #e7f1ee;
      --assistant: #f2f4f7;
      --shadow: 0 1px 4px rgba(20, 24, 21, 0.08);
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --bg: #111412;
        --panel: #191d1a;
        --text: #edf0eb;
        --muted: #a9b0aa;
        --line: #323832;
        --accent: #36a78f;
        --accent-2: #6fa4e8;
        --danger: #e06d70;
        --user: #19332e;
        --assistant: #20252b;
        --shadow: none;
      }
    }
    * {
      box-sizing: border-box;
    }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font: 15px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    button, input, textarea {
      font: inherit;
    }
    .shell {
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr auto;
    }
    header {
      display: grid;
      gap: 12px;
      grid-template-columns: 1fr auto;
      align-items: center;
      padding: 14px 18px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    h1 {
      margin: 0;
      font-size: 18px;
      font-weight: 650;
      letter-spacing: 0;
    }
    .status {
      color: var(--muted);
      font-size: 13px;
      min-height: 20px;
      text-align: right;
    }
    .controls {
      display: grid;
      grid-template-columns: minmax(180px, 1fr) 96px 96px auto auto;
      gap: 8px;
      padding: 12px 18px;
      border-bottom: 1px solid var(--line);
      background: color-mix(in srgb, var(--panel), var(--bg) 28%);
    }
    label {
      display: grid;
      gap: 4px;
      color: var(--muted);
      font-size: 12px;
      min-width: 0;
    }
    input {
      width: 100%;
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 10px;
      background: var(--panel);
      color: var(--text);
    }
    .messages {
      overflow-y: auto;
      padding: 18px;
      display: grid;
      align-content: start;
      gap: 12px;
    }
    .message {
      width: min(900px, 100%);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      box-shadow: var(--shadow);
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    .message .role {
      display: block;
      margin-bottom: 6px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
      text-transform: uppercase;
    }
    .message.user {
      justify-self: end;
      background: var(--user);
    }
    .message.assistant {
      justify-self: start;
      background: var(--assistant);
    }
    .message.error {
      justify-self: start;
      border-color: color-mix(in srgb, var(--danger), var(--line) 45%);
      background: color-mix(in srgb, var(--danger), var(--panel) 88%);
    }
    form {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      padding: 14px 18px 18px;
      border-top: 1px solid var(--line);
      background: var(--panel);
    }
    textarea {
      width: 100%;
      min-height: 52px;
      max-height: 180px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      background: var(--bg);
      color: var(--text);
    }
    button {
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 12px;
      background: var(--panel);
      color: var(--text);
      cursor: pointer;
    }
    button.primary {
      min-width: 84px;
      align-self: end;
      border-color: var(--accent);
      background: var(--accent);
      color: #ffffff;
      font-weight: 650;
    }
    button:disabled {
      cursor: wait;
      opacity: 0.65;
    }
    .toggle {
      display: flex;
      align-items: end;
      gap: 7px;
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }
    .toggle input {
      width: 18px;
      height: 18px;
      margin: 0 0 9px;
    }
    @media (max-width: 760px) {
      header {
        grid-template-columns: 1fr;
      }
      .status {
        text-align: left;
      }
      .controls {
        grid-template-columns: 1fr 1fr;
      }
      .controls label:first-child {
        grid-column: 1 / -1;
      }
      form {
        grid-template-columns: 1fr;
      }
      button.primary {
        width: 100%;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <h1>Hipfire Chat</h1>
      <div id="status" class="status">Ready</div>
    </header>
    <section class="controls" aria-label="Chat controls">
      <label>
        Model
        <input id="model" name="model" list="model-list" autocomplete="off" placeholder="default">
        <datalist id="model-list"></datalist>
      </label>
      <label>
        Tokens
        <input id="maxTokens" name="maxTokens" type="number" min="1" max="131072" value="512">
      </label>
      <label>
        Temp
        <input id="temperature" name="temperature" type="number" min="0" max="2" step="0.05" value="0.7">
      </label>
      <label class="toggle">
        <input id="stream" name="stream" type="checkbox" checked>
        Stream
      </label>
      <button id="clear" type="button">Clear</button>
    </section>
    <main id="messages" class="messages" aria-live="polite"></main>
    <form id="composer">
      <textarea id="prompt" name="prompt" autocomplete="off" spellcheck="true" placeholder="Message hipfire"></textarea>
      <button id="send" class="primary" type="submit">Send</button>
    </form>
  </div>
  <script>
    const state = {
      messages: [],
      busy: false,
    };

    const els = {
      status: document.getElementById("status"),
      messages: document.getElementById("messages"),
      model: document.getElementById("model"),
      modelList: document.getElementById("model-list"),
      maxTokens: document.getElementById("maxTokens"),
      temperature: document.getElementById("temperature"),
      stream: document.getElementById("stream"),
      clear: document.getElementById("clear"),
      composer: document.getElementById("composer"),
      prompt: document.getElementById("prompt"),
      send: document.getElementById("send"),
    };

    function setStatus(text) {
      els.status.textContent = text;
    }

    function setBusy(value) {
      state.busy = value;
      els.send.disabled = value;
      els.clear.disabled = value;
      setStatus(value ? "Generating" : "Ready");
    }

    function render() {
      els.messages.replaceChildren(...state.messages.map((message) => {
        const node = document.createElement("article");
        node.className = `message ${message.role}`;
        const role = document.createElement("span");
        role.className = "role";
        role.textContent = message.role;
        const content = document.createElement("div");
        content.textContent = message.content || "";
        node.append(role, content);
        return node;
      }));
      els.messages.scrollTop = els.messages.scrollHeight;
    }

    function pushMessage(role, content) {
      state.messages.push({role, content});
      render();
      return state.messages[state.messages.length - 1];
    }

    function requestBody(stream, messages) {
      const body = {
        messages: messages
          .filter((message) => message.role === "user" || message.role === "assistant")
          .map((message) => ({role: message.role, content: message.content})),
        max_tokens: Math.max(1, Number(els.maxTokens.value || 512)),
        temperature: Number(els.temperature.value || 0.7),
        stream,
        chat_template_kwargs: {enable_thinking: false},
      };
      const model = els.model.value.trim();
      if (model) {
        body.model = model;
      }
      if (stream) {
        body.stream_options = {include_usage: true};
      }
      return body;
    }

    async function sendChat(event) {
      event.preventDefault();
      if (state.busy) {
        return;
      }
      const prompt = els.prompt.value.trim();
      if (!prompt) {
        els.prompt.focus();
        return;
      }

      pushMessage("user", prompt);
      const requestMessages = state.messages.slice();
      const assistant = pushMessage("assistant", "");
      els.prompt.value = "";
      setBusy(true);

      try {
        if (els.stream.checked) {
          await streamCompletion(assistant, requestMessages);
        } else {
          await fetchCompletion(assistant, requestMessages);
        }
      } catch (err) {
        assistant.role = "error";
        assistant.content = err instanceof Error ? err.message : String(err);
        render();
      } finally {
        setBusy(false);
        els.prompt.focus();
      }
    }

    async function fetchCompletion(assistant, requestMessages) {
      const resp = await fetch("/v1/chat/completions", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(requestBody(false, requestMessages)),
      });
      const payload = await resp.json();
      if (!resp.ok || payload.error) {
        throw new Error(payload.error?.message || `HTTP ${resp.status}`);
      }
      assistant.content = payload.choices?.[0]?.message?.content || "";
      render();
    }

    async function streamCompletion(assistant, requestMessages) {
      const resp = await fetch("/v1/chat/completions", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(requestBody(true, requestMessages)),
      });
      if (!resp.ok || !resp.body) {
        throw new Error(`HTTP ${resp.status}`);
      }
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const {value, done} = await reader.read();
        if (done) {
          break;
        }
        buffer += decoder.decode(value, {stream: true});
        const events = buffer.split("\n\n");
        buffer = events.pop() || "";
        for (const event of events) {
          consumeSseEvent(event, assistant);
        }
      }
      if (buffer.trim()) {
        consumeSseEvent(buffer, assistant);
      }
    }

    function consumeSseEvent(event, assistant) {
      const lines = event.split("\n");
      for (const line of lines) {
        if (!line.startsWith("data:")) {
          continue;
        }
        const data = line.slice(5).trimStart();
        if (!data || data === "[DONE]") {
          continue;
        }
        const payload = JSON.parse(data);
        if (payload.error) {
          throw new Error(payload.error.message || "request failed");
        }
        const delta = payload.choices?.[0]?.delta;
        if (delta?.content) {
          assistant.content += delta.content;
          render();
        }
        if (delta?.reasoning_content) {
          setStatus("Thinking");
        }
      }
    }

    async function loadModels() {
      try {
        const resp = await fetch("/v1/models");
        if (!resp.ok) {
          return;
        }
        const payload = await resp.json();
        const models = Array.isArray(payload.data) ? payload.data : [];
        els.modelList.replaceChildren(...models.map((model) => {
          const option = document.createElement("option");
          option.value = model.id;
          return option;
        }));
        if (!els.model.value && models.length === 1) {
          els.model.value = models[0].id;
        }
      } catch (_) {
      }
    }

    els.composer.addEventListener("submit", sendChat);
    els.clear.addEventListener("click", () => {
      if (state.busy) {
        return;
      }
      state.messages = [];
      render();
      setStatus("Ready");
      els.prompt.focus();
    });
    els.prompt.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        els.composer.requestSubmit();
      }
    });

    loadModels();
    render();
  </script>
</body>
</html>"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_index_targets_openai_chat_endpoint() {
        assert!(CHAT_INDEX_HTML.contains("/v1/chat/completions"));
        assert!(CHAT_INDEX_HTML.contains("/v1/models"));
        assert!(CHAT_INDEX_HTML.contains("Message hipfire"));
    }
}
