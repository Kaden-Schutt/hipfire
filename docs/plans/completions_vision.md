# Vision input for `/v1/chat/completions`

**Status:** draft (revised per adversarial reviews — GLM-5, Gemini, Claude)
**Scope:** Wire image inputs into the OpenAI-compatible HTTP serve path so
that clients (pi-coding-agent, Open WebUI, curl, etc.) can send images
alongside text in `messages[].content` and get VL inference from a loaded
Qwen3.5-VL model.

## Current state

The vision inference pipeline is **complete** end-to-end but only reachable
via `hipfire run --image <path>` (local daemon spawn). The HTTP serve path
(`/v1/chat/completions`) silently drops `image_url` content parts.

| Component | Location | Status |
|---|---|---|
| Image load, resize, normalize | `engine/src/image.rs` | Complete — filesystem paths only |
| Vision encoder (SigLIP-2 ViT + spatial merger) | `engine/src/qwen35_vl.rs` | Complete — GPU |
| Vision weight loading | `qwen35_vl::load_vision_weights` | Complete |
| VL prompt construction (ChatML + vision tokens) | `daemon.rs:2495-2509` | Complete — single image, vision tokens prepended to full prompt |
| Vision embedding injection at IMAGE_PAD positions | `qwen35::forward_scratch_embed` | Complete |
| Daemon IPC `image` field | `daemon.rs:579` | Complete — string path |
| `generate_vl()` dispatch | `daemon.rs:624-628` | Complete — missing thinking/DFlash |
| VL model detection + `vl` flag | `daemon.rs:960-974, 542` | Complete |
| `extractText()` multimodal content parsing | `cli/index.ts:1357` | Filters out image parts |
| HTTP → daemon image pass | `cli/index.ts:787, 1123` | **Missing** — hard `return false` + `useLocal` guard |
| Base64 / URL image decode | — | **Missing** |
| `image.rs` from-bytes variant | — | **Missing** |

## Design

### Approach: base64 over IPC, temp-file-free

Decode base64 in the TypeScript layer, pass raw base64 to the daemon over
the existing newline-delimited JSON IPC protocol, decode to bytes inside
the daemon, and call a new `load_and_preprocess_from_bytes()` function.

Base64-over-JSON is ~33% wire overhead vs raw bytes, but reuses the
existing IPC without protocol changes. For typical sub-10MB images the
overhead is sub-millisecond and the simplicity is worth it. Re-evaluate
if we add video or multi-image.

**Why not temp files?** Decoding base64 to a temp file in the TS layer
would keep the daemon unchanged but adds filesystem I/O, temp-file
cleanup, and path-length limits for no meaningful benefit — the daemon
can decode base64 in-memory faster than the disk round-trip.

**Why not raw bytes over stdin?** The IPC protocol is JSON lines.
Embedding raw binary would require a protocol format change (length-prefixed
binary frames or a separate FD).

### Content part format (OpenAI API)

```json
{
  "type": "image_url",
  "image_url": {
    "url": "data:image/png;base64,iVBORw0KGgo..."
  }
}
```

Only `data:` URIs with MIME type `image/png` or `image/jpeg` are supported.
HTTPS URLs and other MIME types are deferred — see §Postponed.

### In-scope limitations (Phase 1)

- **Single image per request.** Multiple images → HTTP 400.
- **Single-turn only.** Multi-turn conversations with images → HTTP 400.
  The daemon's `generate_vl` wraps the entire prompt in a single user-turn
  template with vision tokens at the front (daemon.rs:2495-2509). For
  multi-turn conversations, this places the image at the wrong user turn,
  producing degraded recall with no error or warning. Rejecting explicitly
  is safer than silently misaligning.
- **Images only in `role: "user"` messages.** Images in system, assistant,
  or tool messages are silently ignored.
- **PNG and JPEG only.** WebP, GIF, BMP, AVIF, TIFF are unsupported (the
  `image` crate is built with only `png` and `jpeg` features). Other
  formats → HTTP 400 with a clear message.

---

## Implementation steps

### Phase 1 — Core path (images work end-to-end)

#### 1.1 `engine/src/image.rs` — add `load_and_preprocess_from_bytes()`

Extract the shared resize+normalize work into a helper that takes a
`DynamicImage`:

```rust
fn preprocess_dynamic_image(
    img: image::DynamicImage,
    patch_size: usize,
    spatial_merge_size: usize,
) -> (Vec<f32>, usize, usize);
```

The existing `load_and_preprocess(path, ...)` calls `image::open(path)`
then delegates to this helper. The new `load_and_preprocess_from_bytes()`
calls `image::load_from_memory(data)` then delegates to the same helper.

```rust
pub fn load_and_preprocess_from_bytes(
    data: &[u8],
    patch_size: usize,
    spatial_merge_size: usize,
) -> Result<(Vec<f32>, usize, usize), String>;
```

Returns `Result` so callers can surface decode errors instead of
panicking. The existing `load_and_preprocess` continues to panic (fine
for CLI). Error variants:
- Unsupported format → `"unsupported image format — supported: png, jpeg"`
- Malformed data → `"failed to decode image: {e}"`
- Decompression bomb → `"image dimensions ({w}×{h}) exceed maximum ({max})"`

**Dimension ceiling check:** After `image::load_from_memory()` but before
`resize_exact()`, check `(img.width() * img.height()) > MAX_DIMENSION_PIXELS`
and return an error. This prevents decompression bombs from allocating
gigabytes before `smart_resize` clamps. Use `MAX_DIMENSION_PIXELS = 4_000_000`
(~4K × 4K, well above `smart_resize`'s `max_pixels = 1,003,520` target).

Add `base64 = "0.22"` dependency to `engine/Cargo.toml`.

#### 1.2 `daemon.rs` — accept `image_base64` in generate messages

Extend the generate dispatch (line 579 area) to accept a new IPC field:

```json
{"type": "generate", "id": "...", "prompt": "...", "image_base64": "iVBORw0KGgo..."}
```

When `image_base64` is present and the model has a vision encoder:
1. Validate payload size: reject if `image_base64.len() > MAX_BASE64_LEN`
   (default: 40 MB base64 string ≈ 30 MB raw bytes). Return error.
2. Decode the base64 string to raw bytes via the `base64` crate.
3. Strip the `data:image/...;base64,` prefix if present (belt-and-suspenders —
   the TS layer should strip it, but the daemon should be tolerant).
4. Call `load_and_preprocess_from_bytes()`.
5. Continue with the existing `extract_patches` → `vision_forward` →
   `generate_vl` flow.

Keep the existing `image` (filesystem path) field working for
backward compatibility with `hipfire run --image`.

**Routing logic at the dispatch point:**
```
if image_base64.is_some() && vision_config.is_some()  → generate_vl (from bytes)
if image.is_some() && vision_config.is_some()          → generate_vl (from path, existing)
if image_base64.is_some() && vision_config.is_none()   → error: "model has no vision encoder"
if image.is_some() && vision_config.is_none()           → error: "model has no vision encoder"
otherwise                                              → generate (text-only)
```

Note: the `image` path now also errors on non-VL models (matching the
`image_base64` behavior). This is a small breaking change for
`hipfire run --image` on text-only models, which previously degraded
silently. The CLI can show the error and exit instead.

**Precedence:** If both `image` and `image_base64` are present,
`image_base64` takes priority. Log a warning: `"both image and
image_base64 provided — using image_base64"`.

**`generate_vl` signature refactor:** The current function has 12
positional parameters (daemon.rs:2423), which is already too many.
Refactor to a struct before adding `image_base64`:

```rust
struct GenerateVLParams<'a> {
    id: &'a str,
    prompt: &'a str,
    system_prompt: Option<&'a str>,
    image_source: ImageSource<'a>,
    temp: f32,
    top_p: f32,
    max_tokens: usize,
    repeat_penalty: f32,
    repeat_window: usize,
}

enum ImageSource<'a> {
    Path(&'a str),
    Base64(&'a str),
}
```

**Capacity guard fix:** The current guard at daemon.rs:2427 uses the
hardcoded `IMAGE_SIZE = 448` constant to estimate visual tokens (~196),
but `smart_resize` can produce up to ~970 visual tokens for a typical
photo (~5× underestimate). Move the guard below the preprocess call
so it uses the actual `(img_h, img_w)` from `load_and_preprocess_from_bytes()`
or `load_and_preprocess()`. The preprocess decode is cheap relative to
the vision encoder forward pass.

**Log line:** The existing `eprintln!("[VL-DEBUG] preprocessing image: {}",
image_path)` at daemon.rs:2451 is wrong for the bytes path. Branch on
source: `"<{}-byte buffer>"` for bytes, `"path: {}"` for path.

#### 1.3 `cli/index.ts` — extract image content parts, pass to daemon

**a) Add `extractContent()` alongside existing `extractText()`.**

Keep `extractText` as a 1-line wrapper for the `tool` and `assistant`
branches that still need text-only extraction:

```typescript
const extractContent = (content: any): { text: string, images: string[] } => {
    if (typeof content === "string") return { text: content, images: [] };
    if (Array.isArray(content)) {
        const textParts: string[] = [];
        const images: string[] = [];
        for (const p of content) {
            if (p?.type === "text") textParts.push(p.text ?? "");
            else if (p?.type === "image_url" && p?.image_url?.url) {
                const url: string = p.image_url.url;
                if (url.startsWith("data:")) {
                    const mimeMatch = url.match(/^data:(image\/(png|jpeg));base64,/);
                    if (mimeMatch) {
                        const raw = url.slice(url.indexOf(",") + 1);
                        images.push(raw);
                    }
                    // Non-image data: URIs silently skipped
                }
                // https: URLs silently skipped (deferred)
            }
        }
        return { text: textParts.join(""), images };
    }
    return { text: String(content), images: [] };
};

const extractText = (content: any): string => extractContent(content).text;
```

Note: MIME type is validated at extraction time (`image/png` and
`image/jpeg` only). Non-image `data:` URIs are silently skipped.
The daemon receives clean base64 — no URI parsing needed.

**b) Update message iteration (lines 1403-1437).**

Collect images from user messages. Extend the existing loop:

```typescript
let requestImages: string[] = [];
let imageInLastUserTurn = false;
// ... inside the loop:
if (role === "user") {
    const content = extractContent(m.content);
    if (content.images.length > 0) {
        requestImages.push(...content.images);
        imageInLastUserTurn = true;
    } else {
        imageInLastUserTurn = false;
    }
    text = content.text;
    // ... existing convParts logic
}
```

**Multi-turn rejection:** After the loop, reject if images are present
in a multi-turn conversation:

```typescript
const hasPriorHistory = nonSystem.length > 1;
if (requestImages.length > 0 && hasPriorHistory && !imageInLastUserTurn) {
    return Response.json(
        { error: { message: "multi-turn vision is not supported — images must be in the last user message of a single-turn request", type: "invalid_request_error" } },
        { status: 400 },
    );
}
```

**Multi-image rejection:** If `requestImages.length > 1`, return HTTP 400:

```typescript
if (requestImages.length > 1) {
    return Response.json(
        { error: { message: "multiple images not supported — only one image per request", type: "invalid_request_error" } },
        { status: 400 },
    );
}
```

**c) Add `image_base64` to `genParams` (line 1528 area).**

```typescript
if (requestImages.length === 1) {
    if (!modelHasVL) {
        return Response.json(
            { error: { message: "model has no vision encoder", type: "invalid_request_error" } },
            { status: 400 },
        );
    }
    genParams.image_base64 = requestImages[0];
}
```

**d) Track `vl` capability from model load.**

Add `let modelHasVL = false;` to the `serve()` function scope.
Three write sites, one read site:

| Site | Line | Action |
|---|---|---|
| Warm-load | 1266-1281 | `modelHasVL = loadResult.vl === true` |
| Per-request reload | 1456-1473 | `modelHasVL = loadResult.vl === true` |
| Unload | 1457 | `modelHasVL = false` |
| Read | genParams build (step c) | `if (!modelHasVL) { ... }` |

**e) Remove image guards in `run()` and `runViaHttp()`.**

Two guards block the HTTP path for images:

1. `cli/index.ts:787` — `if (image) return false;` in `runViaHttp()`.
   Remove this guard.
2. `cli/index.ts:1123` — `const useLocal = ... \|\| image !== undefined;`
   Drop `image !== undefined` from this condition so that `hipfire run
   --image` can also proxy through a running serve daemon.

**f) Bump `requiredMaxSeq` headroom for visual tokens.**

At cli/index.ts:1451, the current headroom is `+1024` for prompt tokens.
A typical photo adds ~970 visual tokens. When `requestImages.length > 0`,
bump headroom:

```typescript
const visualHeadroom = requestImages.length > 0 ? 1024 : 0;
const requiredMaxSeq = Math.max(effective.max_seq, requestMaxTokens + 1024 + visualHeadroom);
```

**g) Error handling — HTTP status code mapping.**

Map daemon error types to appropriate HTTP status codes:

| Error type | HTTP status | Example message |
|---|---|---|
| Malformed base64 | 400 | `"failed to decode base64 image data"` |
| Unsupported image format | 400 | `"unsupported image format — supported: png, jpeg"` |
| Image too large | 413 | `"image payload exceeds maximum size (30 MB)"` |
| Image dimensions too large | 400 | `"image dimensions (WxH) exceed maximum"` |
| Model has no vision encoder | 400 | `"model has no vision encoder"` |
| Multiple images | 400 | `"multiple images not supported"` |
| Multi-turn vision | 400 | `"multi-turn vision is not supported"` |
| Vision encoder internal error | 500 | (forward daemon message) |

The daemon's `generate_vl` should return structured error messages
via `{"type":"error","message":"..."}` that the HTTP handler maps
to the appropriate status code. Parse error messages for known prefixes
to determine the status code; fall back to 500 for unrecognized errors.

#### 1.4 End-to-end tests

**Happy path — single-turn VL:**
```bash
hipfire serve qwen3.5:9b-vl

IMG_B64=$(base64 -w0 photo.png)
curl http://localhost:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "$(jq -n \
    --arg img "data:image/png;base64,$IMG_B64" \
    '{
      model: "qwen3.5:9b-vl",
      messages: [{
        role: "user",
        content: [
          { type: "text", text: "What is in this image?" },
          { type: "image_url", image_url: { url: $img } }
        ]
      }],
      max_tokens: 256
    }')"
```
Expected: VL model describes the image content.

**Error paths:**

```bash
# Non-VL model + image → 400
curl ... -d '{"model":"qwen3.5:9b","messages":[{"role":"user","content":[
  {"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]}]}'

# Malformed base64 → 400
curl ... -d '{"model":"qwen3.5:9b-vl","messages":[{"role":"user","content":[
  {"type":"image_url","image_url":{"url":"data:image/png;base64,!!invalid!!"}}]}]}'

# Multi-turn with image → 400
curl ... -d '{"model":"qwen3.5:9b-vl","messages":[
  {"role":"user","content":"hi"},
  {"role":"assistant","content":"hello"},
  {"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}},{"type":"text","text":"what?"}]}]}'

# Multiple images → 400
curl ... -d '{"model":"qwen3.5:9b-vl","messages":[{"role":"user","content":[
  {"type":"image_url","image_url":{"url":"data:image/png;base64,..."}},
  {"type":"image_url","image_url":{"url":"data:image/png;base64,..."}},
  {"type":"text","text":"compare"}]}]}'

# Unsupported format (WebP) → 400
curl ... -d '{"model":"qwen3.5:9b-vl","messages":[{"role":"user","content":[
  {"type":"image_url","image_url":{"url":"data:image/webp;base64,UklGR..."}}]}]}'

# Empty image_url.url → 400 or graceful skip
```

Note: the VL model name (`qwen3.5:9b-vl`) must be verified as a
pullable target before the test runs. Add to AGENTS.md pull targets if
not already present.

---

### Phase 2 — Feature parity with text-only path

These are not required for the initial ship but should be tracked.

#### 2.1 Thinking mode in `generate_vl`

`generate_vl()` (daemon.rs:2423) currently lacks `max_think_tokens`
support. The text-only `generate()` has full thinking-mode handling:
decoded-text scan via `rfind("<think>")` / `rfind("</think>")`,
force-emit `</think>\n` to close thinking with KV-write consistency,
budget tracking, attractor-block setup. This logic is ~60 lines spread
across 8 locations in `generate()` and is deeply coupled to the decode
loop's architecture.

Port strategy:

1. **Extract a `ThinkState` struct** (~20 lines) holding `think_count`,
   `prev_in_think`, with `update(&decoded_text) -> bool` and
   `should_force_close() -> bool` methods.
2. **Port the force-close injection** (~35 lines) into `generate_vl()`.
   This runs `</think>\n` tokens through `forward_scratch` to maintain KV
   cache consistency — `infer_vl.rs` has a simpler but **incorrect**
   implementation (no KV-write on force-close → hidden-state discontinuity).
   Use the daemon's approach as the reference.
3. **Wire `max_think_tokens`** through the IPC dispatch (already parsed
   at line 621-622) and the `GenerateVLParams` struct.

Estimated: 80-100 lines of new code. The `generate_vl()` decode loop uses
the CPU logits path (`download_f32` + `llama::sample_top_p`) while the
daemon's thinking enforcement uses GPU sampling — the thinking-state
tracking is independent of the sampling path, so this is a clean port.

**Phase 1 mitigation:** Without thinking mode, a VL thinking model can
consume the entire `max_tokens` budget inside thinking and return empty
content (replaying bug #74). As a cheap stopgap in Phase 1, hardcode
`max_think_tokens = 256` in the `generate_vl` dispatch when the client
doesn't specify one. This caps thinking without needing the full
extraction.

**Note (2026-05-07):** earlier revisions of this section described the
markers as `💭` / `💭\n` for both open and close — that was wrong (the
text path uses `<think>` / `</think>`). Implementation followed the
buggy plan literally; both the code and this doc were corrected.

#### 2.2 Multi-turn VL conversations

The daemon does **not** reset per HTTP request — it maintains cumulative
state (`seq_pos`, `conversation_tokens`, KV cache, DeltaNet state) across
requests. The HTTP handler sends `{"type":"reset"}` at the start of each
request (cli/index.ts:1344), clearing all prior context.

Multi-turn VL requires either:
- (a) Removing the per-request reset for VL-capable models, so the
  daemon's cumulative state is preserved across turns.
- (b) Building the multi-turn token sequence on the client side and
  sending it as a single prompt (which the OpenAI messages array already
  does — collapsed into a single ChatML string).
- (c) Supporting interleaved `<|vision_start|>...<|vision_end|>` blocks
  within the prompt token sequence, so the image is placed at the correct
  user turn position rather than always at the front.

Option (c) is the correct long-term solution. The daemon's `generate_vl`
prompt construction (daemon.rs:2495-2509) currently hardcodes a single
vision block at the top of the user turn.

#### 2.3 Streaming `usage` field

The streaming response does not include `usage` in any SSE chunk,
including the final one. The daemon's `done` message carries
`prefill_tokens` (which includes visual tokens), but the HTTP handler
doesn't emit it in the stream. The non-streaming path already maps
`prefill_tokens` to `usage.prompt_tokens` correctly — no change needed
there.

Add `usage: { prompt_tokens, completion_tokens, total_tokens }` to the
final streaming chunk before `[DONE]`.

---

### Postponed (noted for future work)

| Item | Why deferred |
|---|---|
| **HTTPS URL fetch** (`image_url.url` starts with `https://`) | Requires an HTTP client in the daemon or TS layer, proxy config, timeout handling, content-type validation. Non-trivial surface area. Log a warning: `"https image URLs not supported — use data: URIs"`. |
| **Multi-image per request** | `generate_vl()` hardcodes a single vision block. Supporting N images requires interleaved vision/text token construction and N× vision encoder calls. |
| **DFlash + VL** | `generate_vl()` has no speculative decoding path. VL prefill is already per-token (not batched WMMA). DFlash draft models are text-only. Low priority — VL perf is gated by vision encoder latency, not decode speed. |
| **`detail` parameter** (`"auto"`, `"low"`, `"high"`) | OpenAI's image content parts support a `detail` field controlling resolution. HuggingFace's `smart_resize` with different `max_pixels` could approximate this, but the mapping is non-obvious. Default to `"auto"` (current `smart_resize` behavior). |
| **WebP / other image formats** | The `image` crate is built with only `png` and `jpeg` features. WebP support can be added via the `webp` feature flag — low effort, but not blocking for Phase 1. |
| **Inline image in assistant responses** | Not applicable — hipfire generates text only. |
| **Vision-specific quantization for `image_base64` path** | Currently vision weights are always loaded in their stored format (F16 or HFQ4). The base64 path doesn't change this. |

---

## File change summary

| File | Change |
|---|---|
| `engine/src/image.rs` | Add `load_and_preprocess_from_bytes()` with `Result` return; extract shared `preprocess_dynamic_image(DynamicImage, ...)` helper; add dimension ceiling check; add format-specific error messages |
| `engine/Cargo.toml` | Add `base64 = "0.22"` dependency |
| `crates/hipfire-runtime/examples/daemon.rs` | Parse `image_base64` in generate dispatch; refactor `generate_vl` to take `GenerateVLParams` struct with `ImageSource` enum; decode base64 + from-bytes path; move capacity guard below preprocess; fix log line for bytes path; error on `image` + non-VL model (was silent-ignore) |
| `cli/index.ts` | Add `extractContent()` (keep `extractText` as wrapper); MIME validation on `data:` URIs; multi-turn VL rejection (400); multi-image rejection (400); track `modelHasVL` at 3 write sites; add `image_base64` to `genParams`; remove both image guards (787 + 1123); bump `requiredMaxSeq` for visual tokens; map daemon errors to HTTP status codes |

---

## Review cross-reference

This plan was revised to address findings from three adversarial reviews.
All items below were either incorporated, accepted as deferred, or
explicitly rejected with rationale.

### Blocking issues resolved

| ID | Finding | Resolution |
|---|---|---|
| GLM-5 A1 | `vl` flag silently discarded in serve path | §1.3d: explicit `modelHasVL` with 3 write sites + 1 read site |
| GLM-5 A2 + Claude B2 | `extractContent()` code example has shadowing bug | §1.3a: rewritten with correct variable names; `extractText` kept as wrapper |
| GLM-5 A3 | No base64 payload size limit | §1.2: `MAX_BASE64_LEN` validation in daemon; dimension ceiling in `image.rs` |
| Claude B1 | Multi-turn vision tokens bound to wrong user turn | §Design (limitations): multi-turn VL → HTTP 400; §1.3b: rejection logic |

### High-severity gaps resolved

| ID | Finding | Resolution |
|---|---|---|
| GLM-5 B1 + Claude S6 | Thinking-mode port more complex than stated | §2.1: re-spec'd with `ThinkState` extraction, 80-100 LOC estimate; Phase 1 mitigation (hardcode `max_think_tokens=256`) |
| GLM-5 B2 | Image format support gap (PNG/JPEG only) | §Design (limitations): explicit; §1.3a: MIME validation returns 400; §Postponed: WebP noted |
| GLM-5 B3 + Claude S4 | Both `image`+`image_base64` set behavior | §1.2: explicit precedence documented, warning logged |
| GLM-5 B4 | Streaming `usage` missing | §2.3: new tracking item |
| GLM-5 B5 | No MIME type validation | §1.3a: regex validated against `image/(png\|jpeg)` allowlist |
| GLM-5 B6 + Gemini §2.1 + Claude B4 | Approach section contradiction | §Design: rewritten — temp-file paragraph removed, argument honest |
| GLM-5 B7 | Test model not in AGENTS.md | §1.4: noted as prerequisite |
| Gemini §3.2 | Multi-image should 400, not warn | §1.3b: HTTP 400 with clear message |
| Claude B5 | Capacity guard underestimates ~5× | §1.2: guard moved below preprocess call |

### Medium concerns resolved

| ID | Finding | Resolution |
|---|---|---|
| GLM-5 C7 + Gemini §3.1 + Claude §2.3 | "Daemon resets per request" wrong | §2.2: corrected — HTTP handler resets, not daemon |
| GLM-5 C5 | Error-to-HTTP-status mapping | §1.3g: explicit mapping table |
| GLM-5 C6 + Claude B3 | Silent-ignore vs error asymmetry | §1.2: both paths now error consistently |
| Claude S7 | `requiredMaxSeq` missing visual tokens | §1.3f: headroom bump when images present |
| Claude S5 | `generate_vl` 12 positional params | §1.2: `GenerateVLParams` struct refactor |
| Claude M3 | E2E test only happy path | §1.4: 6 error-path tests added |
| Claude M4 | `run()` line 1123 also blocks | §1.3e: both guards removed |
| Gemini §3.3 | Dimension bomb before `smart_resize` | §1.1: dimension ceiling check |

### Items explicitly retained as-is or deferred

| ID | Finding | Decision |
|---|---|---|
| GLM-5 C1 | `prompt_normalize` interaction with VL | Verify during implementation — likely benign (normalization targets user text, not ChatML structure) |
| GLM-5 C3 | VL TTFT — no client indication | Defer — not blocking; document in API docs |
| GLM-5 C4 | Panic vs Result inconsistency in CLI path | Accept for CLI; only `load_and_preprocess_from_bytes` returns `Result` |
| Claude S2 | `image_base64` field naming | Keep `image_base64` for now; rename on next protocol bump if needed |
| Claude S3 | Image position loss in `extractContent` | Accept for Phase 1 (single image); placeholder marker deferred |
| Claude M5 | `IMAGE_SIZE` constant vestigial | Tied to capacity guard fix in §1.2 — remove if guard no longer uses it |
| Claude M7 | `VISION_*_ID` hardcoded | Accept for now; comment noting Qwen3.5-VL specificity. Lowest priority. |
| Claude M8 | "Why not raw bytes" argument | §Design: rewritten |
| Claude M9 | Helper signature explicit | §1.1: `preprocess_dynamic_image(DynamicImage, ...)` specified |

### Items explicitly rejected

| ID | Finding | Rejection rationale |
|---|---|---|
| Gemini §2.2 | IPC pipe blocking from concurrent sends | `Engine.send()` is `async`+`await`ed — no concurrent pipe writes. Event-loop queuing is standard behavior, not a hazard. |
| Gemini §2.3 + Claude S6 | Move thinking mode to Phase 1 | Gap is real but Phase 2 is the right placement (80-100 LOC of coupled code). Runaway risk bounded by `max_tokens`. Cheap Phase 1 mitigation: hardcode default. |
| Gemini §2.4 | Commas in base64 corrupt URI parsing | Base64 alphabet (A-Z, a-z, 0-9, +, /, =) has no commas. Comma-search is safe. |
| Claude B5 | "20× underestimate" on capacity guard | Corrected: actual ~5× (950/196 ≈ 4.9). Principle valid, multiplier wrong. |
