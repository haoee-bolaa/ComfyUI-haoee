# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A ComfyUI custom-nodes plugin that exposes the **好易智算 (Haoee MaaS)** API as ComfyUI nodes — image generation, video generation, and LLM text completion. Users supply a Haoee API key (obtained from https://www.haoee.com/maas/services) inside a workflow, and each node call hits `https://maas.haoee.com` and is billed to that key.

The repo is published as the Comfy registry package `zhenzhen` (PublisherId `t8star`), but the in-repo code is the Haoee integration.

## Commands

There is no build/lint/test step. This is a ComfyUI extension — it is loaded by a ComfyUI host, not run standalone. To exercise it, drop the repo folder into ComfyUI's `custom_nodes/` directory and restart ComfyUI.

```bash
pip install -r requirements.txt        # install runtime deps
```

There are no tests in the repo; do not invent test commands. Verify changes by loading the nodes in ComfyUI and running a workflow.

## Architecture

### Single-file node catalog: `Comfly.py`

All nodes live in one large file (`Comfly.py`, ~3.5k lines). Every node class follows the same contract:

- `NODE_NAME` — short slug used in log/error prefixes (e.g. `"MiniMax"`, `"Doubao"`).
- `INPUT_TYPES` — `required` + `optional` widgets. The API key is passed as a `STRING` input (usually named `api_key`), sourced from the `好易 API Key` node upstream.
- `RETURN_TYPES` / `RETURN_NAMES` — video nodes return `(IO.VIDEO, "STRING", "STRING")` = (video, task_id, response_json); image nodes return `("IMAGE", "STRING", "STRING")`; text nodes return `("STRING", "STRING")`.
- `FUNCTION` — the entry method (`generate_video` / `generate_image` / `completions`).
- `CATEGORY` — Chinese category tree under `好易` (`好易/Video`, `好易/Image`, `好易/Text`).

Nodes register at the bottom of `Comfly.py` in `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS`. **When adding or removing a node, update both mappings and keep display names in Chinese** (e.g. `好易 视频 Kling`). The changelog at the top of `README.md` is the canonical record of node/model add/remove changes — mirror significant model-list changes there.

### API call pattern (video nodes)

Video generation is async on the Haoee side. Every video node follows the same flow, exemplified by `Comfly_HaoeeVideo_MiniMax`:

1. Build headers with `Authorization: Bearer <key>` and a `modelName` header.
2. `POST` the create payload (image input is base64-encoded PNG via `_image_tensor_to_base64`), capture `task_id`.
3. Poll the query endpoint every 10s until `status == "success"` or `HAOEE_POLL_TOTAL_TIMEOUT_SEC` (600s) elapses. Report progress through `comfy.utils.ProgressBar(100)`.
4. On success, return a `ComflyVideoAdapter` wrapping the `download_url` (or local path).

The create/poll endpoint paths and payload shapes differ per provider (MiniMax uses `/v1/video_generation` + `/v1/query/video_generation`; Doubao/Kling/Sora2/Wan/Grok/Seedance each use their own `/api/vN/...` routes) — copy the exact URL constants from the existing node when adding a sibling.

### `ComflyVideoAdapter`

Returned as the `IO.VIDEO` output. Accepts either a remote URL or a local path. `save_to()` downloads the URL and runs an **ffmpeg `+faststart` remux** (moov atom to front) so the video is seekable with duration/thumbnail. ffmpeg is resolved via `folder_paths.get_ffmpeg_path()` then `shutil.which`; if absent it falls back to a plain copy. `get_dimensions()` reads frame size via OpenCV (`cv2`) for local files, and returns a hardcoded `1280x720` for URLs.

### Error handling: `HaoeeNodeError` + `_haoee_raise_*` helpers

This is the most important convention to preserve. All node failures go through typed raisers defined mid-file:

- `_haoee_raise_local` — local validation failures (missing api_key, missing image).
- `_haoee_raise_http` — non-200 HTTP responses (includes status, hint, truncated body).
- `_haoee_raise_api` — remote task reported `fail`/`failed`.
- `_haoee_raise_network` — `requests.exceptions.RequestException`.
- `_haoee_raise_parse` — missing expected fields in a JSON response.

Each builds a message prefixed `[<NODE_NAME>][CATEGORY] ...` and raises `HaoeeNodeError`. The node's outer `try/except` **must re-raise `HaoeeNodeError` as-is** (the `except HaoeeNodeError: raise` clause) — this prevents double-prefixing when errors nest. Any other exception is wrapped via `_haoee_raise_local(..., f"unexpected: ...")` after `traceback.print_exc()`.

### Logging helpers (keep base64 out of the console)

`_haoee_log_http_request` / `_haoee_log_http_response` log request/response bodies with **base64 image payloads stripped** to `<base64 len=N>`. The detection (`_haoee_is_base64_string`, `_haoee_is_media_field_key`, `_haoee_replace_base64_in_json`) walks JSON and replaces any base64-looking string in known media field keys. Set env var `HAOEE_LOG_FULL_BODY=1` to log raw bodies (capped at 50k chars). Always route HTTP logging through these helpers — never `print(response.text)` directly.

### Image / tensor conversion: `utils.py`

`pil2tensor` / `tensor2pil` mirror ComfyUI's own conversions (tensors normalized to `[0,1]`, shape `[B,H,W,3]`). Image input widgets use `IMAGE`; nodes convert to base64 PNG via `_image_tensor_to_base64`. Image-returning nodes parse base64/URL responses from the API and build `IMAGE` tensors.

### Web extensions: `web/`

`WEB_DIRECTORY = "./web"` (set in `__init__.py`). JS files register ComfyUI frontend extensions via `app.registerExtension`. Currently used to make widget combos **dynamic** — e.g. `minimax_options.js` rewrites the `resolution`/`duration` option lists based on the selected `model`, and `seedream_size.js` does similar for the Seedream size widget. When a node's valid option values depend on another widget's value, implement it here rather than hardcoding static combos.

## Conventions

- Module re-export in `__init__.py` is `from .Comfly import ...` — the filename is `Comfly.py` (capital C). Don't rename it without updating the import.
- The Haoee base URL is the module global `baseurl = "https://maas.haoee.com"`. Endpoint paths are built from it.
- HTTP timeouts: single-request `HAOEE_HTTP_TIMEOUT_SEC = 600`, total poll budget `HAOEE_POLL_TOTAL_TIMEOUT_SEC = 600`. Reuse these globals rather than hardcoding.
- Display names and categories are in Chinese; match the existing tone when naming new nodes.
- `.codegraph/` is present — use codegraph (`codegraph_explore` / `codegraph node`) to navigate `Comfly.py` rather than reading the whole file.
