# SuperTonic 3 TTS — Dockerized (Rust)

[![build](https://github.com/bl4ck-bird/supertonic-tts-docker/actions/workflows/docker.yml/badge.svg)](https://github.com/bl4ck-bird/supertonic-tts-docker/actions/workflows/docker.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![ghcr.io](https://img.shields.io/badge/ghcr.io-supertonic-blue?logo=docker)](https://github.com/bl4ck-bird/supertonic-tts-docker/pkgs/container/supertonic)

A tiny, OpenAI-compatible Text-to-Speech server for
[SuperTonic 3](https://github.com/supertone-inc/supertonic), built in Rust
(Axum) on the upstream ONNX inference code.

## Features

- **OpenAI-compatible** `/v1/audio/speech`, plus native `/v1/tts` (full parameter
  set) and `/v1/tts/batch`.
- **Swagger** at `/docs` (generated from the code via `utoipa`) and an interactive
  test page at `/`.
- **Tiny image** — Alpine + musl with ONNX Runtime loaded as a shared library.

| Tag | Formats |
| --- | --- |
| `supertonic:latest` | wav, pcm |
| `supertonic:ffmpeg` | wav, pcm, mp3, opus, aac, flac, ogg |

## Quick Start

```bash
docker run -p 8080:8080 -v "$(pwd)/assets:/assets" \
  ghcr.io/bl4ck-bird/supertonic:latest        # or :ffmpeg for compressed formats
```

Assets (~400MB: `onnx/` model + `voice_styles/`) download into `./assets` on first
start (in-process, verified against the Hugging Face manifest) and are reused
afterwards. Then open the test console at <http://localhost:8080/>, Swagger at
`/docs`, or health at `/v1/health`.

Images publish to GHCR on every push to `main`: `:latest` / `:ffmpeg` track the
newest build, and each is also tagged by date (`:YYYY.MM.DD`, `-ffmpeg`) to pin.

## Usage

### REST API

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/v1/audio/speech` | OpenAI-compatible synthesis |
| `POST` | `/v1/tts` | Native synthesis (full parameter set) |
| `POST` | `/v1/tts/batch` | Batch synthesis (base64 audio per item) |
| `GET`  | `/v1/voices` | List voices (`M1`–`M5`, `F1`–`F5`) |
| `POST` | `/v1/voices/import` | Register a custom Voice Builder style |
| `GET`  | `/v1/health` | Health + capabilities |
| `GET`  | `/openapi.json`, `/docs` | OpenAPI schema + Swagger UI |

### OpenAI client

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")
client.audio.speech.create(
    model="tts-1", voice="M1", input="SuperTonic is lightning fast."
).stream_to_file("out.wav")
```

Use a preset voice (`M1`–`M5`, `F1`–`F5`); OpenAI names (`alloy`, …) are not aliased.

### Native endpoint

```bash
curl -X POST http://localhost:8080/v1/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"안녕하세요","voice":"F1","lang":"ko","steps":8,"speed":1.05}' \
  -o out.wav
```

| Parameter | Range | Default |
| --- | --- | --- |
| `voice` | `M1`–`M5`, `F1`–`F5` | `M1` |
| `lang` | 31 ISO codes + `na` | `na` |
| `steps` | 1–100 (useful 5–12) | 8 |
| `speed` | 0.7–2.0 | 1.05 |
| `silence_duration` | 0.0–10.0 | 0.3 |
| `max_chunk_length` | 1–10000 (bytes) | — |
| `response_format` | `wav`, `pcm` (+ compressed on `:ffmpeg`) | `wav` |

A compressed format on the minimal image returns `501`. Text is chunked
internally (≤300 bytes, ≤120 for ko/ja); `max_chunk_length` forces a smaller
pre-split, with `silence_duration` inserted between pieces.

### Custom voices

```bash
curl -X POST http://localhost:8080/v1/voices/import \
  -H 'Content-Type: application/json' \
  -d '{"name":"my_voice","style_ttl":{...},"style_dp":{...}}'
```

`style_ttl` / `style_dp` are the Voice Builder JSON components. The voice is saved
to the assets volume and usable as `voice: "my_voice"`. An import never overwrites
an existing voice (preset or custom) — delete its file to replace it; the CLI
`import` follows the same rule.

### Command line

The same binary doubles as a CLI (no subcommand = the HTTP server):

```bash
docker run --rm -v "$(pwd)/assets:/assets" -v "$(pwd):/out" \
  ghcr.io/bl4ck-bird/supertonic:latest synth --text "Hello" --out /out/hello.wav
```

Also `voices` (list) and `import --name <n> --file <f>`. Run `synth --help` for
the full flag set.

## Configuration

| Variable | Description |
| --- | --- |
| `SUPERTONIC_PORT` | Port the server binds (default `8080`). Under Compose it is also the published host port but is not injected into the container, so the container there stays on 8080. |
| `SUPERTONIC_THREADS` | `OMP_NUM_THREADS` for ONNX Runtime. |
| `SUPERTONIC_HF_REPO` / `SUPERTONIC_HF_REVISION` | Model repo / revision (default: a pinned commit; set to `main` or a tag to track upstream). |
| `SUPERTONIC_MAX_TEXT_BYTES` | Per-utterance input cap (default `4096`). |
| `SUPERTONIC_MAX_BATCH_TEXT_BYTES` | Per-batch total input cap (default `50000`). |
| `SUPERTONIC_FFMPEG_TIMEOUT_SECS` | Kill a stuck ffmpeg after N seconds (default `120`; `:ffmpeg` only). |
| `SUPERTONIC_MEM_LIMIT` | Compose memory limit (default `2g`). |
| `SUPERTONIC_UID` / `SUPERTONIC_GID` | Compose run user (default: host UID/GID if exported, else root). |

Input caps bound peak memory (synthesized audio dwarfs the text); raise them and
`SUPERTONIC_MEM_LIMIT` together for longer single-shot synthesis.

**Run as non-root** via Compose with
`SUPERTONIC_UID=$(id -u) SUPERTONIC_GID=$(id -g) docker compose up`. The chosen
user must be able to write `./assets`; if it cannot and a download is needed, the
server exits early with a clear message (`chown` the dir, or pre-populate and
mount read-only).

**Custom assets**: point `SUPERTONIC_HF_REPO`/`SUPERTONIC_HF_REVISION` at another
repo, or drop your own files into `./assets/onnx/` + `./assets/voice_styles/` (a
`.<dir>-complete` marker skips downloading that dir).

**Limits**: body 16 MiB (→`413`), request timeout 300s (→`408`), batch ≤100 items
& ≤50,000 bytes (→`400`), per-utterance ≤4,096 bytes and invalid `lang` (→`400`).

## Build from source

```bash
docker compose up --build                        # minimal (wav/pcm)
docker compose -f compose-ffmpeg.yml up --build  # with ffmpeg (compressed)
```

## Security

- The API is **unauthenticated** — keep it behind a reverse proxy / private
  network, not exposed directly to the internet.
- Downloads are over HTTPS and verified against the Hugging Face manifest before
  use; files you pre-populate yourself are trusted by size.

## Licenses

- **Server code** (this repository): [MIT](LICENSE).
- **Inference code** (`src/helper.rs`, vendored from
  [`supertone-inc/supertonic`](https://github.com/supertone-inc/supertonic) @
  `v3.0.0`): MIT — see `THIRD_PARTY_LICENSES`.
- **Model weights** (downloaded from
  [`Supertone/supertonic-3`](https://huggingface.co/Supertone/supertonic-3) at
  runtime): **OpenRAIL-M**, with use-based restrictions — review the model card.
