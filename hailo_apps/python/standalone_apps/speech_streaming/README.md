# Speech Streaming — Hailo-8/8L/10H

Continuous Whisper speech-to-text that runs on **all Hailo accelerators** (Hailo-8, Hailo-8L, Hailo-10H).

Reads raw PCM audio (s16le, 16kHz, mono) from stdin, transcribes in chunk-sized windows (5s or 10s depending on model), and prints each chunk's text to stdout immediately.

Optionally exposes a Server-Sent Events endpoint so HTTP clients can stream transcription results in real time.

Unlike `simple_whisper_chat` (H10-only, uses `hailo_platform.genai.Speech2Text`), this app uses the low-level HailoRT `InferModel` API with separate encoder/decoder HEFs — compatible with all Hailo devices.

## Prerequisites

- Hailo-8, Hailo-8L, or Hailo-10H accelerator
- Python 3.10+, HailoRT 4.20+
- `ffmpeg` (for file-to-PCM conversion): `sudo apt install ffmpeg`

## Installation

```bash
pip install -e ".[speech-rec]"
```

Models (HEF files and decoder assets) are managed by the repo's central resource system
(`resources_config.yaml`) and **auto-downloaded on first run** via `resolve_hef_paths()`.

## Usage

**From a microphone (via arecord):**
```bash
arecord -f S16_LE -r 16000 -c 1 -t raw | \
    python -m hailo_apps.python.standalone_apps.speech_streaming.speech_streaming
```

**From an audio/video file (via ffmpeg):**
```bash
ffmpeg -i input.mp4 -f s16le -ac 1 -ar 16000 - | \
    python -m hailo_apps.python.standalone_apps.speech_streaming.speech_streaming
```

**With SSE API enabled:**
```bash
arecord -f S16_LE -r 16000 -c 1 -t raw | \
    python -m hailo_apps.python.standalone_apps.speech_streaming.speech_streaming \
    --api --port 5001

# Then from another terminal:
curl -N http://localhost:5001/events
```

**Save transcription to a file:**
```bash
ffmpeg -i lecture.mp3 -f s16le -ac 1 -ar 16000 - | \
    python -m hailo_apps.python.standalone_apps.speech_streaming.speech_streaming \
    > transcript.txt
```

### Options

| Flag | Description |
|---|---|
| `--overlap SECONDS` | Overlap between consecutive chunks (default: 2.0, 0 to disable). Prevents word-boundary gibberish by ensuring every word appears fully in at least one chunk. |
| `--normalize` | Peak-normalize each chunk to 0.9. Useful for quiet sources like microphones. Off by default — well-leveled sources (radio, files) don't need it. |
| `--api` | Start SSE server for streaming results to HTTP clients |
| `--port PORT` | SSE server port (default: 5001, requires `--api`) |
| `--arch {hailo8,hailo8l,hailo10h}` | Target architecture (auto-detected if omitted) |
| `--variant {base,tiny,tiny.en}` | Whisper variant (default: `base`) |
| `--list-models` | List available models and exit |

## Supported Models

| Variant | Hailo-8 | Hailo-8L | Hailo-10H |
|---|---|---|---|
| `base` | 5s chunks | 5s chunks | 10s chunks |
| `tiny` | 10s chunks | 10s chunks | 10s chunks |
| `tiny.en` | — | — | 10s chunks |

## stdin Format

Raw PCM, signed 16-bit little-endian, 16 kHz, mono. This is the format produced by:
- `arecord -f S16_LE -r 16000 -c 1 -t raw`
- `ffmpeg -i <input> -f s16le -ac 1 -ar 16000 -`

## Output

- **stdout**: Transcription text only (one chunk per line). Safe to pipe or redirect.
- **stderr**: Diagnostic messages (model info, progress, errors).
