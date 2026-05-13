"""
Streaming Speech-to-Text for Hailo-8/8L/10H.

Reads raw PCM audio (s16le, 16kHz, mono) from stdin, transcribes in chunks
using Whisper on any Hailo accelerator, and prints text to stdout immediately.

Optionally exposes a Server-Sent Events endpoint for HTTP clients.

Usage:
    # From a microphone (via arecord):
    arecord -f S16_LE -r 16000 -c 1 -t raw | \
        python -m hailo_apps.python.standalone_apps.speech_streaming.speech_streaming

    # From an audio/video file (via ffmpeg):
    ffmpeg -i input.mp4 -f s16le -ac 1 -ar 16000 - | \
        python -m hailo_apps.python.standalone_apps.speech_streaming.speech_streaming

    # With SSE API:
    arecord -f S16_LE -r 16000 -c 1 -t raw | \
        python -m hailo_apps.python.standalone_apps.speech_streaming.speech_streaming \
        --api --port 5001

    # List available models:
    python -m hailo_apps.python.standalone_apps.speech_streaming.speech_streaming \
        --list-models
"""

import argparse
import logging
import os
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np


def check_dependencies():
    """Exit with instructions if required packages are missing."""
    missing = []
    for dep in ["torch", "transformers"]:
        try:
            __import__(dep)
        except ImportError:
            missing.append(dep)
    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}", file=sys.stderr)
        print(
            "\nRun the following command from the 'hailo-apps' repository root directory "
            "(where pyproject.toml is located):",
            file=sys.stderr,
        )
        print('  pip install -e ".[speech-rec]"', file=sys.stderr)
        sys.exit(1)


def _setup_imports():
    """Handle imports with fallback for dev-mode."""
    try:
        from hailo_apps.python.core.common.toolbox import resolve_arch
        from hailo_apps.python.core.common.hailo_logger import get_logger
        from hailo_apps.python.core.common.core import resolve_hef_paths
        from hailo_apps.python.core.common.defines import (
            WHISPER_H8_APP, RESOURCES_ROOT_PATH_DEFAULT,
            RESOURCES_NPY_DIR_NAME,
        )
        return resolve_arch, get_logger, resolve_hef_paths, WHISPER_H8_APP, \
            RESOURCES_ROOT_PATH_DEFAULT, RESOURCES_NPY_DIR_NAME
    except ImportError:
        repo_root = None
        for p in Path(__file__).resolve().parents:
            if (p / "hailo_apps" / "config" / "config_manager.py").exists():
                repo_root = p
                break
        if repo_root:
            sys.path.insert(0, str(repo_root))
        from hailo_apps.python.core.common.toolbox import resolve_arch
        from hailo_apps.python.core.common.hailo_logger import get_logger
        from hailo_apps.python.core.common.core import resolve_hef_paths
        from hailo_apps.python.core.common.defines import (
            WHISPER_H8_APP, RESOURCES_ROOT_PATH_DEFAULT,
            RESOURCES_NPY_DIR_NAME,
        )
        return resolve_arch, get_logger, resolve_hef_paths, WHISPER_H8_APP, \
            RESOURCES_ROOT_PATH_DEFAULT, RESOURCES_NPY_DIR_NAME


SAMPLE_RATE = 16000


# --- Variant-to-model-name mapping ---
VARIANT_MODELS = {
    "base": {
        "hailo8": (
            "base-whisper-encoder-5s",
            "base-whisper-decoder-fixed-sequence-matmul-split",
        ),
        "hailo8l": (
            "base-whisper-encoder-5s_h8l",
            "base-whisper-decoder-fixed-sequence-matmul-split_h8l",
        ),
        "hailo10h": (
            "base-whisper-encoder-10s",
            "base-whisper-decoder-10s-out-seq-64",
        ),
    },
    "tiny": {
        "hailo8": (
            "tiny-whisper-encoder-10s_15dB",
            "tiny-whisper-decoder-fixed-sequence-matmul-split",
        ),
        "hailo8l": (
            "tiny-whisper-encoder-10s_15dB_h8l",
            "tiny-whisper-decoder-fixed-sequence-matmul-split_h8l",
        ),
        "hailo10h": (
            "tiny-whisper-encoder-10s",
            "tiny-whisper-decoder-fixed-sequence",
        ),
    },
    "tiny.en": {
        "hailo10h": (
            "tiny_en-whisper-encoder-10s",
            "tiny_en-whisper-decoder-fixed-sequence",
        ),
    },
}


def _ensure_npy_assets(variant, npy_dir, app_name, arch, resources_root):
    """Check that decoder npy assets exist; auto-download if missing."""
    needed = [
        f"token_embedding_weight_{variant}.npy",
        f"onnx_add_input_{variant}.npy",
    ]
    missing = [f for f in needed if not (Path(npy_dir) / f).exists()]
    if not missing:
        return

    print(f"\nDecoder tokenization assets not found for variant '{variant}'.", file=sys.stderr)
    print("   Downloading automatically...\n", file=sys.stderr)

    try:
        from hailo_apps.installation.download_resources import (
            ResourceDownloader, load_config, DEFAULT_RESOURCES_CONFIG_PATH,
        )
        config = load_config(Path(DEFAULT_RESOURCES_CONFIG_PATH))
        downloader = ResourceDownloader(
            config=config,
            hailo_arch=arch,
            resource_root=Path(resources_root),
        )
        downloader.collect_npy_by_tag(app_name)
        downloader.execute(parallel=False)
    except Exception as e:
        print(f"   Download failed: {e}", file=sys.stderr)

    still_missing = [f for f in needed if not (Path(npy_dir) / f).exists()]
    if still_missing:
        print(f"\nDecoder assets still missing: {still_missing}", file=sys.stderr)
        print(f"   Expected in: {npy_dir}", file=sys.stderr)
        print("   Try running the full resource download:", file=sys.stderr)
        print(
            f"   python -m hailo_apps.installation.download_resources --group {app_name}\n",
            file=sys.stderr,
        )
        sys.exit(1)


# --- Stdin streaming ---


def _read_stdin_chunks(chunk_bytes: int):
    """Yield exactly chunk_bytes from stdin. Final partial chunk is zero-padded."""
    buf = b""
    while True:
        needed = chunk_bytes - len(buf)
        data = sys.stdin.buffer.read(needed)
        if not data:
            if buf:
                buf += b"\0" * (chunk_bytes - len(buf))
                yield buf
            return
        buf += data
        if len(buf) >= chunk_bytes:
            yield buf[:chunk_bytes]
            buf = buf[chunk_bytes:]


def _pcm_to_mel(pcm_bytes: bytes, chunk_length: int, pad_or_trim, log_mel_spectrogram):
    """Convert raw s16le PCM bytes to a mel spectrogram for the encoder.

    Returns None if the chunk is silence (below RMS threshold).
    """
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 0.005:
        return None

    peak = np.max(np.abs(audio))
    if peak > 1e-6:
        audio = audio * (0.9 / peak)

    audio = pad_or_trim(audio, int(chunk_length * SAMPLE_RATE))

    mel = log_mel_spectrogram(audio).to("cpu")
    mel = np.expand_dims(mel, axis=0)
    mel = np.expand_dims(mel, axis=2)
    mel = np.transpose(mel, [0, 2, 3, 1])
    return mel


# --- SSE broadcast ---


class TranscriptionBroadcast:
    """Thread-safe fan-out of transcription strings to SSE listeners."""

    def __init__(self):
        self._lock = threading.Lock()
        self._listeners: list[queue.Queue] = []

    def subscribe(self) -> queue.Queue:
        q = queue.Queue()
        with self._lock:
            self._listeners.append(q)
        return q

    def unsubscribe(self, q: queue.Queue):
        with self._lock:
            try:
                self._listeners.remove(q)
            except ValueError:
                pass

    def publish(self, text: str):
        with self._lock:
            for q in self._listeners:
                q.put(text)


def _start_sse_server(broadcast: TranscriptionBroadcast, port: int):
    """Start a Flask SSE server on a daemon thread."""
    from flask import Flask, Response

    werkzeug_logger = logging.getLogger("werkzeug")
    werkzeug_logger.setLevel(logging.CRITICAL)
    werkzeug_logger.disabled = True

    app = Flask(__name__)
    app.logger.setLevel(logging.ERROR)
    app.logger.disabled = True

    @app.route("/events")
    def events():
        def generate():
            q = broadcast.subscribe()
            try:
                while True:
                    text = q.get()
                    yield f"data: {text}\n\n"
            except GeneratorExit:
                broadcast.unsubscribe(q)

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    def run_server():
        try:
            app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False, threaded=True)
        except OSError as e:
            if "Address already in use" in str(e):
                print(f"SSE server: port {port} already in use.", file=sys.stderr)
            else:
                print(f"SSE server error: {e}", file=sys.stderr)

    t = threading.Thread(target=run_server, daemon=True)
    t.start()
    time.sleep(0.1)
    return t


# --- CLI ---


def get_args():
    parser = argparse.ArgumentParser(
        description="Streaming Speech-to-Text (Hailo-8/8L/10H)",
    )
    parser.add_argument(
        "--arch", type=str, default=None,
        choices=["hailo8", "hailo8l", "hailo10h"],
        help="Target architecture (auto-detected if omitted)",
    )
    parser.add_argument(
        "--variant", type=str, default="base",
        choices=["base", "tiny", "tiny.en"],
        help="Whisper model variant (default: base)",
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="List available models and exit",
    )
    parser.add_argument(
        "--api", action="store_true",
        help="Start SSE server for streaming results to HTTP clients",
    )
    parser.add_argument(
        "--port", type=int, default=5001,
        help="SSE server port (default: 5001, requires --api)",
    )
    return parser.parse_args()


# --- Main ---


def main():
    (resolve_arch, get_logger, resolve_hef_paths, WHISPER_H8_APP,
     RESOURCES_ROOT, NPY_DIR) = _setup_imports()
    logger = get_logger(__name__)

    args = get_args()
    arch = resolve_arch(args.arch)
    variant = args.variant

    if args.list_models:
        from hailo_apps.python.core.common.core import handle_list_models_flag
        handle_list_models_flag(args, WHISPER_H8_APP)
        return

    check_dependencies()

    try:
        encoder_name, decoder_name = VARIANT_MODELS[variant][arch]
    except KeyError:
        available = list(VARIANT_MODELS.get(variant, {}).keys())
        logger.error(
            f"No models for variant='{variant}' arch='{arch}'. "
            f"Available archs: {available}"
        )
        sys.exit(1)

    resolved = resolve_hef_paths(
        hef_paths=[encoder_name, decoder_name],
        app_name=WHISPER_H8_APP,
        arch=arch,
    )
    encoder_path = str(resolved[0].path)
    decoder_path = str(resolved[1].path)

    print(f"Architecture: {arch}", file=sys.stderr)
    print(f"Variant: Whisper {variant}", file=sys.stderr)
    print(f"Encoder: {encoder_path}", file=sys.stderr)
    print(f"Decoder: {decoder_path}", file=sys.stderr)

    npy_dir = Path(RESOURCES_ROOT) / NPY_DIR
    _ensure_npy_assets(variant, npy_dir, WHISPER_H8_APP, arch, RESOURCES_ROOT)

    try:
        from .whisper_pipeline import WhisperPipeline
        from .audio_utils import pad_or_trim, log_mel_spectrogram
        from .postprocessing import clean_transcription
    except ImportError:
        from whisper_pipeline import WhisperPipeline
        from audio_utils import pad_or_trim, log_mel_spectrogram
        from postprocessing import clean_transcription

    print("\nInitializing Whisper pipeline...", file=sys.stderr)
    add_embed = arch in ("hailo8", "hailo8l")
    pipeline = WhisperPipeline(
        encoder_path, decoder_path, variant=variant, npy_dir=str(npy_dir),
        add_embed=add_embed,
    )
    chunk_length = pipeline.get_chunk_length()
    chunk_bytes = chunk_length * SAMPLE_RATE * 2
    print(f"Ready (chunk length: {chunk_length}s, {chunk_bytes} bytes per chunk)", file=sys.stderr)

    broadcast = None
    if args.api:
        broadcast = TranscriptionBroadcast()
        _start_sse_server(broadcast, args.port)
        print(f"SSE endpoint: http://0.0.0.0:{args.port}/events", file=sys.stderr)

    print("Reading audio from stdin...", file=sys.stderr)

    try:
        for pcm_chunk in _read_stdin_chunks(chunk_bytes):
            mel = _pcm_to_mel(pcm_chunk, chunk_length, pad_or_trim, log_mel_spectrogram)
            if mel is None:
                continue
            pipeline.send_data(mel)
            text = pipeline.get_transcription()
            text = clean_transcription(text)
            if text.strip():
                print(text, flush=True)
                if broadcast is not None:
                    broadcast.publish(text)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
    finally:
        pipeline.stop()
        print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
