from __future__ import annotations

import subprocess
import threading
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np


SUPPORTED_FORMATS = frozenset({"mp3", "opus", "aac", "flac", "wav", "pcm"})
CONTENT_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "application/octet-stream",
}


class EncodingError(RuntimeError):
    pass


def pcm16_bytes(audio: np.ndarray) -> bytes:
    values = np.asarray(audio).reshape(-1)
    if values.dtype == np.int16:
        return values.tobytes()
    if np.issubdtype(values.dtype, np.floating):
        peak = float(np.max(np.abs(values))) if values.size else 0.0
        if peak <= 1.5:
            values = values * 32767.0
    return np.clip(values, -32768, 32767).astype(np.int16).tobytes()


def _ffmpeg_output_args(response_format: str) -> list[str]:
    if response_format == "mp3":
        return ["-c:a", "libmp3lame", "-b:a", "128k", "-write_xing", "0", "-f", "mp3"]
    if response_format == "opus":
        return ["-c:a", "libopus", "-b:a", "96k", "-f", "ogg"]
    if response_format == "aac":
        return ["-c:a", "aac", "-b:a", "128k", "-f", "adts"]
    if response_format == "flac":
        return ["-c:a", "flac", "-f", "flac"]
    if response_format == "wav":
        return ["-c:a", "pcm_s16le", "-f", "wav"]
    raise EncodingError(f"unsupported encoded format: {response_format}")


class AudioEncoder:
    """Encode PCM fragments to an HTTP-safe byte stream without temporary files."""

    def __init__(self, ffmpeg_path: Path, read_size: int = 64 * 1024) -> None:
        self.ffmpeg_path = ffmpeg_path
        self.read_size = read_size

    def encode(
        self,
        chunks: Iterable[np.ndarray],
        *,
        sample_rate: int,
        response_format: str,
        cancelled: threading.Event,
        emit: Callable[[bytes], bool],
    ) -> int:
        if response_format not in SUPPORTED_FORMATS:
            raise EncodingError(f"unsupported response format: {response_format}")
        if sample_rate <= 0:
            raise EncodingError(f"invalid sample rate: {sample_rate}")

        if response_format == "pcm":
            emitted = 0
            for chunk in chunks:
                if cancelled.is_set():
                    break
                data = pcm16_bytes(chunk)
                if data and not emit(data):
                    cancelled.set()
                    break
                emitted += len(data)
            return emitted

        command = [
            str(self.ffmpeg_path),
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-f",
            "s16le",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-flush_packets",
            "1",
            *_ffmpeg_output_args(response_format),
            "pipe:1",
        ]
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None

        emitted = 0
        reader_error: list[BaseException] = []
        stderr_tail = bytearray()

        def read_stdout() -> None:
            nonlocal emitted
            try:
                while not cancelled.is_set():
                    data = process.stdout.read(self.read_size)
                    if not data:
                        return
                    if not emit(data):
                        cancelled.set()
                        return
                    emitted += len(data)
            except BaseException as exc:
                reader_error.append(exc)
                cancelled.set()

        def read_stderr() -> None:
            while True:
                data = process.stderr.read(4096)
                if not data:
                    return
                stderr_tail.extend(data)
                if len(stderr_tail) > 16384:
                    del stderr_tail[:-16384]

        stdout_thread = threading.Thread(target=read_stdout, name="tts-ffmpeg-stdout", daemon=True)
        stderr_thread = threading.Thread(target=read_stderr, name="tts-ffmpeg-stderr", daemon=True)
        stdout_thread.start()
        stderr_thread.start()

        write_error: BaseException | None = None
        try:
            for chunk in chunks:
                if cancelled.is_set():
                    break
                data = pcm16_bytes(chunk)
                if data:
                    process.stdin.write(data)
            process.stdin.close()
        except (BrokenPipeError, OSError) as exc:
            write_error = exc
            cancelled.set()
        except BaseException:
            cancelled.set()
            raise
        finally:
            if cancelled.is_set() and process.poll() is None:
                process.terminate()
            try:
                return_code = process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                return_code = process.wait(timeout=10)
            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)
            process.stdout.close()
            process.stderr.close()

        if cancelled.is_set():
            return emitted
        error_text = stderr_tail.decode("utf-8", errors="replace").strip()
        if reader_error:
            raise EncodingError(f"ffmpeg output reader failed: {reader_error[-1]!r}")
        if write_error is not None:
            raise EncodingError(f"ffmpeg input failed: {write_error!r}; {error_text}")
        if return_code != 0:
            raise EncodingError(f"ffmpeg exited with code {return_code}: {error_text}")
        if emitted <= 0:
            raise EncodingError("ffmpeg produced no audio")
        return emitted
