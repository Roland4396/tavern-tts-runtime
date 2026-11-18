from __future__ import annotations

import logging
import os
import queue
import signal
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Protocol

from .catalog import VoiceProfile
from .encoding import AudioEncoder
from .engine import EngineRequest, GptSovitsEngine
from .settings import Settings


LOG = logging.getLogger("tts-runtime")


@dataclass(frozen=True)
class SynthesisSpec:
    text: str
    speed: float
    response_format: str
    profile: VoiceProfile


@dataclass(frozen=True)
class AudioChunk:
    data: bytes


@dataclass(frozen=True)
class StreamFailure:
    message: str
    error_type: str = "server_error"
    code: str = "tts_failed"
    status_code: int = 500


@dataclass(frozen=True)
class StreamEnd:
    pass


StreamEvent = AudioChunk | StreamFailure | StreamEnd


@dataclass
class SynthesisJob:
    spec: SynthesisSpec
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    events: queue.Queue[StreamEvent] = field(default_factory=lambda: queue.Queue(maxsize=8))
    cancelled: threading.Event = field(default_factory=threading.Event)
    finished: threading.Event = field(default_factory=threading.Event)
    submitted_at: float = field(default_factory=time.monotonic)


class QueueCapacityError(RuntimeError):
    pass


class Engine(Protocol):
    current_profile: VoiceProfile | None

    def synthesize(self, request: EngineRequest): ...

    def stop(self) -> None: ...

    def close(self) -> None: ...


class InferenceCoordinator:
    def __init__(
        self,
        settings: Settings,
        *,
        engine: Engine | None = None,
        encoder: AudioEncoder | None = None,
    ) -> None:
        self.settings = settings
        self._engine: Engine = engine or GptSovitsEngine(settings)
        self._encoder = encoder or AudioEncoder(settings.ffmpeg_path)
        self._jobs: queue.Queue[SynthesisJob | object] = queue.Queue(
            maxsize=settings.max_pending
        )
        self._stop_sentinel = object()
        self._unload_sentinel = object()
        self._shutdown = threading.Event()
        self._unload_scheduled = threading.Event()
        self._lock = threading.RLock()
        self._worker: threading.Thread | None = None
        self._idle_monitor: threading.Thread | None = None
        self._active_job: SynthesisJob | None = None
        self._state = "COLD"
        self._last_error = ""
        self._last_activity = time.monotonic()
        self._load_failures = 0
        self._next_load_attempt = 0.0

    def start(self) -> None:
        with self._lock:
            if self._worker and self._worker.is_alive():
                return
            self._shutdown.clear()
            self._worker = threading.Thread(
                target=self._worker_loop,
                name="tts-inference-worker",
                daemon=True,
            )
            self._idle_monitor = threading.Thread(
                target=self._idle_loop,
                name="tts-idle-monitor",
                daemon=True,
            )
            self._worker.start()
            self._idle_monitor.start()
            LOG.info("runtime_started max_pending=%d", self.settings.max_pending)

    def submit(self, spec: SynthesisSpec, request_id: str = "") -> SynthesisJob:
        self.start()
        job = SynthesisJob(spec=spec, request_id=request_id or uuid.uuid4().hex)
        with self._lock:
            in_flight = self._jobs.qsize() + (1 if self._active_job else 0)
            if in_flight >= self.settings.max_pending:
                raise QueueCapacityError("TTS request queue is full")
            try:
                self._jobs.put_nowait(job)
            except queue.Full as exc:
                raise QueueCapacityError("TTS request queue is full") from exc
        LOG.info(
            "request_queued request_id=%s voice=%s chars=%d format=%s depth=%d",
            job.request_id,
            spec.profile.voice_id,
            len(spec.text),
            spec.response_format,
            self._jobs.qsize(),
        )
        return job

    def cancel(self, job: SynthesisJob) -> None:
        job.cancelled.set()
        with self._lock:
            if self._active_job is job:
                self._engine.stop()
        LOG.info("request_cancelled request_id=%s", job.request_id)

    def request_unload(self) -> bool:
        if self._unload_scheduled.is_set():
            return False
        self._unload_scheduled.set()
        try:
            self._jobs.put_nowait(self._unload_sentinel)
        except queue.Full:
            self._unload_scheduled.clear()
            return False
        return True

    def snapshot(self) -> dict[str, object]:
        now = time.monotonic()
        with self._lock:
            worker_alive = bool(self._worker and self._worker.is_alive())
            profile = self._engine.current_profile
            return {
                "state": self._state,
                "worker_alive": worker_alive,
                "queue_depth": self._jobs.qsize(),
                "active_request_id": self._active_job.request_id if self._active_job else None,
                "loaded_voice": profile.voice_id if profile else None,
                "loaded_revision": profile.revision if profile else None,
                "idle_seconds": max(0, int(now - self._last_activity)),
                "last_error": self._last_error,
            }

    def shutdown(self, timeout: float = 30.0) -> None:
        self._shutdown.set()
        with self._lock:
            active = self._active_job
            if active:
                active.cancelled.set()
                self._engine.stop()
        try:
            self._jobs.put_nowait(self._stop_sentinel)
        except queue.Full:
            try:
                pending = self._jobs.get_nowait()
                if isinstance(pending, SynthesisJob):
                    pending.cancelled.set()
                    self._finish(pending, StreamFailure("service is shutting down"))
            except queue.Empty:
                pass
            self._jobs.put_nowait(self._stop_sentinel)
        if self._worker:
            self._worker.join(timeout=timeout)
        if self._idle_monitor:
            self._idle_monitor.join(timeout=5)
        LOG.info("runtime_stopped")

    def _set_state(self, state: str, error: str = "") -> None:
        with self._lock:
            self._state = state
            self._last_error = error

    def _finish_cancelled(self, job: SynthesisJob, *, reason: str) -> None:
        self._finish(job)
        LOG.info(
            "request_finished request_id=%s voice=%s bytes=0 cancelled=true "
            "reason=%s elapsed_ms=%d",
            job.request_id,
            job.spec.profile.voice_id,
            reason,
            int((time.monotonic() - job.submitted_at) * 1000),
        )

    def _put_event(self, job: SynthesisJob, event: StreamEvent) -> bool:
        while not self._shutdown.is_set():
            if job.cancelled.is_set() and isinstance(event, AudioChunk):
                return False
            try:
                job.events.put(event, timeout=0.25)
                return True
            except queue.Full:
                if job.cancelled.is_set():
                    return False
        return False

    def _finish(self, job: SynthesisJob, event: StreamEvent = StreamEnd()) -> None:
        self._put_event(job, event)
        job.finished.set()

    def _load_backoff_error(self) -> str | None:
        remaining = self._next_load_attempt - time.monotonic()
        if remaining <= 0:
            return None
        return f"model loading is backing off for {remaining:.1f}s"

    def _record_load_failure(self) -> None:
        self._load_failures += 1
        delay = min(30 * (2 ** (self._load_failures - 1)), 300)
        self._next_load_attempt = time.monotonic() + delay

    def _pcm_chunks(self, job: SynthesisJob):
        started = time.monotonic()
        expected_rate: int | None = None
        request = EngineRequest(
            text=job.spec.text,
            speed=job.spec.speed,
            profile=job.spec.profile,
        )
        for sample_rate, audio in self._engine.synthesize(request):
            if job.cancelled.is_set():
                self._engine.stop()
                return
            if time.monotonic() - started > self.settings.synthesis_timeout_seconds:
                job.cancelled.set()
                self._engine.stop()
                raise TimeoutError("TTS synthesis timed out")
            if expected_rate is None:
                expected_rate = int(sample_rate)
            elif expected_rate != int(sample_rate):
                raise RuntimeError(
                    f"sample rate changed from {expected_rate} to {sample_rate}"
                )
            yield int(sample_rate), audio

    def _run_job(self, job: SynthesisJob) -> None:
        if job.cancelled.is_set():
            self._finish(job)
            return
        backoff = self._load_backoff_error()
        if backoff:
            self._finish(
                job,
                StreamFailure(backoff, code="model_load_backoff", status_code=503),
            )
            return

        self._set_state("LOADING")
        first_audio = None
        rest = None
        try:
            iterator = iter(self._pcm_chunks(job))
            first_audio = next(iterator)
            rest = iterator
            self._load_failures = 0
            self._next_load_attempt = 0.0
            self._set_state("BUSY")
            sample_rate, first_chunk = first_audio

            def audio_only():
                yield first_chunk
                assert rest is not None
                for current_rate, chunk in rest:
                    if current_rate != sample_rate:
                        raise RuntimeError(
                            f"sample rate changed from {sample_rate} to {current_rate}"
                        )
                    yield chunk

            emitted = self._encoder.encode(
                audio_only(),
                sample_rate=sample_rate,
                response_format=job.spec.response_format,
                cancelled=job.cancelled,
                emit=lambda data: self._put_event(job, AudioChunk(data)),
            )
            if not job.cancelled.is_set() and emitted <= 0:
                raise RuntimeError("TTS encoder produced no audio")
            self._finish(job)
            LOG.info(
                "request_finished request_id=%s voice=%s bytes=%d cancelled=%s elapsed_ms=%d",
                job.request_id,
                job.spec.profile.voice_id,
                emitted,
                str(job.cancelled.is_set()).lower(),
                int((time.monotonic() - job.submitted_at) * 1000),
            )
        except StopIteration:
            if job.cancelled.is_set():
                self._finish_cancelled(job, reason="stopped_before_first_audio")
                return
            error = "GPT-SoVITS produced no audio"
            self._record_load_failure()
            self._set_state("FAILED", error)
            self._finish(job, StreamFailure(error, code="empty_audio"))
        except BaseException as exc:
            if job.cancelled.is_set():
                self._finish_cancelled(
                    job,
                    reason=f"stopped_with_{type(exc).__name__}",
                )
                return
            error = f"{type(exc).__name__}: {exc}"
            self._record_load_failure()
            self._set_state("FAILED", error)
            self._finish(job, StreamFailure(str(exc) or type(exc).__name__))
            LOG.exception("request_failed request_id=%s", job.request_id)
            try:
                self._engine.close()
            except Exception:
                LOG.exception("engine_reset_failed")
        finally:
            with self._lock:
                self._last_activity = time.monotonic()
            if self._engine.current_profile is not None and self._state != "FAILED":
                self._set_state("READY")

    def _unload(self) -> None:
        self._unload_scheduled.clear()
        with self._lock:
            if self._active_job is not None:
                return
        if self._engine.current_profile is None:
            self._set_state("COLD")
            return
        self._set_state("UNLOADING")
        try:
            self._engine.close()
            self._set_state("COLD")
            LOG.info("models_unloaded")
            if self.settings.exit_on_unload and not self._shutdown.is_set():
                LOG.info("runtime_exit_requested reason=models_unloaded")
                os.kill(os.getpid(), signal.SIGTERM)
        except Exception as exc:
            self._set_state("FAILED", str(exc))
            LOG.exception("models_unload_failed")

    def _worker_loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                item = self._jobs.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is self._stop_sentinel:
                break
            if item is self._unload_sentinel:
                self._unload()
                continue
            assert isinstance(item, SynthesisJob)
            with self._lock:
                self._active_job = item
            try:
                self._run_job(item)
            finally:
                with self._lock:
                    self._active_job = None
        self._unload()

    def _idle_loop(self) -> None:
        interval = max(1, min(15, self.settings.idle_seconds // 4))
        while not self._shutdown.wait(interval):
            with self._lock:
                idle_for = time.monotonic() - self._last_activity
                can_unload = (
                    self._active_job is None
                    and self._jobs.empty()
                    and self._engine.current_profile is not None
                    and idle_for >= self.settings.idle_seconds
                )
            if can_unload:
                self.request_unload()
