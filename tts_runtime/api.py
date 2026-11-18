from __future__ import annotations

import asyncio
import logging
import queue
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from . import __version__
from .catalog import CatalogError, UnknownVoiceError, VoiceCatalog
from .encoding import CONTENT_TYPES, SUPPORTED_FORMATS
from .runtime import (
    AudioChunk,
    InferenceCoordinator,
    QueueCapacityError,
    StreamEnd,
    StreamFailure,
    SynthesisJob,
    SynthesisSpec,
)
from .settings import Settings


LOG = logging.getLogger("tts-runtime")
MODEL_ALIASES = frozenset(
    {
        "tts-1",
        "tts-1-hd",
        "gpt-4o-mini-tts",
        "gpt-sovits-v2proplus",
    }
)


class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = Field(min_length=1, max_length=128)
    input: str = Field(min_length=1)
    voice: str = Field(min_length=1, max_length=128)
    response_format: str = "mp3"
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    instructions: str | None = None
    stream_format: str | None = None


def _request_id(request: Request) -> str:
    supplied = request.headers.get("x-client-request-id", "").strip()
    if supplied and len(supplied) <= 512 and supplied.isascii():
        return supplied
    return uuid.uuid4().hex


def _error(
    message: str,
    *,
    status_code: int,
    error_type: str,
    code: str,
    param: str | None = None,
    request_id: str = "",
) -> JSONResponse:
    payload: dict[str, Any] = {
        "error": {
            "message": message,
            "type": error_type,
            "param": param,
            "code": code,
        }
    }
    headers = {"Cache-Control": "no-store"}
    if request_id:
        headers["X-Request-Id"] = request_id
    return JSONResponse(payload, status_code=status_code, headers=headers)


async def _poll_event(job: SynthesisJob, timeout: float = 1.0):
    def get_event():
        try:
            return job.events.get(timeout=timeout)
        except queue.Empty:
            return None

    return await asyncio.to_thread(get_event)


async def _first_event(
    request: Request,
    job: SynthesisJob,
    coordinator: InferenceCoordinator,
    timeout_seconds: int,
):
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if await request.is_disconnected():
            coordinator.cancel(job)
            return None
        event = await _poll_event(job)
        if event is not None:
            return event
    coordinator.cancel(job)
    return StreamFailure(
        "timed out waiting for the first audio chunk",
        error_type="timeout_error",
        code="first_byte_timeout",
        status_code=504,
    )


async def _stream_job(
    request: Request,
    job: SynthesisJob,
    coordinator: InferenceCoordinator,
    first_chunk: AudioChunk,
) -> AsyncIterator[bytes]:
    completed = False
    try:
        yield first_chunk.data
        while True:
            if await request.is_disconnected():
                return
            event = await _poll_event(job)
            if event is None:
                continue
            if isinstance(event, AudioChunk):
                yield event.data
                continue
            if isinstance(event, StreamFailure):
                LOG.error(
                    "stream_failed_after_headers request_id=%s code=%s message=%s",
                    job.request_id,
                    event.code,
                    event.message,
                )
                return
            if isinstance(event, StreamEnd):
                completed = True
                return
    except asyncio.CancelledError:
        raise
    finally:
        if not completed:
            coordinator.cancel(job)


def create_app(
    *,
    settings: Settings | None = None,
    catalog: VoiceCatalog | None = None,
    coordinator: InferenceCoordinator | None = None,
) -> FastAPI:
    runtime_settings = settings or Settings.from_env()
    voice_catalog = catalog or VoiceCatalog(
        runtime_settings.role_root,
        runtime_settings.catalog_path,
    )
    inference = coordinator or InferenceCoordinator(runtime_settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        runtime_settings.validate_paths()
        voices = voice_catalog.refresh(force=True)
        inference.start()
        LOG.info(
            "api_started version=%s voices=%d idle_seconds=%d device=%s",
            __version__,
            len(voices),
            runtime_settings.idle_seconds,
            runtime_settings.device,
        )
        try:
            yield
        finally:
            await asyncio.to_thread(inference.shutdown)

    api = FastAPI(
        title="Private GPT-SoVITS Speech API",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )

    @api.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        errors = exc.errors()
        first = errors[0] if errors else {}
        location = first.get("loc") or []
        param = str(location[-1]) if location else None
        return _error(
            str(first.get("msg") or "invalid request"),
            status_code=400,
            error_type="invalid_request_error",
            code="invalid_request",
            param=param,
            request_id=_request_id(request),
        )

    @api.get("/health")
    async def health():
        try:
            voices = voice_catalog.refresh()
            catalog_status = "ready"
        except CatalogError as exc:
            voices = ()
            catalog_status = f"error: {exc}"
        snapshot = inference.snapshot()
        healthy = bool(snapshot.get("worker_alive")) and bool(voices)
        return JSONResponse(
            {
                "status": "ok" if healthy else "degraded",
                "catalog": catalog_status,
                "voice_count": len(voices),
                **snapshot,
            },
            headers={"Cache-Control": "no-store"},
        )

    @api.get("/v1/audio/voices")
    async def voices(request: Request):
        request_id = _request_id(request)
        try:
            items = voice_catalog.public_voices()
        except CatalogError as exc:
            return _error(
                str(exc),
                status_code=503,
                error_type="server_error",
                code="catalog_unavailable",
                request_id=request_id,
            )
        return JSONResponse(
            {
                "object": "list",
                "data": items,
                "revision": voice_catalog.revision,
            },
            headers={"Cache-Control": "no-store", "X-Request-Id": request_id},
        )

    @api.post("/v1/audio/speech")
    async def create_speech(payload: SpeechRequest, request: Request):
        request_id = _request_id(request)
        model = payload.model.strip().lower()
        if model not in MODEL_ALIASES:
            return _error(
                f"unsupported TTS model: {payload.model}",
                status_code=400,
                error_type="invalid_request_error",
                code="invalid_model",
                param="model",
                request_id=request_id,
            )
        text = payload.input.strip()
        if not text:
            return _error(
                "input must not be empty",
                status_code=400,
                error_type="invalid_request_error",
                code="invalid_input",
                param="input",
                request_id=request_id,
            )
        if len(text) > runtime_settings.max_input_chars:
            return _error(
                f"input exceeds {runtime_settings.max_input_chars} characters",
                status_code=400,
                error_type="invalid_request_error",
                code="input_too_long",
                param="input",
                request_id=request_id,
            )
        response_format = payload.response_format.strip().lower()
        if response_format not in SUPPORTED_FORMATS:
            return _error(
                f"unsupported response_format: {payload.response_format}",
                status_code=400,
                error_type="invalid_request_error",
                code="invalid_response_format",
                param="response_format",
                request_id=request_id,
            )
        if payload.stream_format not in (None, "", "audio"):
            return _error(
                "only binary audio streaming is supported",
                status_code=400,
                error_type="invalid_request_error",
                code="unsupported_stream_format",
                param="stream_format",
                request_id=request_id,
            )
        try:
            profile = voice_catalog.resolve(payload.voice)
        except UnknownVoiceError as exc:
            return _error(
                str(exc),
                status_code=400,
                error_type="invalid_request_error",
                code="invalid_voice",
                param="voice",
                request_id=request_id,
            )
        except CatalogError as exc:
            return _error(
                str(exc),
                status_code=503,
                error_type="server_error",
                code="catalog_unavailable",
                request_id=request_id,
            )

        try:
            job = inference.submit(
                SynthesisSpec(
                    text=text,
                    speed=float(payload.speed),
                    response_format=response_format,
                    profile=profile,
                ),
                request_id=request_id,
            )
        except QueueCapacityError as exc:
            response = _error(
                str(exc),
                status_code=429,
                error_type="rate_limit_error",
                code="queue_full",
                request_id=request_id,
            )
            response.headers["Retry-After"] = "5"
            return response

        first = await _first_event(
            request,
            job,
            inference,
            runtime_settings.first_byte_timeout_seconds,
        )
        if first is None:
            return _error(
                "client disconnected",
                status_code=499,
                error_type="client_error",
                code="client_disconnected",
                request_id=request_id,
            )
        if isinstance(first, StreamFailure):
            return _error(
                first.message,
                status_code=first.status_code,
                error_type=first.error_type,
                code=first.code,
                request_id=request_id,
            )
        if isinstance(first, StreamEnd):
            return _error(
                "TTS produced no audio",
                status_code=500,
                error_type="server_error",
                code="empty_audio",
                request_id=request_id,
            )
        assert isinstance(first, AudioChunk)
        return StreamingResponse(
            _stream_job(request, job, inference, first),
            media_type=CONTENT_TYPES[response_format],
            headers={
                "Cache-Control": "no-store",
                "X-Request-Id": request_id,
                "X-TTS-Voice": profile.voice_id,
            },
        )

    @api.post("/_internal/unload")
    async def unload():
        scheduled = inference.request_unload()
        return JSONResponse(
            {"scheduled": scheduled, **inference.snapshot()},
            status_code=202 if scheduled else 200,
            headers={"Cache-Control": "no-store"},
        )

    api.state.settings = runtime_settings
    api.state.catalog = voice_catalog
    api.state.coordinator = inference
    return api


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
app = create_app()
