from __future__ import annotations

import queue
import threading
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tts_runtime.catalog import VoiceProfile
from tts_runtime.runtime import AudioChunk, InferenceCoordinator, StreamEnd, SynthesisSpec
from tts_runtime.settings import Settings


class FakeEngine:
    def __init__(self) -> None:
        self.current_profile = None
        self.closed = 0
        self.stopped = 0

    def synthesize(self, request):
        self.current_profile = request.profile
        yield 32000, np.array([0, 100, -100], dtype=np.int16)

    def stop(self) -> None:
        self.stopped += 1

    def close(self) -> None:
        self.current_profile = None
        self.closed += 1


class FakeEncoder:
    def encode(self, chunks, **kwargs):
        list(chunks)
        data = b"encoded-audio"
        kwargs["emit"](data)
        return len(data)


class BlockingEngine(FakeEngine):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def synthesize(self, request):
        self.current_profile = request.profile
        self.started.set()
        self.release.wait(5)
        if False:
            yield 32000, np.array([], dtype=np.int16)

    def stop(self) -> None:
        super().stop()
        self.release.set()


class RuntimeTest(unittest.TestCase):
    def settings(self, root: Path, idle_seconds: int = 30) -> Settings:
        return Settings(
            engine_root=root,
            role_root=root,
            catalog_path=root / "catalog.json",
            state_dir=root / "state",
            weight_root=root,
            ffmpeg_path=root / "ffmpeg",
            device="cpu",
            idle_seconds=idle_seconds,
            max_pending=2,
            max_input_chars=4096,
            first_byte_timeout_seconds=10,
            synthesis_timeout_seconds=10,
        )

    def profile(self, root: Path) -> VoiceProfile:
        return VoiceProfile(
            voice_id="firefly",
            display_name="流萤",
            language="zh",
            model="gpt-sovits-v2proplus",
            model_version="v2ProPlus",
            revision="rev1",
            directory=root,
            gpt_path=root / "gpt.ckpt",
            sovits_path=root / "sovits.pth",
            reference_path=root / "reference.wav",
            reference_text="测试。",
        )

    def test_serial_job_stream_and_idle_unload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine = FakeEngine()
            coordinator = InferenceCoordinator(
                self.settings(root, idle_seconds=1),
                engine=engine,
                encoder=FakeEncoder(),
            )
            coordinator.start()
            try:
                job = coordinator.submit(
                    SynthesisSpec("你好", 1.0, "mp3", self.profile(root))
                )
                first = job.events.get(timeout=3)
                self.assertIsInstance(first, AudioChunk)
                self.assertEqual(first.data, b"encoded-audio")
                self.assertIsInstance(job.events.get(timeout=3), StreamEnd)
                self.assertTrue(job.finished.wait(3))
                self.assertEqual(coordinator.snapshot()["state"], "READY")

                deadline = time.monotonic() + 4
                while time.monotonic() < deadline:
                    if coordinator.snapshot()["state"] == "COLD":
                        break
                    time.sleep(0.1)
                self.assertEqual(coordinator.snapshot()["state"], "COLD")
                self.assertGreaterEqual(engine.closed, 1)
            finally:
                coordinator.shutdown()

    def test_unload_requests_process_exit_when_configured(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine = FakeEngine()
            settings = replace(self.settings(root), exit_on_unload=True)
            coordinator = InferenceCoordinator(
                settings,
                engine=engine,
                encoder=FakeEncoder(),
            )
            coordinator.start()
            try:
                job = coordinator.submit(
                    SynthesisSpec("你好", 1.0, "mp3", self.profile(root))
                )
                self.assertTrue(job.finished.wait(3))
                while not job.events.empty():
                    job.events.get_nowait()

                with patch("tts_runtime.runtime.os.kill") as kill:
                    self.assertTrue(coordinator.request_unload())
                    deadline = time.monotonic() + 3
                    while time.monotonic() < deadline and not kill.called:
                        time.sleep(0.05)

                kill.assert_called_once()
                self.assertEqual(coordinator.snapshot()["state"], "COLD")
            finally:
                coordinator.shutdown()

    def test_cancel_before_first_audio_is_not_a_runtime_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine = BlockingEngine()
            coordinator = InferenceCoordinator(
                self.settings(root),
                engine=engine,
                encoder=FakeEncoder(),
            )
            coordinator.start()
            try:
                job = coordinator.submit(
                    SynthesisSpec("取消测试", 1.0, "mp3", self.profile(root))
                )
                self.assertTrue(engine.started.wait(2))
                coordinator.cancel(job)
                self.assertIsInstance(job.events.get(timeout=3), StreamEnd)
                self.assertTrue(job.finished.wait(3))
                deadline = time.monotonic() + 2
                while coordinator.snapshot()["active_request_id"] and time.monotonic() < deadline:
                    time.sleep(0.01)
                snapshot = coordinator.snapshot()
                self.assertEqual(snapshot["state"], "READY")
                self.assertEqual(snapshot["last_error"], "")
            finally:
                coordinator.shutdown()


if __name__ == "__main__":
    unittest.main()
