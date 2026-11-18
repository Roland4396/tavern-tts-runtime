from __future__ import annotations

import gc
import os
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from .catalog import VoiceProfile
from .settings import Settings


@dataclass(frozen=True)
class EngineRequest:
    text: str
    speed: float
    profile: VoiceProfile


class GptSovitsEngine:
    """Own all mutable GPT-SoVITS state inside the inference worker thread."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.pipeline: Any | None = None
        self.current_profile: VoiceProfile | None = None

    def _import_engine(self):
        root = str(self.settings.engine_root)
        package_root = str(self.settings.engine_root / "GPT_SoVITS")
        for path in (package_root, root):
            if path not in sys.path:
                sys.path.insert(0, path)
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

        return TTS, TTS_Config

    def load(self, profile: VoiceProfile) -> None:
        if (
            self.pipeline is not None
            and self.current_profile is not None
            and self.current_profile.voice_id == profile.voice_id
            and self.current_profile.revision == profile.revision
        ):
            return

        if self.pipeline is None:
            TTS, TTS_Config = self._import_engine()
            self.settings.state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            config = TTS_Config(
                {
                    "custom": {
                        "device": self.settings.device,
                        "is_half": True,
                        "version": "v2ProPlus",
                        "t2s_weights_path": str(profile.gpt_path),
                        "vits_weights_path": str(profile.sovits_path),
                        "bert_base_path": str(
                            self.settings.weight_root / "chinese-roberta-wwm-ext-large"
                        ),
                        "cnhuhbert_base_path": str(
                            self.settings.weight_root / "chinese-hubert-base"
                        ),
                    }
                }
            )
            config.configs_path = str(self.settings.state_dir / "tts-infer.yaml")
            self.pipeline = TTS(config)
        else:
            self.pipeline.stop()
            self.pipeline.init_t2s_weights(str(profile.gpt_path))
            self.pipeline.init_vits_weights(str(profile.sovits_path))
            self.pipeline.empty_cache()
        self.current_profile = profile

    def synthesize(self, request: EngineRequest) -> Iterator[tuple[int, np.ndarray]]:
        self.load(request.profile)
        assert self.pipeline is not None
        inputs = {
            "text": request.text,
            "text_lang": "all_zh",
            "ref_audio_path": str(request.profile.reference_path),
            "aux_ref_audio_paths": [],
            "prompt_text": request.profile.reference_text,
            "prompt_lang": "all_zh",
            "top_k": 15,
            "top_p": 1.0,
            "temperature": 1.0,
            "text_split_method": "cut5",
            "batch_size": 1,
            "batch_threshold": 0.75,
            "split_bucket": False,
            "speed_factor": request.speed,
            "fragment_interval": 0.3,
            "seed": -1,
            "parallel_infer": True,
            "repetition_penalty": 1.35,
            "sample_steps": 32,
            "super_sampling": False,
            "return_fragment": True,
            "streaming_mode": False,
            "fixed_length_chunk": False,
            "overlap_length": 2,
            "min_chunk_length": 16,
        }
        yield from self.pipeline.run(inputs)

    def stop(self) -> None:
        if self.pipeline is not None:
            self.pipeline.stop()

    def close(self) -> None:
        pipeline = self.pipeline
        self.pipeline = None
        self.current_profile = None
        if pipeline is not None:
            pipeline.stop()
            del pipeline
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
