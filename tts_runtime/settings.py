from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _boolean(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "1" if default else "0").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


@dataclass(frozen=True)
class Settings:
    engine_root: Path
    role_root: Path
    catalog_path: Path
    state_dir: Path
    weight_root: Path
    ffmpeg_path: Path
    device: str
    idle_seconds: int
    max_pending: int
    max_input_chars: int
    first_byte_timeout_seconds: int
    synthesis_timeout_seconds: int
    exit_on_unload: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        engine_root = Path(
            os.environ.get("TTS_ENGINE_ROOT", "/home/worker/codes/audio-synthesis")
        ).resolve()
        release_root = Path(
            os.environ.get(
                "TTS_RELEASE_ROOT",
                "/home/worker/tts_release/gpt_sovits_v2proplus",
            )
        ).resolve()
        state_dir = Path(
            os.environ.get("TTS_STATE_DIR", "/home/worker/.local/state/tts-runtime")
        ).resolve()
        return cls(
            engine_root=engine_root,
            role_root=release_root / "roles",
            catalog_path=release_root / "catalog.json",
            state_dir=state_dir,
            weight_root=Path(
                os.environ.get(
                    "TTS_WEIGHT_ROOT", "/mnt/data/models/GPT-SoVITS-V2ProPlus"
                )
            ).resolve(),
            ffmpeg_path=Path(
                os.environ.get("TTS_FFMPEG", str(engine_root / ".venv/bin/ffmpeg"))
            ).resolve(),
            device=os.environ.get("TTS_DEVICE", "cuda:0").strip() or "cuda:0",
            idle_seconds=_positive_int("TTS_IDLE_SECONDS", 1800),
            max_pending=_positive_int("TTS_MAX_PENDING", 8),
            max_input_chars=_positive_int("TTS_MAX_INPUT_CHARS", 4096),
            first_byte_timeout_seconds=_positive_int(
                "TTS_FIRST_BYTE_TIMEOUT_SECONDS", 600
            ),
            synthesis_timeout_seconds=_positive_int(
                "TTS_SYNTHESIS_TIMEOUT_SECONDS", 900
            ),
            exit_on_unload=_boolean("TTS_EXIT_ON_UNLOAD"),
        )

    def validate_paths(self) -> None:
        required = (
            self.engine_root,
            self.role_root,
            self.catalog_path,
            self.weight_root / "chinese-roberta-wwm-ext-large",
            self.weight_root / "chinese-hubert-base",
            self.ffmpeg_path,
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("missing TTS runtime paths: " + ", ".join(missing))
