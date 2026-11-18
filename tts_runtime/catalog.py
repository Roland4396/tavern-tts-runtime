from __future__ import annotations

import hashlib
import json
import os
import errno
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VOICE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
REQUIRED_RUNTIME_FILES = (
    "gpt.ckpt",
    "sovits.pth",
    "reference.wav",
    "reference.txt",
    "metadata.json",
    "SHA256SUMS",
)
UNSUPPORTED_PERMISSION_ERRNOS = frozenset(
    {errno.ENOSYS, errno.EOPNOTSUPP, errno.EPERM}
)


class CatalogError(RuntimeError):
    pass


class UnknownVoiceError(CatalogError):
    pass


def chmod_if_supported(path: Path, mode: int) -> bool:
    """Apply private modes while tolerating object filesystems without chmod."""
    try:
        path.chmod(mode)
        return True
    except OSError as exc:
        if exc.errno not in UNSUPPORTED_PERMISSION_ERRNOS:
            raise
        return False


@dataclass(frozen=True)
class VoiceProfile:
    voice_id: str
    display_name: str
    language: str
    model: str
    model_version: str
    revision: str
    directory: Path
    gpt_path: Path
    sovits_path: Path
    reference_path: Path
    reference_text: str

    def public_dict(self) -> dict[str, str]:
        return {
            "id": self.voice_id,
            "name": self.display_name,
            "language": self.language,
            "model": self.model,
            "model_version": self.model_version,
            "revision": self.revision,
        }


def _safe_role_directory(role_root: Path, voice_id: str) -> Path:
    if not VOICE_ID_RE.fullmatch(voice_id):
        raise CatalogError(f"invalid voice id: {voice_id!r}")
    role_root = role_root.resolve()
    directory = (role_root / voice_id).resolve()
    if directory.parent != role_root:
        raise CatalogError(f"voice directory escapes role root: {voice_id}")
    if directory.is_symlink() or not directory.is_dir():
        raise CatalogError(f"voice directory is missing or is a symlink: {directory}")
    return directory


def _required_file(directory: Path, name: str) -> Path:
    path = directory / name
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise CatalogError(f"invalid runtime file: {path}")
    return path


def _metadata_value(metadata: dict[str, Any], key: str, legacy_key: str = "") -> str:
    value = metadata.get(key)
    if value in (None, "") and legacy_key:
        value = metadata.get(legacy_key)
    return str(value or "").strip()


def load_profile(role_root: Path, entry: dict[str, Any]) -> VoiceProfile:
    voice_id = str(entry.get("id") or entry.get("voice_id") or "").strip()
    directory = _safe_role_directory(role_root, voice_id)
    for name in REQUIRED_RUNTIME_FILES:
        _required_file(directory, name)

    try:
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CatalogError(f"invalid metadata for {voice_id}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise CatalogError(f"metadata must be an object: {directory / 'metadata.json'}")

    metadata_id = _metadata_value(metadata, "voice_id", "role_id")
    if metadata_id != voice_id:
        raise CatalogError(
            f"catalog/metadata voice id mismatch: {voice_id!r} != {metadata_id!r}"
        )
    if _metadata_value(metadata, "status") not in {"ready", "complete"}:
        raise CatalogError(f"voice is not ready: {voice_id}")

    display_name = _metadata_value(metadata, "display_name", "role")
    language = _metadata_value(metadata, "language") or "zh"
    model = _metadata_value(metadata, "model_id") or "gpt-sovits-v2proplus"
    model_version = _metadata_value(metadata, "model_version") or "v2ProPlus"
    revision = str(entry.get("revision") or metadata.get("revision") or "").strip()
    if not display_name or not revision:
        raise CatalogError(f"voice metadata lacks name or revision: {voice_id}")

    reference_text = (directory / "reference.txt").read_text(encoding="utf-8").strip()
    if not reference_text:
        raise CatalogError(f"reference text is empty: {voice_id}")

    return VoiceProfile(
        voice_id=voice_id,
        display_name=display_name,
        language=language,
        model=model,
        model_version=model_version,
        revision=revision,
        directory=directory,
        gpt_path=directory / "gpt.ckpt",
        sovits_path=directory / "sovits.pth",
        reference_path=directory / "reference.wav",
        reference_text=reference_text,
    )


class VoiceCatalog:
    def __init__(self, role_root: Path, catalog_path: Path) -> None:
        self.role_root = role_root.resolve()
        self.catalog_path = catalog_path.resolve()
        self._lock = threading.RLock()
        self._stamp: tuple[int, int] | None = None
        self._profiles: tuple[VoiceProfile, ...] = ()
        self._aliases: dict[str, VoiceProfile] = {}
        self._revision = ""

    @property
    def revision(self) -> str:
        with self._lock:
            return self._revision

    def refresh(self, force: bool = False) -> tuple[VoiceProfile, ...]:
        try:
            stat = self.catalog_path.stat()
        except OSError as exc:
            raise CatalogError(f"catalog is unavailable: {self.catalog_path}") from exc
        stamp = (stat.st_mtime_ns, stat.st_size)
        with self._lock:
            if not force and self._stamp == stamp:
                return self._profiles

            try:
                raw = json.loads(self.catalog_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise CatalogError(f"invalid catalog: {exc}") from exc
            entries = raw.get("voices") if isinstance(raw, dict) else None
            if not isinstance(entries, list) or not entries:
                raise CatalogError("catalog contains no voices")

            profiles = tuple(load_profile(self.role_root, item) for item in entries)
            aliases: dict[str, VoiceProfile] = {}
            for profile in profiles:
                for alias in (profile.voice_id, profile.display_name):
                    key = alias.casefold()
                    existing = aliases.get(key)
                    if existing and existing.voice_id != profile.voice_id:
                        raise CatalogError(f"ambiguous voice alias: {alias}")
                    aliases[key] = profile

            self._profiles = profiles
            self._aliases = aliases
            self._revision = str(raw.get("revision") or "")
            self._stamp = stamp
            return self._profiles

    def resolve(self, voice: str) -> VoiceProfile:
        self.refresh()
        with self._lock:
            profile = self._aliases.get(voice.strip().casefold())
        if profile is None:
            raise UnknownVoiceError(f"unknown voice: {voice}")
        return profile

    def public_voices(self) -> list[dict[str, str]]:
        return [profile.public_dict() for profile in self.refresh()]


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def catalog_revision(entries: list[dict[str, str]]) -> str:
    return hashlib.sha256(_canonical_json(entries)).hexdigest()


def build_catalog(role_root: Path, catalog_path: Path) -> dict[str, Any]:
    role_root = role_root.resolve()
    entries: list[dict[str, str]] = []
    if not role_root.is_dir():
        raise CatalogError(f"role root does not exist: {role_root}")

    for directory in sorted(role_root.iterdir(), key=lambda path: path.name):
        if directory.name.startswith(".") or not directory.is_dir():
            continue
        metadata_path = directory / "metadata.json"
        if not metadata_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise CatalogError(f"invalid metadata: {metadata_path}: {exc}") from exc
        if not isinstance(metadata, dict):
            raise CatalogError(f"metadata must be an object: {metadata_path}")
        voice_id = _metadata_value(metadata, "voice_id", "role_id")
        revision = _metadata_value(metadata, "revision")
        entry = {"id": voice_id, "revision": revision}
        load_profile(role_root, entry)
        entries.append(entry)

    if not entries:
        raise CatalogError("no ready voices found")
    entries.sort(key=lambda item: item["id"])
    result: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "revision": catalog_revision(entries),
        "voices": entries,
    }
    catalog_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = catalog_path.with_name(f".{catalog_path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    chmod_if_supported(temporary, 0o600)
    os.replace(temporary, catalog_path)
    return result
