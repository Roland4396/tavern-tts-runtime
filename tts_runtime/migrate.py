from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .catalog import REQUIRED_RUNTIME_FILES, build_catalog, chmod_if_supported


CHINA_TIME = ZoneInfo("Asia/Shanghai")
CORE_SOURCE_FILES = frozenset({"gpt.ckpt", "sovits.pth", "reference.wav"})
BACKUP_SUFFIX = ".normalize-backup"
UNSUPPORTED_LINK_ERRNOS = frozenset(
    {errno.ENOSYS, errno.EOPNOTSUPP, errno.EPERM, errno.EXDEV}
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_preserving(source: Path, destination: Path) -> None:
    if source.is_symlink():
        raise RuntimeError(f"refusing to archive symlink: {source}")
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)


def _tree_manifest(paths: list[Path], base: Path) -> dict[str, tuple[str, int, str]]:
    manifest: dict[str, tuple[str, int, str]] = {}

    def add(path: Path) -> None:
        if path.is_symlink():
            raise RuntimeError(f"refusing to archive symlink: {path}")
        relative = path.relative_to(base).as_posix()
        if path.is_dir():
            manifest[relative] = ("directory", 0, "")
            for child in sorted(path.iterdir(), key=lambda item: item.name):
                add(child)
            return
        if not path.is_file():
            raise RuntimeError(f"refusing to archive special file: {path}")
        manifest[relative] = ("file", path.stat().st_size, sha256(path))

    for entry in sorted(paths, key=lambda item: item.name):
        add(entry)
    return manifest


def _archive_non_runtime(role_dir: Path, archive_root: Path) -> None:
    excluded = set(REQUIRED_RUNTIME_FILES)
    extras = [path for path in role_dir.iterdir() if path.name not in excluded]
    if not extras:
        return
    source_manifest = _tree_manifest(extras, role_dir)
    archive_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    destination = archive_root / role_dir.name
    if destination.exists():
        archived_manifest = _tree_manifest(list(destination.iterdir()), destination)
        if archived_manifest != source_manifest:
            raise RuntimeError(
                f"legacy artifact archive differs from source: {destination}"
            )
        return

    staging = archive_root / f".{role_dir.name}.archive.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(mode=0o700)
    try:
        for source in extras:
            _copy_preserving(source, staging / source.name)
        copied_manifest = _tree_manifest(list(staging.iterdir()), staging)
        if copied_manifest != source_manifest:
            raise RuntimeError(f"legacy artifact archive verification failed: {role_dir}")
        for path in staging.rglob("*"):
            chmod_if_supported(path, 0o700 if path.is_dir() else 0o600)
        os.replace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _legacy_metadata(role_dir: Path) -> dict:
    path = role_dir / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"metadata must be an object: {path}")
    return value


def _reference_text(role_dir: Path, metadata: dict) -> str:
    path = role_dir / "reference.txt"
    if path.is_file():
        value = path.read_text(encoding="utf-8").strip()
        if value:
            return value
    value = str(metadata.get("reference_text") or "").strip()
    if not value:
        raise ValueError(f"reference text is missing: {role_dir}")
    return value


def _replace_directory(staging: Path, destination: Path) -> None:
    backup = destination.with_name(f".{destination.name}{BACKUP_SUFFIX}")
    if backup.exists():
        raise FileExistsError(f"stale normalization backup exists: {backup}")
    os.replace(destination, backup)
    try:
        os.replace(staging, destination)
    except BaseException:
        os.replace(backup, destination)
        raise
    shutil.rmtree(backup)


def _is_normalized(
    role_dir: Path,
    metadata: dict,
    *,
    allow_extras: bool = False,
) -> bool:
    if metadata.get("schema_version") != 1:
        return False
    if str(metadata.get("voice_id") or "") != role_dir.name:
        return False
    if str(metadata.get("status") or "") != "ready":
        return False
    hashes = metadata.get("sha256")
    if not isinstance(hashes, dict):
        return False
    if any(
        not str(hashes.get(name) or "")
        for name in (*CORE_SOURCE_FILES, "reference.txt")
    ):
        return False
    names = {path.name for path in role_dir.iterdir()}
    required = set(REQUIRED_RUNTIME_FILES)
    if not required.issubset(names) or (not allow_extras and names != required):
        return False
    return all(
        not (role_dir / name).is_symlink()
        and (role_dir / name).is_file()
        and (role_dir / name).stat().st_size > 0
        for name in REQUIRED_RUNTIME_FILES
    )


def _recover_interrupted_swaps(role_root: Path) -> None:
    for staging in role_root.glob(".*.normalize-*.staging"):
        if staging.is_dir() and not staging.is_symlink():
            shutil.rmtree(staging)
    for backup in sorted(role_root.glob(f".*{BACKUP_SUFFIX}")):
        voice_id = backup.name[1 : -len(BACKUP_SUFFIX)]
        destination = role_root / voice_id
        if not destination.exists():
            os.replace(backup, destination)
            continue
        metadata = _legacy_metadata(destination)
        if not _is_normalized(destination, metadata):
            raise RuntimeError(
                f"cannot discard interrupted normalization backup: {backup}"
            )
        shutil.rmtree(backup)


def _remove_non_runtime(role_dir: Path) -> None:
    required = set(REQUIRED_RUNTIME_FILES)
    for path in role_dir.iterdir():
        if path.name in required:
            continue
        if path.is_symlink():
            raise RuntimeError(f"refusing to remove archived symlink: {path}")
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _cleanup_transaction_files(role_dir: Path) -> None:
    patterns = (
        ".reference.txt.*.hash",
        ".reference.txt.*.tmp",
        ".SHA256SUMS.*.tmp",
        ".metadata.json.*.tmp",
    )
    for pattern in patterns:
        for path in role_dir.glob(pattern):
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"invalid migration transaction file: {path}")
            path.unlink()


def _restrict_role_permissions(role_dir: Path) -> None:
    chmod_if_supported(role_dir, 0o700)
    for path in role_dir.iterdir():
        chmod_if_supported(path, 0o700 if path.is_dir() else 0o600)


def _normalized_metadata(
    *,
    role_dir: Path,
    metadata: dict,
    reference_text: str,
    display_name: str,
    hashes: dict[str, str],
) -> dict:
    provenance = dict(metadata.get("provenance") or {})
    for key in (
        "run_id",
        "completed_at_china",
        "gpt_epoch",
        "gpt_epochs",
        "sovits_epoch",
        "sovits_epochs",
        "training_samples",
        "training_minutes",
    ):
        if key in metadata:
            provenance[key] = metadata[key]
    revision = hashlib.sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 1,
        "voice_id": role_dir.name,
        "display_name": display_name,
        "language": str(metadata.get("language") or "zh"),
        "model_id": "gpt-sovits-v2proplus",
        "model_version": "v2ProPlus",
        "status": "ready",
        "revision": revision,
        "reference_text": reference_text,
        "normalized_at_china": datetime.now(CHINA_TIME).isoformat(timespec="seconds"),
        "sha256": hashes,
        "provenance": provenance,
    }


def _atomic_write_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
        chmod_if_supported(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _normalize_in_place(
    role_dir: Path,
    metadata: dict,
    reference_text: str,
    display_name: str,
) -> dict:
    reference_value = reference_text + "\n"
    reference_temporary = role_dir / f".reference.txt.{os.getpid()}.hash"
    try:
        reference_temporary.write_text(reference_value, encoding="utf-8")
        hashes = {
            name: sha256(role_dir / name) for name in sorted(CORE_SOURCE_FILES)
        }
        hashes["reference.txt"] = sha256(reference_temporary)
    finally:
        if reference_temporary.exists():
            reference_temporary.unlink()

    normalized = _normalized_metadata(
        role_dir=role_dir,
        metadata=metadata,
        reference_text=reference_text,
        display_name=display_name,
        hashes=hashes,
    )
    sums = "".join(
        f"{digest}  {name}\n" for name, digest in sorted(hashes.items())
    )
    _atomic_write_text(role_dir / "reference.txt", reference_value)
    _atomic_write_text(role_dir / "SHA256SUMS", sums)
    _atomic_write_text(
        role_dir / "metadata.json",
        json.dumps(normalized, ensure_ascii=False, indent=2) + "\n",
    )
    _remove_non_runtime(role_dir)
    _restrict_role_permissions(role_dir)
    return normalized


def normalize_role(role_dir: Path, archive_root: Path) -> dict:
    voice_id = role_dir.name
    _cleanup_transaction_files(role_dir)
    metadata = _legacy_metadata(role_dir)
    if _is_normalized(role_dir, metadata, allow_extras=True):
        _archive_non_runtime(role_dir, archive_root)
        _remove_non_runtime(role_dir)
        _restrict_role_permissions(role_dir)
        return metadata
    reference_text = _reference_text(role_dir, metadata)
    display_name = str(
        metadata.get("display_name") or metadata.get("role") or metadata.get("name") or ""
    ).strip()
    if not display_name:
        raise ValueError(f"display name is missing: {role_dir}")
    for name in CORE_SOURCE_FILES:
        path = role_dir / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)

    _archive_non_runtime(role_dir, archive_root)
    staging = role_dir.with_name(f".{voice_id}.normalize-{os.getpid()}.staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(mode=0o700)
    try:
        try:
            for name in sorted(CORE_SOURCE_FILES):
                os.link(role_dir / name, staging / name)
        except OSError as exc:
            if exc.errno not in UNSUPPORTED_LINK_ERRNOS:
                raise
            shutil.rmtree(staging)
            return _normalize_in_place(
                role_dir,
                metadata,
                reference_text,
                display_name,
            )
        (staging / "reference.txt").write_text(reference_text + "\n", encoding="utf-8")
        hashes = {
            name: sha256(staging / name)
            for name in ("gpt.ckpt", "sovits.pth", "reference.wav", "reference.txt")
        }
        normalized = _normalized_metadata(
            role_dir=role_dir,
            metadata=metadata,
            reference_text=reference_text,
            display_name=display_name,
            hashes=hashes,
        )
        (staging / "metadata.json").write_text(
            json.dumps(normalized, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (staging / "SHA256SUMS").write_text(
            "".join(f"{digest}  {name}\n" for name, digest in sorted(hashes.items())),
            encoding="utf-8",
        )
        for path in staging.iterdir():
            chmod_if_supported(path, 0o600)
        _replace_directory(staging, role_dir)
        chmod_if_supported(role_dir, 0o700)
        return normalized
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize GPT-SoVITS runtime voices")
    parser.add_argument(
        "--release-root",
        type=Path,
        default=Path("/home/worker/tts_release/gpt_sovits_v2proplus"),
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=Path("/home/worker/experiments/gpt_sovits_release_history"),
    )
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    if os.geteuid() == 0:
        raise PermissionError("run normalization as worker, not root or sudo")
    os.umask(0o077)
    args = parse_args()
    release_root = args.release_root.resolve()
    role_root = release_root / "roles"
    release_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_path = release_root / ".normalize.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        _recover_interrupted_swaps(role_root)
        if args.check:
            result = build_catalog(role_root, release_root / "catalog.json")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return

        metadata = []
        for role_dir in sorted(role_root.iterdir(), key=lambda path: path.name):
            if role_dir.name.startswith(".") or not role_dir.is_dir():
                continue
            print(f"normalizing voice={role_dir.name}", flush=True)
            metadata.append(normalize_role(role_dir, args.archive_root.resolve()))
        catalog = build_catalog(role_root, release_root / "catalog.json")
        print(
            json.dumps(
                {
                    "normalized": len(metadata),
                    "catalog_revision": catalog["revision"],
                    "voices": [item["voice_id"] for item in metadata],
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
