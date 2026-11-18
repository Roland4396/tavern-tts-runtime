from __future__ import annotations

import errno
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tts_runtime.migrate import (
    BACKUP_SUFFIX,
    _archive_non_runtime,
    _recover_interrupted_swaps,
    normalize_role,
)


class MigrationTest(unittest.TestCase):
    def make_legacy_role(self, root: Path, voice_id: str = "firefly") -> Path:
        role = root / "roles" / voice_id
        role.mkdir(parents=True)
        (role / "gpt.ckpt").write_bytes(b"gpt")
        (role / "sovits.pth").write_bytes(b"sovits")
        (role / "reference.wav").write_bytes(b"wav")
        (role / "reference.txt").write_text("test reference\n", encoding="utf-8")
        (role / "training.list").write_text("legacy\n", encoding="utf-8")
        (role / "metadata.json").write_text(
            json.dumps(
                {
                    "role_id": voice_id,
                    "role": "Firefly",
                    "reference_text": "test reference",
                    "status": "complete",
                }
            ),
            encoding="utf-8",
        )
        (role / "SHA256SUMS").write_text("legacy\n", encoding="utf-8")
        return role

    def test_normalization_is_idempotent_after_archive_was_already_published(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            role = self.make_legacy_role(root)
            archive = root / "archive"
            _archive_non_runtime(role, archive)

            first = normalize_role(role, archive)
            second = normalize_role(role, archive)

            self.assertEqual(first["revision"], second["revision"])
            self.assertEqual(
                {path.name for path in role.iterdir()},
                {
                    "gpt.ckpt",
                    "sovits.pth",
                    "reference.wav",
                    "reference.txt",
                    "metadata.json",
                    "SHA256SUMS",
                },
            )
            self.assertEqual(
                (archive / "firefly" / "training.list").read_text(), "legacy\n"
            )
            self.assertEqual(role.stat().st_mode & 0o777, 0o700)
            self.assertTrue(
                all(
                    path.stat().st_mode & 0o777 == 0o600
                    for path in role.iterdir()
                )
            )

    def test_recovers_swap_interrupted_before_new_directory_publish(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            role = self.make_legacy_role(root)
            backup = role.with_name(f".{role.name}{BACKUP_SUFFIX}")
            os.replace(role, backup)
            stale = role.with_name(f".{role.name}.normalize-123.staging")
            stale.mkdir()

            _recover_interrupted_swaps(root / "roles")

            self.assertTrue(role.is_dir())
            self.assertFalse(backup.exists())
            self.assertFalse(stale.exists())

    def test_falls_back_to_in_place_commit_when_hardlinks_are_unsupported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            role = self.make_legacy_role(root)
            archive = root / "archive"

            with patch(
                "tts_runtime.migrate.os.link",
                side_effect=OSError(errno.ENOSYS, "hardlinks unsupported"),
            ):
                metadata = normalize_role(role, archive)

            self.assertEqual(metadata["schema_version"], 1)
            self.assertEqual(metadata["status"], "ready")
            self.assertFalse((role / "training.list").exists())
            self.assertEqual((archive / "firefly" / "training.list").read_text(), "legacy\n")
            self.assertEqual(
                {path.name for path in role.iterdir()},
                {
                    "gpt.ckpt",
                    "sovits.pth",
                    "reference.wav",
                    "reference.txt",
                    "metadata.json",
                    "SHA256SUMS",
                },
            )


if __name__ == "__main__":
    unittest.main()
