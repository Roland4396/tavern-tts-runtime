from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tts_runtime.catalog import UnknownVoiceError, VoiceCatalog, build_catalog


class VoiceCatalogTest(unittest.TestCase):
    def make_voice(self, root: Path, voice_id: str, name: str) -> None:
        directory = root / "roles" / voice_id
        directory.mkdir(parents=True)
        hashes = {}
        for filename, content in (
            ("gpt.ckpt", b"gpt"),
            ("sovits.pth", b"sovits"),
            ("reference.wav", b"wav"),
            ("reference.txt", "测试参考文本。\n".encode()),
        ):
            (directory / filename).write_bytes(content)
            hashes[filename] = hashlib.sha256(content).hexdigest()
        revision = hashlib.sha256(
            json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        (directory / "metadata.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "voice_id": voice_id,
                    "display_name": name,
                    "language": "zh",
                    "model_id": "gpt-sovits-v2proplus",
                    "model_version": "v2ProPlus",
                    "status": "ready",
                    "revision": revision,
                    "sha256": hashes,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (directory / "SHA256SUMS").write_text(
            "".join(f"{value}  {key}\n" for key, value in sorted(hashes.items())),
            encoding="utf-8",
        )

    def test_build_reload_and_resolve_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_voice(root, "firefly", "流萤")
            self.make_voice(root, "march7", "三月七")
            catalog_path = root / "catalog.json"
            result = build_catalog(root / "roles", catalog_path)
            self.assertEqual([item["id"] for item in result["voices"]], ["firefly", "march7"])

            catalog = VoiceCatalog(root / "roles", catalog_path)
            self.assertEqual(catalog.resolve("firefly").display_name, "流萤")
            self.assertEqual(catalog.resolve("流萤").voice_id, "firefly")
            self.assertEqual(len(catalog.public_voices()), 2)
            with self.assertRaises(UnknownVoiceError):
                catalog.resolve("alloy")


if __name__ == "__main__":
    unittest.main()
