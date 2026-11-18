from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tts_runtime.launcher import create_private_socket


class LauncherTest(unittest.TestCase):
    def test_creates_owner_only_unix_socket_and_replaces_stale_socket(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "service.sock"
            first = create_private_socket(path)
            first.close()
            second = create_private_socket(path)
            try:
                self.assertTrue(path.is_socket())
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
                self.assertTrue(second.get_inheritable())
            finally:
                second.close()
                path.unlink()

    def test_refuses_to_replace_regular_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "service.sock"
            path.write_text("keep", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                create_private_socket(path)
            self.assertEqual(path.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
