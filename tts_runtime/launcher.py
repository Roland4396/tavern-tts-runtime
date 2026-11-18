from __future__ import annotations

import os
import socket
import stat
import sys
from pathlib import Path


def create_private_socket(path: Path) -> socket.socket:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        mode = path.lstat().st_mode
        if not stat.S_ISSOCK(mode):
            raise RuntimeError(f"refusing to replace non-socket path: {path}")
        path.unlink()

    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        listener.bind(str(path))
        path.chmod(0o600)
        listener.listen(64)
        listener.set_inheritable(True)
        return listener
    except BaseException:
        listener.close()
        if path.is_socket():
            path.unlink()
        raise


def main() -> None:
    raw_path = os.environ.get(
        "TTS_SOCKET_PATH",
        "/home/worker/.local/state/tts-runtime/service.sock",
    ).strip()
    if not raw_path:
        raise ValueError("TTS_SOCKET_PATH must not be empty")
    listener = create_private_socket(Path(raw_path))
    arguments = [
        sys.executable,
        "-m",
        "uvicorn",
        "tts_runtime.api:app",
        "--fd",
        str(listener.fileno()),
        "--workers",
        "1",
        "--no-access-log",
        "--timeout-keep-alive",
        "600",
    ]
    os.execv(sys.executable, arguments)


if __name__ == "__main__":
    main()
