from __future__ import annotations

import json
import os
import shutil
import socket
import threading
import time
import uuid
from pathlib import Path

from .canonical import canonical_json


class LockError(RuntimeError):
    pass


class DirectoryLock:
    """An NFS-safe mkdir lock with owner tokens, heartbeat, and stale takeover."""

    def __init__(
        self,
        path: Path,
        *,
        acquire_timeout: float = 60.0,
        stale_after: float = 300.0,
        heartbeat_interval: float = 10.0,
    ) -> None:
        if min(acquire_timeout, stale_after, heartbeat_interval) <= 0:
            raise ValueError("lock timeouts must be positive")
        self.path = path
        self.acquire_timeout = acquire_timeout
        self.stale_after = stale_after
        self.heartbeat_interval = min(heartbeat_interval, stale_after / 3)
        self.token = uuid.uuid4().hex
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def owner_path(self) -> Path:
        return self.path / "owner.json"

    def _write_owner(self) -> None:
        temporary = self.path / f"owner.{self.token}.tmp"
        temporary.write_bytes(
            canonical_json(
                {
                    "token": self.token,
                    "host": socket.gethostname(),
                    "pid": os.getpid(),
                    "heartbeat": time.time(),
                }
            )
            + b"\n"
        )
        os.replace(temporary, self.owner_path)

    def _is_stale(self) -> bool:
        try:
            record = json.loads(self.owner_path.read_text(encoding="utf-8"))
            heartbeat = float(record["heartbeat"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            try:
                heartbeat = self.path.stat().st_mtime
            except OSError:
                return False
        return time.time() - heartbeat > self.stale_after

    def _recover(self) -> None:
        stale = self.path.with_name(f"{self.path.name}.stale.{uuid.uuid4().hex}")
        try:
            os.rename(self.path, stale)
        except OSError:
            return
        shutil.rmtree(stale, ignore_errors=True)

    def acquire(self) -> DirectoryLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.acquire_timeout
        while True:
            try:
                self.path.mkdir()
                self._write_owner()
                break
            except FileExistsError:
                if self._is_stale():
                    self._recover()
                    continue
                if time.monotonic() >= deadline:
                    raise LockError(f"timed out waiting for lock {self.path}") from None
                time.sleep(0.1)
        self._thread = threading.Thread(target=self._heartbeat, daemon=True)
        self._thread.start()
        return self

    def _heartbeat(self) -> None:
        while not self._stop.wait(self.heartbeat_interval):
            try:
                current = json.loads(self.owner_path.read_text(encoding="utf-8"))
                if current.get("token") != self.token:
                    return
                self._write_owner()
            except OSError:
                return

    def release(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.heartbeat_interval + 1)
        try:
            current = json.loads(self.owner_path.read_text(encoding="utf-8"))
            if current.get("token") != self.token:
                raise LockError(f"lock ownership changed for {self.path}")
            self.owner_path.unlink()
            self.path.rmdir()
        except FileNotFoundError:
            return
        except OSError as error:
            raise LockError(f"could not release lock {self.path}: {error}") from error

    def __enter__(self) -> DirectoryLock:
        return self.acquire()

    def __exit__(self, *_: object) -> None:
        self.release()
