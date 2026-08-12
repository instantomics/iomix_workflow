from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import canonical_digest, canonical_json
from .locking import DirectoryLock


class HashError(ValueError):
    pass


@dataclass(frozen=True)
class Identity:
    kind: str
    sha256: str
    size_bytes: int
    entries: int | None = None

    def as_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "kind": self.kind,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }
        if self.entries is not None:
            value["entries"] = self.entries
        return value


class HashCache:
    """Stat-proxy cache. SHA-256 remains the only content identity."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.records: dict[str, dict[str, Any]] = {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                self.records = value
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            pass

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with DirectoryLock(
            self.path.with_name(f".{self.path.name}.lock"),
            acquire_timeout=30,
            stale_after=300,
        ):
            try:
                latest = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(latest, dict):
                    latest.update(self.records)
                    self.records = latest
            except (FileNotFoundError, OSError, json.JSONDecodeError):
                pass
            temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
            temporary.write_bytes(canonical_json(self.records) + b"\n")
            os.replace(temporary, self.path)

    def file(self, path: Path, *, deep: bool = False) -> Identity:
        try:
            status = path.lstat()
        except OSError as error:
            raise HashError(f"cannot stat {path}: {error}") from error
        if not stat.S_ISREG(status.st_mode):
            raise HashError(f"not a regular file: {path}")
        key = str(path.resolve())
        proxy = {
            "size": status.st_size,
            "mtime_ns": status.st_mtime_ns,
            "ctime_ns": status.st_ctime_ns,
            "device": status.st_dev,
            "inode": status.st_ino,
        }
        record = self.records.get(key)
        if not deep and record and all(record.get(name) == value for name, value in proxy.items()):
            return Identity("file", record["sha256"], status.st_size)
        digest = hashlib.sha256()
        try:
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
        except OSError as error:
            raise HashError(f"cannot hash {path}: {error}") from error
        second = path.stat()
        if (second.st_size, second.st_mtime_ns, second.st_ctime_ns) != (
            status.st_size,
            status.st_mtime_ns,
            status.st_ctime_ns,
        ):
            raise HashError(f"file changed while hashing: {path}")
        self.records[key] = {**proxy, "sha256": digest.hexdigest()}
        self._save()
        return Identity("file", digest.hexdigest(), status.st_size)

    def directory(self, path: Path, *, deep: bool = False) -> Identity:
        if not path.is_dir() or path.is_symlink():
            raise HashError(f"not a real directory: {path}")
        members: list[dict[str, Any]] = []
        total = 0
        for member in sorted(path.rglob("*"), key=lambda item: item.relative_to(path).as_posix()):
            relative = member.relative_to(path).as_posix()
            if member.is_symlink():
                raise HashError(f"directory identity forbids symlinks: {member}")
            if member.is_dir():
                members.append({"path": relative, "kind": "directory"})
            elif member.is_file():
                identity = self.file(member, deep=deep)
                total += identity.size_bytes
                members.append({"path": relative, **identity.as_dict()})
            else:
                raise HashError(f"directory identity forbids special files: {member}")
        return Identity("directory", canonical_digest(members), total, len(members))

    def identity(self, path: Path, *, kind: str | None = None, deep: bool = False) -> Identity:
        actual = kind or ("directory" if path.is_dir() else "file")
        if actual == "directory":
            return self.directory(path, deep=deep)
        return self.file(path, deep=deep)
