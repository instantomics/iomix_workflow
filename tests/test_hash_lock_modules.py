from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from iomix_workflow.hashing import HashCache
from iomix_workflow.locking import DirectoryLock
from iomix_workflow.models import Module, ModuleConfig
from iomix_workflow.modules import ModuleError, activate_modules


def test_hash_cache_proxy_deep_and_directory_identity(tmp_path: Path) -> None:
    cache = HashCache(tmp_path / "hashes.json")
    data = tmp_path / "data"
    data.mkdir()
    file = data / "x.txt"
    file.write_text("one", encoding="utf-8")
    first = cache.file(file)
    second = cache.file(file)
    assert first == second
    assert cache.directory(data).entries == 1
    file.write_text("two", encoding="utf-8")
    assert cache.file(file).sha256 != first.sha256
    assert cache.file(file, deep=True).sha256 == cache.file(file).sha256


def test_lock_recovers_stale_directory(tmp_path: Path) -> None:
    path = tmp_path / "lock"
    path.mkdir()
    (path / "owner.json").write_text(
        json.dumps({"token": "dead", "heartbeat": time.time() - 100}), encoding="utf-8"
    )
    with DirectoryLock(path, stale_after=0.1, heartbeat_interval=0.02):
        owner = json.loads((path / "owner.json").read_text(encoding="utf-8"))
        assert owner["token"] != "dead"
    assert not path.exists()


def test_bounded_module_activation_and_identity(tmp_path: Path) -> None:
    modulecmd = tmp_path / "modulecmd"
    modulecmd.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = \"--version\" ]; then echo fake-1; exit; fi\n"
        "printf '%s\\n' \"PATH='/module/bin with spaces'\" 'export PATH' "
        "\"LOADED='yes'\" 'export LOADED'\n",
        encoding="utf-8",
    )
    modulecmd.chmod(0o755)
    environment, identity = activate_modules(
        (Module(name="science/1"),),
        ModuleConfig(modulecmd=str(modulecmd), timeout_seconds=1.0),
    )
    assert environment["LOADED"] == "yes"
    assert environment["PATH"] == "/module/bin with spaces"
    assert identity["modules"][0]["name"] == "science/1"  # type: ignore[index]
    assert identity["base_environment_sha256"]
    assert identity["modulecmd"]["sha256"]  # type: ignore[index]


def test_module_activation_rejects_duplicate_modules(tmp_path: Path) -> None:
    with pytest.raises(ModuleError, match="duplicate module"):
        activate_modules(
            (Module(name="same"), Module(name="same")),
            ModuleConfig(modulecmd=str(tmp_path / "unused"), timeout_seconds=1.0),
            base={"PATH": str(tmp_path)},
        )


def test_directory_identity_rejects_symlink_members(tmp_path: Path) -> None:
    directory = tmp_path / "directory"
    directory.mkdir()
    target = tmp_path / "target"
    target.write_text("data", encoding="utf-8")
    (directory / "link").symlink_to(target)
    with pytest.raises(ValueError, match="forbids symlinks"):
        HashCache(tmp_path / "hashes.json").directory(directory)


def test_module_parser_does_not_concatenate_newline_statements(tmp_path: Path) -> None:
    modulecmd = tmp_path / "modulecmd"
    modulecmd.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = \"--version\" ]; then echo fake-1; exit; fi\n"
        "printf '%s\\n' \"FIRST='one'\" 'export FIRST' \"SECOND='two'\" 'export SECOND'\n",
        encoding="utf-8",
    )
    modulecmd.chmod(0o755)
    environment, _ = activate_modules(
        (Module(name="science/1"),),
        ModuleConfig(modulecmd=str(modulecmd), timeout_seconds=1.0),
    )
    assert environment["FIRST"] == "one"
    assert environment["SECOND"] == "two"
