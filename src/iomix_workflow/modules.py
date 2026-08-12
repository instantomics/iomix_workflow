from __future__ import annotations

import hashlib
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

from .canonical import canonical_digest
from .models import Module, ModuleConfig

_ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DEPENDENCY_ENVIRONMENT = frozenset(
    {
        "CMAKE_PREFIX_PATH",
        "CPATH",
        "LD_LIBRARY_PATH",
        "LIBRARY_PATH",
        "LMOD_CMD",
        "LOADEDMODULES",
        "MODULEPATH",
        "MODULESHOME",
        "PKG_CONFIG_PATH",
        "_LMFILES_",
    }
)


class ModuleError(RuntimeError):
    pass


def _dependency_environment(environment: dict[str, str]) -> dict[str, str]:
    """Keep only variables that can select module providers or dependencies."""
    return {key: environment[key] for key in sorted(_DEPENDENCY_ENVIRONMENT & environment.keys())}


def _provider_digest(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise ModuleError(f"cannot fingerprint module provider {path}: {error}") from error
    return digest.hexdigest()


def _parse_mutations(output: str) -> list[tuple[str, ...]]:
    lexer = shlex.shlex(output, posix=True, punctuation_chars=";")
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        tokens = list(lexer)
    except ValueError as error:
        raise ModuleError(f"invalid modulecmd shell output: {error}") from error
    mutations: list[tuple[str, ...]] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token and set(token) == {";"}:
            index += 1
            continue
        if token in {"export", "unset"}:
            operation = token
            index += 1
            keys: list[str] = []
            while index < len(tokens) and set(tokens[index]) != {";"}:
                if tokens[index] in {"export", "unset"} or "=" in tokens[index]:
                    break
                keys.append(tokens[index])
                index += 1
            if not keys or not all(_ENV_KEY.fullmatch(key) for key in keys):
                raise ModuleError(f"unsupported modulecmd {operation} statement")
            mutations.append((operation, *keys))
            continue
        if token == "test" and tokens[index : index + 4] == ["test", "0", "=", "1"]:
            mutations.append(("assert_false",))
            index += 4
            continue
        if "=" in token:
            key, value = token.split("=", 1)
            if not _ENV_KEY.fullmatch(key):
                raise ModuleError(f"unsupported modulecmd assignment: {token!r}")
            mutations.append(("set", key, value))
            index += 1
            continue
        raise ModuleError(f"unsupported modulecmd shell token: {token!r}")
    return mutations


def activate_modules(
    modules: tuple[Module, ...], config: ModuleConfig, *, base: dict[str, str] | None = None
) -> tuple[dict[str, str], dict[str, object]]:
    environment = dict(os.environ if base is None else base)
    if not modules:
        dependency_environment = _dependency_environment(environment)
        return environment, {
            "modules": [],
            "modulecmd": None,
            "dependency_environment": dependency_environment,
            "base_environment_sha256": canonical_digest(dependency_environment),
        }
    names = [module.name for module in modules]
    if len(names) != len(set(names)):
        raise ModuleError("duplicate module declarations are not permitted")
    executable = shutil.which(config.modulecmd, path=environment.get("PATH"))
    if executable is None:
        raise ModuleError(f"modulecmd does not exist: {config.modulecmd}")
    executable_path = Path(executable).resolve()
    if not executable_path.is_file():
        raise ModuleError(f"modulecmd is not a regular file: {executable_path}")
    base_environment = _dependency_environment(environment)
    identities: list[dict[str, object]] = []
    for module in modules:
        try:
            result = subprocess.run(
                [str(executable_path), "sh", "load", module.name],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                env=environment,
                timeout=config.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ModuleError(f"module activation failed for {module.name!r}: {error}") from error
        if result.returncode:
            detail = result.stderr.strip() or result.stdout.strip()
            raise ModuleError(f"module activation failed for {module.name!r}: {detail}")
        mutations = _parse_mutations(result.stdout)
        for mutation in mutations:
            operation, *values = mutation
            if operation == "set":
                environment[values[0]] = values[1]
            elif operation == "unset":
                for key in values:
                    environment.pop(key, None)
            elif operation == "assert_false":
                raise ModuleError(f"module activation failed for {module.name!r}")
            elif any(key not in environment for key in values):
                raise ModuleError(f"modulecmd exported an unset variable: {values}")
        identities.append(
            {
                "name": module.name,
                "mutations": mutations,
                "activation_sha256": canonical_digest(mutations),
            }
        )
    try:
        version = subprocess.run(
            [str(executable_path), "--version"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=config.timeout_seconds,
            check=False,
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        version = "unavailable"
    return environment, {
        "modules": identities,
        "dependency_environment": base_environment,
        "base_environment_sha256": canonical_digest(base_environment),
        "modulecmd": {
            "sha256": _provider_digest(executable_path),
            "version": version,
        },
    }
