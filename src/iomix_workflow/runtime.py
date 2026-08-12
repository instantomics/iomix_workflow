from __future__ import annotations

import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import __version__
from .canonical import canonical_digest, canonical_json
from .graph import CompiledWorkflow, StepInstance, interpolate, resolve_step_output, token_names
from .hashing import HashCache, HashError, Identity
from .locking import DirectoryLock
from .models import (
    ExternalInput,
    LiteralInput,
    OutputPathInput,
    RepositoryInput,
    StepOutputInput,
    _relative_path,
)
from .modules import activate_modules

_SHELL_VARIABLE = "IOMIX_WORKFLOW_TOKEN_"
_TOKEN_START = "${"


def _shell_references(text: str, variables: dict[str, str]) -> str:
    result: list[str] = []
    quote: str | None = None
    index = 0
    while index < len(text):
        character = text[index]
        if character == "\\" and quote != "'":
            if text.startswith(_TOKEN_START, index + 1):
                raise ExecutionError("shell interpolation placeholders cannot be escaped")
            result.append(character)
            index += 1
            if index < len(text):
                result.append(text[index])
                index += 1
            continue
        if character in {"'", '"'} and (quote is None or quote == character):
            quote = None if quote == character else character
            result.append(character)
            index += 1
            continue
        if text.startswith(_TOKEN_START, index):
            end = text.find("}", index + 2)
            if end >= 0:
                name = text[index + 2 : end]
                variable = variables.get(name)
                if variable is not None:
                    reference = f"${{{variable}}}"
                    if quote is None:
                        result.append(f'"{reference}"')
                    elif quote == "'":
                        result.append(f"'\"{reference}\"'")
                    else:
                        result.append(reference)
                    index = end + 1
                    continue
        result.append(character)
        index += 1
    return "".join(result)


class ExecutionError(RuntimeError):
    pass


class FingerprintDeferred(ExecutionError):
    """An exact fingerprint needs a dependency output that does not exist yet."""


@dataclass(frozen=True)
class RuntimePaths:
    repository: Path
    output: Path
    workflow_file: Path
    state: Path
    logs: Path
    stages: Path
    cache: Path
    locks: Path


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(canonical_json(value) + b"\n")
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def runtime_paths(workflow_file: Path, output: Path, workflow_id: str) -> RuntimePaths:
    repository = workflow_file.resolve().parent
    root = output.resolve() / ".iomix" / "workflow" / workflow_id
    return RuntimePaths(
        repository=repository,
        output=output.resolve(),
        workflow_file=workflow_file.resolve(),
        state=root / "state",
        logs=root / "logs",
        stages=root / "stages",
        cache=root / "cache",
        locks=root / "locks",
    )


class Runtime:
    def __init__(
        self,
        compiled: CompiledWorkflow,
        paths: RuntimePaths,
        *,
        deep: bool = False,
    ) -> None:
        self.compiled = compiled
        self.paths = paths
        self.deep = deep
        self.hashes = HashCache(paths.output / ".iomix" / "hashes" / "files.json")

    def _safe(self, root: Path, relative: str) -> Path:
        root = root.resolve()
        lexical = root / relative
        current = root
        for part in Path(relative).parts[:-1]:
            current /= part
            if current.is_symlink():
                raise ExecutionError(f"path traverses a symlink: {relative}")
        if lexical.is_symlink():
            raise ExecutionError(f"path is a symlink: {relative}")
        candidate = lexical.resolve()
        if candidate != root and root not in candidate.parents:
            raise ExecutionError(f"path escapes root {root}: {relative}")
        return candidate

    def _outputs(self, instance: StepInstance, *, stage: Path | None = None) -> dict[str, Path]:
        parameters = {**self.compiled.workflow.parameters, **instance.step.parameters}
        values = {f"foreach.{key}": value for key, value in instance.axes.items()}
        values |= {f"parameters.{key}": value for key, value in parameters.items()}
        root = stage or self.paths.output
        result = {}
        for output_id, output in instance.step.outputs.items():
            relative = interpolate(output.path, values)
            try:
                _relative_path(relative, field="interpolated output path")
            except ValueError as error:
                raise ExecutionError(str(error)) from error
            result[output_id] = self._safe(root, relative)
        return result

    def _assert_distinct_outputs(self, instance: StepInstance) -> None:
        paths = sorted(
            ((path, output_id) for output_id, path in self._outputs(instance).items()),
            key=lambda item: str(item[0]),
        )
        for index, (path, output_id) in enumerate(paths):
            for other, other_id in paths[index + 1 :]:
                if path in other.parents or other in path.parents:
                    raise ExecutionError(
                        f"declared outputs overlap: {output_id!r} ({path}) and "
                        f"{other_id!r} ({other})"
                    )

    def _inputs(self, instance: StepInstance) -> tuple[dict[str, Any], dict[str, Any]]:
        values: dict[str, Any] = {}
        identities: dict[str, Any] = {}
        for input_id, declaration in instance.step.inputs.items():
            if isinstance(declaration, RepositoryInput):
                path = self._safe(self.paths.repository, declaration.repository_path)
                identity = self.hashes.identity(path, deep=self.deep)
                values[input_id], identities[input_id] = path, identity.as_dict()
            elif isinstance(declaration, OutputPathInput):
                path = self._safe(self.paths.output, declaration.output_path)
                identity = self.hashes.identity(path, deep=self.deep)
                values[input_id], identities[input_id] = path, identity.as_dict()
            elif isinstance(declaration, ExternalInput):
                path = Path(declaration.external_path).resolve()
                identity = self.hashes.identity(path, deep=self.deep)
                if declaration.sha256 and declaration.sha256 != identity.sha256:
                    raise ExecutionError(f"external input {input_id!r} SHA-256 mismatch")
                if (
                    declaration.size_bytes is not None
                    and declaration.size_bytes != identity.size_bytes
                ):
                    raise ExecutionError(f"external input {input_id!r} size mismatch")
                values[input_id], identities[input_id] = path, identity.as_dict()
            elif isinstance(declaration, LiteralInput):
                values[input_id] = declaration.literal
                identities[input_id] = {
                    "kind": "literal",
                    "sha256": canonical_digest(declaration.literal),
                }
            elif isinstance(declaration, StepOutputInput):
                producer_id, output_id = resolve_step_output(self.compiled, instance, declaration)
                producer = self.compiled.instances[producer_id]
                path = self._outputs(producer)[output_id]
                try:
                    identity = self.hashes.identity(
                        path,
                        kind=producer.step.outputs[output_id].kind,
                        deep=self.deep,
                    )
                except (HashError, OSError) as error:
                    raise FingerprintDeferred(
                        f"dependency output is not available: {producer_id}.{output_id}: {error}"
                    ) from error
                values[input_id], identities[input_id] = path, {
                    **identity.as_dict(),
                    "producer": producer_id,
                    "output": output_id,
                }
        return values, identities

    def _repository_identity(self, instance: StepInstance) -> dict[str, Any]:
        files: dict[str, Any] = {}
        run = instance.step.run
        scripts = tuple(item for item in (run.python, run.bash) if item)
        for relative in (*scripts, "pyproject.toml", "uv.lock"):
            path = self.paths.repository / relative
            if path.is_file():
                files[relative] = self.hashes.file(path, deep=self.deep).as_dict()
            elif relative in scripts:
                raise ExecutionError(f"declared script does not exist: {relative}")
            else:
                files[relative] = {"present": False}
        for input_id, declaration in instance.step.inputs.items():
            if isinstance(declaration, RepositoryInput):
                files[f"input:{input_id}"] = self.hashes.identity(
                    self.paths.repository / declaration.repository_path, deep=self.deep
                ).as_dict()
        return files

    def _module_environment(self, instance: StepInstance) -> tuple[dict[str, str], dict[str, Any]]:
        modules = (
            *self.compiled.workflow.modules,
            *instance.step.modules,
            *(
                module
                for tool in instance.step.tools
                for module in self.compiled.workflow.tools[tool].modules
            ),
        )
        return activate_modules(modules, self.compiled.workflow.module_config)

    def _resolve_executable(self, value: str, environment: dict[str, str]) -> Path | None:
        if "/" in value:
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = self.paths.repository / candidate
            executable = shutil.which(str(candidate), path=environment.get("PATH"))
        else:
            executable = shutil.which(value, path=environment.get("PATH"))
        return Path(executable).resolve() if executable else None

    def _declared_executables(
        self, instance: StepInstance, environment: dict[str, str]
    ) -> dict[str, Path]:
        result: dict[str, Path] = {}
        for name in instance.step.tools:
            tool = self.compiled.workflow.tools[name]
            executable = self._resolve_executable(tool.executable, environment)
            if executable is None:
                raise ExecutionError(f"tool {name!r} executable not found: {tool.executable}")
            result[name] = executable
        return result

    def _probe(self, argv: list[str], environment: dict[str, str], timeout: float) -> str:
        limit = 64 * 1024
        with tempfile.TemporaryFile() as output:
            try:
                process = subprocess.Popen(
                    argv,
                    cwd=self.paths.repository,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    return_code = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
                    raise ExecutionError(f"tool probe timed out after {timeout} seconds") from None
            except OSError as error:
                raise ExecutionError(f"tool probe failed: {error}") from error
            if return_code:
                raise ExecutionError(f"tool probe exited {return_code}")
            size = output.tell()
            if size > limit:
                raise ExecutionError(f"tool probe output exceeds {limit} bytes")
            output.seek(0)
            return output.read(limit + 1).decode("utf-8", errors="replace")

    def _tool_identities(
        self,
        instance: StepInstance,
        environment: dict[str, str],
    ) -> dict[str, Any]:
        identities: dict[str, Any] = {}
        executables = self._declared_executables(instance, environment)
        for name, executable_path in executables.items():
            tool = self.compiled.workflow.tools[name]
            probe_executable = self._resolve_executable(tool.version.argv[0], environment)
            if probe_executable != executable_path:
                raise ExecutionError(
                    f"tool {name!r} version probe must execute its declared executable"
                )
            probe = self._probe(
                [str(executable_path), *tool.version.argv[1:]],
                environment,
                tool.version.timeout_seconds,
            )
            output = "\n".join(line.rstrip() for line in probe.strip().splitlines())
            identities[name] = {
                "executable_identity": self.hashes.file(
                    executable_path, deep=self.deep
                ).as_dict(),
                "version": output,
                "version_sha256": canonical_digest(output),
            }
        return identities

    def _engine_identities(self, environment: dict[str, str]) -> dict[str, Any]:
        python = Path(sys.executable).resolve()
        value: dict[str, Any] = {
            "python": {
                "identity": self.hashes.file(python, deep=self.deep).as_dict(),
                "implementation": sys.implementation.name,
                "version": list(sys.version_info[:3]),
                "cache_tag": sys.implementation.cache_tag,
                "compiler": platform.python_compiler(),
            }
        }
        bash = self._resolve_executable("bash", environment)
        if bash is not None:
            value["bash"] = {
                "identity": self.hashes.file(bash, deep=self.deep).as_dict(),
            }
        return value

    def _run_executable_identity(
        self,
        instance: StepInstance,
        inputs: dict[str, Any],
        environment: dict[str, str],
    ) -> dict[str, Any]:
        run = instance.step.run
        if run.python:
            executable = Path(sys.executable).resolve()
            owner = "python"
        else:
            bash = self._resolve_executable("bash", environment)
            if run.bash or run.shell:
                if bash is None:
                    raise ExecutionError("engine-owned Bash executable was not found")
                executable, owner = bash, "bash"
            else:
                parameters = {
                    **self.compiled.workflow.parameters,
                    **instance.step.parameters,
                }
                tokens = {f"foreach.{key}": value for key, value in instance.axes.items()}
                tokens |= {f"parameters.{key}": value for key, value in parameters.items()}
                tokens |= {f"inputs.{key}": value for key, value in inputs.items()}
                requested_value = interpolate((run.argv or ("",))[0], tokens)
                if not requested_value:
                    raise ExecutionError("interpolated argv executable must not be empty")
                requested = self._resolve_executable(requested_value, environment)
                if requested is None:
                    raise ExecutionError(f"argv executable was not found: {requested_value}")
                if requested == Path(sys.executable).resolve():
                    executable, owner = requested, "python"
                elif bash is not None and requested == bash:
                    executable, owner = requested, "bash"
                else:
                    matches = [
                        name
                        for name, path in self._declared_executables(
                            instance, environment
                        ).items()
                        if path == requested
                    ]
                    if len(matches) != 1:
                        raise ExecutionError(
                            f"argv executable {requested_value!r} must correspond to exactly "
                            "one declared tool"
                        )
                    executable, owner = requested, f"tool:{matches[0]}"
        return {
            "owner": owner,
            "identity": self.hashes.file(executable, deep=self.deep).as_dict(),
        }

    def fingerprint(
        self,
        instance: StepInstance,
    ) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, str]]:
        input_values, input_identities = self._inputs(instance)
        environment, module_identity = self._module_environment(instance)
        value = {
            "schema": self.compiled.workflow.schema_,
            "engine_version": __version__,
            "workflow_digest": canonical_digest(
                self.compiled.workflow.model_dump(mode="json", by_alias=True, exclude_none=False)
            ),
            "step": instance.step.model_dump(mode="json", exclude_none=False),
            "instance": {"id": instance.instance_id, "axes": instance.axes},
            "inputs": input_identities,
            "parameters": {**self.compiled.workflow.parameters, **instance.step.parameters},
            "repository": self._repository_identity(instance),
            "engines": self._engine_identities(environment),
            "run_executable": self._run_executable_identity(
                instance, input_values, environment
            ),
            "modules": module_identity,
            "tools": self._tool_identities(instance, environment),
            "resources": instance.step.resources.model_dump(mode="json"),
        }
        return canonical_digest(value), value, input_values, environment

    def submission_identity(self, instance: StepInstance) -> str:
        workflow = self.compiled.workflow
        return canonical_digest(
            {
                "schema": workflow.schema_,
                "engine_version": __version__,
                "workflow_id": workflow.workflow_id,
                "instance": {"id": instance.instance_id, "axes": instance.axes},
                "step": instance.step.model_dump(mode="json", exclude_none=False),
                "dependencies": instance.dependencies,
                "parameters": workflow.parameters,
                "modules": [item.model_dump(mode="json") for item in workflow.modules],
                "module_config": workflow.module_config.model_dump(mode="json"),
                "tools": {
                    name: workflow.tools[name].model_dump(mode="json")
                    for name in instance.step.tools
                },
                "repository": self._repository_identity(instance),
                "bootstrap_argv": workflow.slurm.bootstrap_argv,
            }
        )

    def state(self, instance_id: str) -> dict[str, Any] | None:
        return read_json(self.paths.state / f"{canonical_digest(instance_id)}.json")

    def _state_path(self, instance_id: str) -> Path:
        return self.paths.state / f"{canonical_digest(instance_id)}.json"

    def _state_lock(self, instance_id: str) -> DirectoryLock:
        return DirectoryLock(self.paths.locks / f"{canonical_digest(instance_id)}.state")

    def _write_state(self, instance_id: str, **values: Any) -> None:
        with self._state_lock(instance_id):
            current = self.state(instance_id) or {"instance_id": instance_id}
            atomic_json(
                self._state_path(instance_id),
                {**current, **values, "updated_at": time.time()},
            )

    def write_submission_state(
        self,
        instance_id: str,
        job_id: str,
        *,
        expected_updated_at: float | None,
        **values: Any,
    ) -> dict[str, Any]:
        with self._state_lock(instance_id):
            current = self.state(instance_id) or {"instance_id": instance_id}
            if (
                current.get("status") == "completed"
                and current.get("updated_at") != expected_updated_at
            ):
                return current
            if current.get("slurm_job_id") == job_id and current.get("status") in {
                "running",
                "completed",
                "failed",
            }:
                return current
            updated = {
                **current,
                **values,
                "status": "submitted",
                "executor": "slurm",
                "slurm_job_id": job_id,
                "updated_at": time.time(),
            }
            atomic_json(self._state_path(instance_id), updated)
            return updated

    def _write_execution_state(
        self,
        instance_id: str,
        *,
        executor: str,
        **values: Any,
    ) -> bool:
        with self._state_lock(instance_id):
            current = self.state(instance_id) or {"instance_id": instance_id}
            job_id = os.environ.get("SLURM_JOB_ID") if executor == "slurm" else None
            if job_id and current.get("slurm_job_id") not in {None, job_id}:
                return False
            atomic_json(
                self._state_path(instance_id),
                {
                    **current,
                    **values,
                    **({"slurm_job_id": job_id} if job_id else {}),
                    "executor": executor,
                    "updated_at": time.time(),
                },
            )
            return True

    def _complete_execution(
        self,
        instance: StepInstance,
        stage: Path,
        outputs: dict[str, Any],
        *,
        executor: str,
    ) -> None:
        with self._state_lock(instance.instance_id):
            current = self.state(instance.instance_id) or {
                "instance_id": instance.instance_id
            }
            job_id = os.environ.get("SLURM_JOB_ID") if executor == "slurm" else None
            if job_id and current.get("slurm_job_id") not in {None, job_id}:
                raise ExecutionError(f"obsolete SLURM worker for {instance.instance_id}")
            self._publish(instance, stage)
            atomic_json(
                self._state_path(instance.instance_id),
                {
                    **current,
                    "status": "completed",
                    "outputs": outputs,
                    "completed_at": time.time(),
                    "cache_explanation": "completed and atomically published",
                    "updated_at": time.time(),
                },
            )

    def cache_status(self, instance: StepInstance, fingerprint: str) -> tuple[bool, str]:
        self._assert_distinct_outputs(instance)
        record = self.state(instance.instance_id)
        if not record:
            return False, "no successful cache record"
        if record.get("status") != "completed":
            return False, f"prior status is {record.get('status', 'unknown')}"
        if record.get("fingerprint") != fingerprint:
            return False, "fingerprint changed"
        for output_id, output in instance.step.outputs.items():
            path = self._outputs(instance)[output_id]
            if not path.exists():
                return False, f"output {output_id!r} is missing"
            expected = record.get("outputs", {}).get(output_id, {})
            try:
                actual = self.hashes.identity(path, kind=output.kind, deep=self.deep)
            except (OSError, ValueError, ExecutionError) as error:
                return False, f"output {output_id!r} is invalid: {error}"
            if (
                expected.get("sha256") != actual.sha256
                or expected.get("size_bytes") != actual.size_bytes
            ):
                return False, f"output {output_id!r} identity changed"
        return True, "exact fingerprint and declared outputs are present"

    def plan(self, selected: tuple[str, ...], *, no_cache: bool = False) -> list[dict[str, Any]]:
        result = []
        for instance_id in selected:
            instance = self.compiled.instances[instance_id]
            try:
                fingerprint, _, _, _ = self.fingerprint(instance)
                hit, explanation = self.cache_status(instance, fingerprint)
            except (ExecutionError, OSError, ValueError) as error:
                fingerprint, hit, explanation = None, False, f"unresolved: {error}"
            if no_cache:
                hit, explanation = False, "cache disabled"
            result.append(
                {
                    "instance_id": instance_id,
                    "dependencies": list(instance.dependencies),
                    "fingerprint": fingerprint,
                    "cache": "hit" if hit else "miss",
                    "explanation": explanation,
                }
            )
        return result

    def _request(
        self,
        instance: StepInstance,
        stage: Path,
        inputs: dict[str, Any],
        environment: dict[str, str],
    ) -> tuple[list[str], dict[str, str], Path]:
        outputs = self._outputs(instance, stage=stage)
        parameters = {**self.compiled.workflow.parameters, **instance.step.parameters}
        tokens = {f"foreach.{key}": value for key, value in instance.axes.items()}
        tokens |= {f"parameters.{key}": value for key, value in parameters.items()}
        tokens |= {f"inputs.{key}": value for key, value in inputs.items()}
        tokens |= {f"outputs.{key}": value for key, value in outputs.items()}
        request = {
            "instance_id": instance.instance_id,
            "foreach": instance.axes,
            "inputs": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in inputs.items()
            },
            "outputs": {key: str(value) for key, value in outputs.items()},
            "parameters": parameters,
        }
        request_path = stage.parent / "request.json"
        atomic_json(request_path, request)
        for path in outputs.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        env = dict(environment)
        env["IOMIX_WORKFLOW_REQUEST"] = str(request_path)
        for namespace, values in (("INPUT", inputs), ("OUTPUT", outputs)):
            for key, value in values.items():
                normalized = key.upper().replace(".", "_").replace("-", "_")
                env[f"IOMIX_WORKFLOW_{namespace}_{normalized}"] = str(value)
        run = instance.step.run
        arguments = [interpolate(value, tokens) for value in run.args]
        if run.python:
            command = [
                str(Path(sys.executable).resolve()),
                str(self._safe(self.paths.repository, run.python)),
                *arguments,
            ]
        elif run.bash:
            bash = self._resolve_executable("bash", environment)
            if bash is None:
                raise ExecutionError("engine-owned Bash executable was not found")
            command = [str(bash), str(self._safe(self.paths.repository, run.bash)), *arguments]
        elif run.argv:
            command = [interpolate(value, tokens) for value in run.argv]
            if not command[0]:
                raise ExecutionError("interpolated argv executable must not be empty")
            requested = self._resolve_executable(command[0], environment)
            if requested is None:
                raise ExecutionError(f"argv executable was not found: {command[0]}")
            engines = {Path(sys.executable).resolve()}
            bash = self._resolve_executable("bash", environment)
            if bash is not None:
                engines.add(bash)
            matches = [
                path
                for path in self._declared_executables(instance, environment).values()
                if path == requested
            ]
            if requested not in engines and len(matches) != 1:
                raise ExecutionError(
                    f"argv executable {command[0]!r} must correspond to exactly one declared tool"
                )
            command[0] = str(requested)
        else:
            references: dict[str, str] = {}
            for index, name in enumerate(sorted(set(token_names(run.shell or "")))):
                value = tokens[name]
                if isinstance(value, (dict, list, tuple)):
                    raise ExecutionError(f"shell interpolation token must be scalar: ${{{name}}}")
                variable = f"{_SHELL_VARIABLE}{index}"
                env[variable] = "" if value is None else str(value)
                references[name] = variable
            shell = _shell_references(run.shell or "", references)
            command = ["bash", "-o", "errexit", "-o", "nounset", "-o", "pipefail", "-c", shell]
        return command, env, request_path

    def _validate_staged(self, instance: StepInstance, stage: Path) -> dict[str, Identity]:
        result: dict[str, Identity] = {}
        for output_id, output in instance.step.outputs.items():
            path = self._outputs(instance, stage=stage)[output_id]
            if output.kind == "file" and (not path.is_file() or path.is_symlink()):
                raise ExecutionError(
                    f"declared file output is missing or invalid: {output_id} ({path})"
                )
            if output.kind == "directory" and (not path.is_dir() or path.is_symlink()):
                raise ExecutionError(
                    f"declared directory output is missing or invalid: {output_id} ({path})"
                )
            result[output_id] = self.hashes.identity(path, kind=output.kind, deep=True)
        return result

    def _publish(self, instance: StepInstance, stage: Path) -> None:
        staged = self._outputs(instance, stage=stage)
        final = self._outputs(instance)
        backups: dict[str, Path] = {}
        published: list[str] = []
        for output_id in sorted(staged):
            source, destination = staged[output_id], final[output_id]
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                if source.stat().st_dev != destination.parent.stat().st_dev:
                    raise ExecutionError(
                        f"staging and output are not on the same filesystem: {destination}"
                    )
            except OSError as error:
                raise ExecutionError(f"cannot validate publication filesystem: {error}") from error
        try:
            for output_id in sorted(staged):
                destination = final[output_id]
                previous = destination.with_name(
                    f".{destination.name}.previous.{os.getpid()}.{time.time_ns()}"
                )
                backups[output_id] = previous
                if destination.exists():
                    os.replace(destination, previous)
            for output_id in sorted(staged):
                os.replace(staged[output_id], final[output_id])
                published.append(output_id)
        except Exception:
            for output_id in reversed(published):
                destination = final[output_id]
                if destination.is_dir():
                    shutil.rmtree(destination)
                elif destination.exists():
                    destination.unlink()
            for output_id, previous in backups.items():
                if previous.exists():
                    os.replace(previous, final[output_id])
            raise
        for previous in backups.values():
            if previous.is_dir():
                shutil.rmtree(previous)
            elif previous.exists():
                previous.unlink()

    def _execution_locks(self, instance: StepInstance) -> list[DirectoryLock]:
        locks = {
            self.paths.locks / canonical_digest(instance.instance_id),
            *(
                self.paths.output / ".iomix" / "locks" / canonical_digest(str(path))
                for path in self._outputs(instance).values()
            ),
        }
        return [DirectoryLock(path) for path in sorted(locks, key=str)]

    def run_local(
        self, instance_id: str, *, no_cache: bool = False, executor: str = "local"
    ) -> dict[str, Any]:
        instance = self.compiled.instances[instance_id]
        self._assert_distinct_outputs(instance)
        fingerprint, details, inputs, environment = self.fingerprint(instance)
        hit, explanation = self.cache_status(instance, fingerprint)
        if hit and not no_cache:
            return {"instance_id": instance_id, "status": "cached", "explanation": explanation}
        with ExitStack() as locks:
            for lock in self._execution_locks(instance):
                locks.enter_context(lock)
            fingerprint, details, inputs, environment = self.fingerprint(instance)
            hit, explanation = self.cache_status(instance, fingerprint)
            if hit and not no_cache:
                return {"instance_id": instance_id, "status": "cached", "explanation": explanation}
            stage_name = (
                f"{canonical_digest(instance_id)}.{os.getpid()}.{time.time_ns()}"
            )
            stage_root = self.paths.stages / stage_name
            stage = stage_root / "outputs"
            stage.mkdir(parents=True)
            log_path = self.paths.logs / f"{canonical_digest(instance_id)}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            command, environment, request_path = self._request(instance, stage, inputs, environment)
            if not self._write_execution_state(
                instance_id,
                executor=executor,
                status="running",
                fingerprint=fingerprint,
                fingerprint_details=details,
                command=command,
                request=str(request_path),
                log=str(log_path),
                started_at=time.time(),
                cache_explanation="executing because "
                + ("cache disabled" if no_cache else explanation),
            ):
                raise ExecutionError(f"obsolete SLURM worker for {instance_id}")
            try:
                with log_path.open("ab") as log:
                    process = subprocess.Popen(
                        command,
                        cwd=self.paths.repository,
                        env=environment,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    timeout = (
                        instance.step.resources.timeout_seconds
                        or instance.step.resources.time_minutes * 60
                    )
                    try:
                        return_code = process.wait(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait(timeout=10)
                        raise ExecutionError(f"step timed out after {timeout} seconds") from None
                if return_code:
                    raise ExecutionError(f"step exited with status {return_code}")
                identities = self._validate_staged(instance, stage)
                outputs = {name: value.as_dict() for name, value in identities.items()}
                self._complete_execution(
                    instance,
                    stage,
                    outputs,
                    executor=executor,
                )
                return {"instance_id": instance_id, "status": "completed", "outputs": outputs}
            except (ExecutionError, OSError, ValueError) as error:
                self._write_execution_state(
                    instance_id,
                    executor=executor,
                    status="failed",
                    failure=str(error),
                    failed_at=time.time(),
                    cache_explanation="failure never creates valid cache state",
                )
                raise
            finally:
                shutil.rmtree(stage_root, ignore_errors=True)

    def execute_selected(
        self,
        selected: tuple[str, ...],
        *,
        no_cache: bool = False,
    ) -> list[dict[str, Any]]:
        result = []
        selected_set = set(selected)
        missing = sorted(
            dependency
            for instance_id in selected
            for dependency in self.compiled.instances[instance_id].dependencies
            if dependency not in selected_set
        )
        if missing:
            raise ExecutionError(f"selection omits dependencies: {missing}")
        for instance_id in selected:
            instance = self.compiled.instances[instance_id]
            failed = []
            for dependency in instance.dependencies:
                dependency_instance = self.compiled.instances[dependency]
                try:
                    dependency_fingerprint, _, _, _ = self.fingerprint(dependency_instance)
                    valid, _ = self.cache_status(dependency_instance, dependency_fingerprint)
                except (ExecutionError, OSError, ValueError):
                    valid = False
                if not valid:
                    failed.append(dependency)
            if failed:
                raise ExecutionError(f"dependencies did not complete for {instance_id}: {failed}")
            result.append(self.run_local(instance_id, no_cache=no_cache))
        return result
