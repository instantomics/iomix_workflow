from __future__ import annotations

import math
import re
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from .canonical import canonical_json

Identifier = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$", max_length=128),
]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
JsonScalar = str | int | float | bool | None


class Model(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True, validate_default=True)


def _relative_path(value: str, *, field: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or value == "."
        or path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or ".iomix" in path.parts
        or str(path) != value
    ):
        raise ValueError(f"{field} must be a normalized, non-empty relative path")
    return value


class ModuleConfig(Model):
    modulecmd: str = "modulecmd"
    timeout_seconds: Annotated[float, Field(gt=0, le=300)] = 30.0


class Module(Model):
    name: Annotated[str, StringConstraints(min_length=1, max_length=256)]


class VersionProbe(Model):
    argv: tuple[Annotated[str, StringConstraints(min_length=1)], ...]
    timeout_seconds: Annotated[float, Field(gt=0, le=300)] = 30.0

    @model_validator(mode="after")
    def nonempty(self) -> VersionProbe:
        if not self.argv:
            raise ValueError("version argv must not be empty")
        return self


class Tool(Model):
    executable: Annotated[str, StringConstraints(min_length=1)]
    version: VersionProbe
    modules: tuple[Module, ...] = ()


class PythonRun(Model):
    python: str
    args: tuple[str, ...] = ()

    @model_validator(mode="after")
    def path_valid(self) -> PythonRun:
        _relative_path(self.python, field="python")
        return self


class BashRun(Model):
    bash: str
    args: tuple[str, ...] = ()

    @model_validator(mode="after")
    def path_valid(self) -> BashRun:
        _relative_path(self.bash, field="bash")
        return self


class ArgvRun(Model):
    argv: tuple[str, ...]

    @model_validator(mode="after")
    def nonempty(self) -> ArgvRun:
        if not self.argv or not self.argv[0]:
            raise ValueError("argv must not be empty")
        return self


class ShellRun(Model):
    shell: Annotated[str, StringConstraints(min_length=1)]


class Run(Model):
    python: str | None = None
    bash: str | None = None
    argv: tuple[str, ...] | None = None
    shell: str | None = None
    args: tuple[str, ...] = ()

    @model_validator(mode="after")
    def exactly_one(self) -> Run:
        selected = [
            self.python is not None,
            self.bash is not None,
            self.argv is not None,
            self.shell is not None,
        ]
        if sum(selected) != 1:
            raise ValueError("run must declare exactly one of python, bash, argv, or shell")
        if self.python is not None:
            _relative_path(self.python, field="python")
        if self.bash is not None:
            _relative_path(self.bash, field="bash")
        if self.argv is not None and (not self.argv or not self.argv[0]):
            raise ValueError("argv must not be empty")
        if self.shell is not None and not self.shell:
            raise ValueError("shell must not be empty")
        if self.args and self.python is None and self.bash is None:
            raise ValueError("args is only valid for python or bash runs")
        return self


class RepositoryInput(Model):
    repository_path: str

    @model_validator(mode="after")
    def path_valid(self) -> RepositoryInput:
        _relative_path(self.repository_path, field="repository_path")
        return self


class OutputPathInput(Model):
    output_path: str

    @model_validator(mode="after")
    def path_valid(self) -> OutputPathInput:
        _relative_path(self.output_path, field="output_path")
        return self


class StepOutputInput(Model):
    step: Identifier
    output: Identifier


class ExternalInput(Model):
    external_path: Annotated[str, StringConstraints(min_length=1)]
    sha256: Sha256 | None = None
    size_bytes: Annotated[int, Field(ge=0)] | None = None


def _json_value(value: Any, depth: int = 0) -> None:
    if depth > 32:
        raise ValueError("literal exceeds maximum depth")
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("literal floats must be finite")
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _json_value(item, depth + 1)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for item in value.values():
            _json_value(item, depth + 1)
        return
    raise ValueError("literal must be a JSON value")


class LiteralInput(Model):
    literal: Any

    @model_validator(mode="after")
    def literal_valid(self) -> LiteralInput:
        _json_value(self.literal)
        return self


Input = RepositoryInput | OutputPathInput | StepOutputInput | ExternalInput | LiteralInput


class Artifact(Model):
    artifact_id: Identifier
    role: Identifier
    media_type: Annotated[str, StringConstraints(min_length=3, max_length=255)]
    schema_id: Identifier
    schema_version: Annotated[int, Field(ge=1)]
    extensions: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def extensions_valid(self) -> Artifact:
        _json_value(self.extensions)
        return self


class Output(Model):
    path: str
    kind: Literal["file", "directory"] = "file"
    artifact: Artifact | None = None

    @model_validator(mode="after")
    def path_valid(self) -> Output:
        _relative_path(self.path, field="output path")
        token_free = re.sub(r"\$\{[^{}]+\}", "x", self.path)
        _relative_path(token_free, field="interpolated output path")
        return self


class Resources(Model):
    cpus: Annotated[int, Field(ge=1)] = 1
    memory_mb: Annotated[int, Field(ge=1)] = 1024
    time_minutes: Annotated[int, Field(ge=1)] = 60
    gpus: Annotated[int, Field(ge=0)] = 0
    timeout_seconds: Annotated[float, Field(gt=0)] | None = None


class Step(Model):
    run: Run
    needs: tuple[Identifier, ...] = ()
    foreach: dict[Identifier, tuple[JsonScalar, ...]] = Field(default_factory=dict)
    inputs: dict[Identifier, Input] = Field(default_factory=dict)
    outputs: dict[Identifier, Output]
    parameters: dict[Identifier, Any] = Field(default_factory=dict)
    resources: Resources = Field(default_factory=Resources)
    modules: tuple[Module, ...] = ()
    tools: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def valid(self) -> Step:
        if not self.outputs:
            raise ValueError("a step must declare at least one output")
        for axis, values in self.foreach.items():
            if not values:
                raise ValueError(f"foreach axis {axis!r} must not be empty")
            seen: set[bytes] = set()
            for value in values:
                _json_value(value)
                identity = canonical_json(value)
                if identity in seen:
                    raise ValueError(f"foreach axis {axis!r} contains a duplicate canonical value")
                seen.add(identity)
        for value in self.parameters.values():
            _json_value(value)
        return self


class SlurmConfig(Model):
    account: str | None = None
    partition: str | None = None
    sbatch: str = "sbatch"
    squeue: str = "squeue"
    sacct: str = "sacct"
    scancel: str = "scancel"
    command_timeout_seconds: Annotated[float, Field(gt=0, le=300)] = 30.0
    extra_args: tuple[str, ...] = ()
    bootstrap_argv: Annotated[
        tuple[Annotated[str, StringConstraints(min_length=1, max_length=4096)], ...],
        Field(max_length=64),
    ] = ("iom", "env", "exec-receipt", "${receipt}", "--")


class Workflow(Model):
    schema_: Literal["iomix://workflow/v1"] = Field(alias="schema")
    workflow_id: Identifier
    parameters: dict[Identifier, Any] = Field(default_factory=dict)
    modules: tuple[Module, ...] = ()
    module_config: ModuleConfig = Field(default_factory=ModuleConfig)
    tools: dict[Identifier, Tool] = Field(default_factory=dict)
    slurm: SlurmConfig = Field(default_factory=SlurmConfig)
    steps: dict[Identifier, Step]

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        """Return the editor/runtime JSON Schema generated from the validated model."""
        return cls.model_json_schema(mode="validation")

    @model_validator(mode="after")
    def valid(self) -> Workflow:
        if not self.steps:
            raise ValueError("workflow must declare at least one step")
        for value in self.parameters.values():
            _json_value(value)
        for step_id, step in self.steps.items():
            missing_tools = sorted(set(step.tools) - self.tools.keys())
            if missing_tools:
                raise ValueError(f"step {step_id!r} refers to unknown tools: {missing_tools}")
        return self
