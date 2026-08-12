from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from .canonical import canonical_digest, canonical_json
from .models import Step, StepOutputInput, Workflow, _relative_path

_TOKEN = re.compile(r"\$\{([^{}]+)\}")


class GraphError(ValueError):
    pass


@dataclass(frozen=True)
class StepInstance:
    instance_id: str
    step_id: str
    axes: dict[str, Any]
    step: Step
    dependencies: tuple[str, ...]


@dataclass(frozen=True)
class CompiledWorkflow:
    workflow: Workflow
    instances: dict[str, StepInstance]
    order: tuple[str, ...]

    def select(
        self, *, targets: tuple[str, ...] = (), steps: tuple[str, ...] = ()
    ) -> tuple[str, ...]:
        if targets and steps:
            raise GraphError("--target and --step cannot be combined")
        requested = targets or steps
        if not requested:
            return self.order
        direct: set[str] = set()
        for selector in requested:
            if selector in self.instances:
                direct.add(selector)
            elif selector in self.workflow.steps:
                direct.update(i for i, item in self.instances.items() if item.step_id == selector)
            else:
                raise GraphError(f"unknown selector: {selector}")
        if steps:
            return tuple(item for item in self.order if item in direct)
        closure = set(direct)
        todo = list(direct)
        while todo:
            for dependency in self.instances[todo.pop()].dependencies:
                if dependency not in closure:
                    closure.add(dependency)
                    todo.append(dependency)
        return tuple(item for item in self.order if item in closure)


def token_names(text: str) -> tuple[str, ...]:
    return tuple(match.group(1) for match in _TOKEN.finditer(text))


def interpolate(text: str, values: dict[str, Any]) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in values:
            raise GraphError(f"undeclared interpolation token: ${{{name}}}")
        value = values[name]
        if isinstance(value, (dict, list, tuple)):
            raise GraphError(f"interpolation token must be scalar: ${{{name}}}")
        return "" if value is None else str(value)

    return _TOKEN.sub(replace, text)


def _instance_id(step_id: str, axes: dict[str, Any]) -> str:
    if not axes:
        return step_id
    labels = ",".join(
        f"{name}={canonical_digest(value)[:12]}" for name, value in sorted(axes.items())
    )
    return f"{step_id}[{labels}]"


def _expand(step_id: str, step: Step) -> list[tuple[str, dict[str, Any]]]:
    names = sorted(step.foreach)
    products = itertools.product(*(step.foreach[name] for name in names))
    expanded = [
        (_instance_id(step_id, axes), axes)
        for values in products
        for axes in [dict(zip(names, values, strict=True))]
    ]
    instance_ids = [instance_id for instance_id, _ in expanded]
    if len(instance_ids) != len(set(instance_ids)):
        raise GraphError(f"step {step_id!r} expansion produced duplicate instance IDs")
    return expanded


def _match_dependencies(
    consumer_axes: dict[str, Any], producer: list[tuple[str, dict[str, Any]]]
) -> tuple[str, ...]:
    shared = set(consumer_axes) & set(producer[0][1])
    matches = tuple(
        instance_id
        for instance_id, axes in producer
        if all(canonical_json(axes[name]) == canonical_json(consumer_axes[name]) for name in shared)
    )
    if not matches:
        raise GraphError("dependency expansion has no shared-axis match")
    return matches


def _validate_tokens(workflow: Workflow, step_id: str, step: Step) -> None:
    allowed = {f"foreach.{key}" for key in step.foreach}
    allowed |= {f"inputs.{key}" for key in step.inputs}
    allowed |= {f"outputs.{key}" for key in step.outputs}
    allowed |= {f"parameters.{key}" for key in workflow.parameters}
    allowed |= {f"parameters.{key}" for key in step.parameters}
    strings: list[str] = []
    if step.run.python:
        if token_names(step.run.python):
            raise GraphError(f"step {step_id!r} Python script path cannot be interpolated")
        strings.append(step.run.python)
    if step.run.bash:
        if token_names(step.run.bash):
            raise GraphError(f"step {step_id!r} Bash script path cannot be interpolated")
        strings.append(step.run.bash)
    if step.run.argv:
        strings.extend(step.run.argv)
    if step.run.shell:
        strings.append(step.run.shell)
    strings.extend(step.run.args)
    output_allowed = {f"foreach.{key}" for key in step.foreach}
    output_allowed |= {f"parameters.{key}" for key in workflow.parameters}
    output_allowed |= {f"parameters.{key}" for key in step.parameters}
    for output in step.outputs.values():
        unknown = set(token_names(output.path)) - output_allowed
        if unknown:
            raise GraphError(
                f"step {step_id!r} output paths may only use foreach and parameter tokens: "
                f"{sorted(unknown)}"
            )
    for text in strings:
        unknown = set(token_names(text)) - allowed
        if unknown:
            raise GraphError(f"step {step_id!r} uses undeclared tokens: {sorted(unknown)}")


def compile_workflow(workflow: Workflow) -> CompiledWorkflow:
    expanded = {step_id: _expand(step_id, step) for step_id, step in workflow.steps.items()}
    dependencies: dict[str, tuple[str, ...]] = {}
    instances: dict[str, StepInstance] = {}
    for step_id, step in workflow.steps.items():
        _validate_tokens(workflow, step_id, step)
        declared = set(step.needs)
        declared |= {
            value.step
            for value in step.inputs.values()
            if isinstance(value, StepOutputInput)
        }
        unknown = declared - workflow.steps.keys()
        if unknown:
            raise GraphError(f"step {step_id!r} has unknown dependencies: {sorted(unknown)}")
        if step_id in declared:
            raise GraphError(f"step {step_id!r} depends on itself")
        for instance_id, axes in expanded[step_id]:
            matched = tuple(
                dependency
                for needed in sorted(declared)
                for dependency in _match_dependencies(axes, expanded[needed])
            )
            dependencies[instance_id] = matched
            instances[instance_id] = StepInstance(instance_id, step_id, axes, step, matched)

    visiting: set[str] = set()
    visited: set[str] = set()
    order: list[str] = []

    def visit(instance_id: str) -> None:
        if instance_id in visiting:
            raise GraphError(f"workflow contains a cycle involving {instance_id}")
        if instance_id in visited:
            return
        visiting.add(instance_id)
        for dependency in dependencies[instance_id]:
            visit(dependency)
        visiting.remove(instance_id)
        visited.add(instance_id)
        order.append(instance_id)

    for instance_id in sorted(instances):
        visit(instance_id)

    output_paths: dict[str, tuple[str, str]] = {}
    artifacts: dict[str, tuple[str, str, str]] = {}
    for instance_id in order:
        instance = instances[instance_id]
        parameters = {**workflow.parameters, **instance.step.parameters}
        values = {f"foreach.{key}": value for key, value in instance.axes.items()}
        values |= {f"parameters.{key}": value for key, value in parameters.items()}
        for output_id, output in instance.step.outputs.items():
            path = interpolate(output.path, values)
            try:
                _relative_path(path, field="interpolated output path")
            except ValueError as error:
                raise GraphError(f"{instance_id}.{output_id}: {error}") from error
            normalized = str(PurePosixPath(path))
            owner = output_paths.get(normalized)
            if owner:
                raise GraphError(
                    f"output collision at {normalized!r}: {owner[0]}.{owner[1]} and "
                    f"{instance_id}.{output_id}"
                )
            output_paths[normalized] = (instance_id, output_id)
            if output.artifact is not None:
                artifact_id = output.artifact.artifact_id
                owner = artifacts.get(artifact_id)
                if owner:
                    raise GraphError(
                        f"duplicate artifact ID {artifact_id!r}: {owner[0]}.{owner[1]} "
                        f"({owner[2]}) and {instance_id}.{output_id} ({normalized})"
                    )
                artifacts[artifact_id] = (instance_id, output_id, normalized)
    sorted_paths = sorted(output_paths)
    for index, path in enumerate(sorted_paths):
        parent = PurePosixPath(path)
        for other in sorted_paths[index + 1 :]:
            child = PurePosixPath(other)
            if parent in child.parents or child in parent.parents:
                first, second = output_paths[path], output_paths[other]
                raise GraphError(
                    f"output paths overlap: {first[0]}.{first[1]} ({path}) and "
                    f"{second[0]}.{second[1]} ({other})"
                )
    return CompiledWorkflow(workflow, instances, tuple(order))


def resolve_step_output(
    compiled: CompiledWorkflow, consumer: StepInstance, reference: StepOutputInput
) -> tuple[str, str]:
    candidates = [
        item
        for item in consumer.dependencies
        if compiled.instances[item].step_id == reference.step
        and reference.output in compiled.instances[item].step.outputs
    ]
    if len(candidates) != 1:
        raise GraphError(
            f"{consumer.instance_id} input {reference.step}.{reference.output} resolves to "
            f"{len(candidates)} instances; exactly one is required"
        )
    return candidates[0], reference.output
