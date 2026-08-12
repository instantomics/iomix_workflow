from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .canonical import canonical_digest, canonical_json
from .graph import GraphError, compile_workflow
from .locking import LockError
from .modules import ModuleError
from .runtime import ExecutionError, Runtime, runtime_paths
from .slurm import SlurmExecutor
from .yaml_loader import YamlError, load_workflow


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="iomix-workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "plan", "run", "status", "logs", "cancel"):
        command = subparsers.add_parser(name)
        command.add_argument("--workflow", default="workflow.yaml", help="workflow YAML path")
        command.add_argument("--output-root", default=".", help="workflow output root")
        command.add_argument("--json", action="store_true", help="emit canonical JSON")
        if name in {"plan", "run", "status", "logs", "cancel"}:
            command.add_argument("--target", action="append", default=[])
            command.add_argument("--step", action="append", default=[])
        if name in {"plan", "run"}:
            command.add_argument("--deep", action="store_true")
            command.add_argument("--no-cache", action="store_true")
        if name == "run":
            command.add_argument("--executor", choices=("local", "slurm", "auto"), default="local")
        if name == "status":
            command.add_argument("--refresh", action="store_true")
        if name == "logs":
            command.add_argument("--lines", type=int, default=200)
    worker = subparsers.add_parser("_execute", help=argparse.SUPPRESS)
    worker.add_argument("--workflow", default="workflow.yaml")
    worker.add_argument("--output-root", default=".")
    worker.add_argument("--json", action="store_true")
    worker.add_argument("--instance", required=True)
    worker.add_argument("--deep", action="store_true")
    worker.add_argument("--no-cache", action="store_true")
    return parser


def _emit(value: Any, use_json: bool) -> None:
    if use_json:
        sys.stdout.buffer.write(canonical_json(value) + b"\n")
        return
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                fields = [
                    str(item.get("instance_id", "")),
                    str(item.get("status", item.get("cache", ""))),
                ]
                if item.get("explanation"):
                    fields.append(str(item["explanation"]))
                if item.get("job_id"):
                    fields.append(f"job={item['job_id']}")
                print("\t".join(fields))
            else:
                print(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            print(f"{key}: {item}")
    else:
        print(value)


def _load(arguments: argparse.Namespace) -> tuple[Any, Runtime]:
    workflow_path = Path(arguments.workflow)
    workflow = load_workflow(workflow_path)
    compiled = compile_workflow(workflow)
    deep = bool(getattr(arguments, "deep", False))
    paths = runtime_paths(workflow_path, Path(arguments.output_root), workflow.workflow_id)
    return compiled, Runtime(compiled, paths, deep=deep)


def _selected(compiled: Any, arguments: argparse.Namespace) -> tuple[str, ...]:
    return compiled.select(
        targets=tuple(getattr(arguments, "target", ())),
        steps=tuple(getattr(arguments, "step", ())),
    )


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        compiled, runtime = _load(arguments)
        if arguments.command == "check":
            value = {
                "status": "valid",
                "workflow_id": compiled.workflow.workflow_id,
                "workflow_digest": canonical_digest(
                    compiled.workflow.model_dump(mode="json", by_alias=True, exclude_none=False)
                ),
                "instances": len(compiled.instances),
                "schema": compiled.workflow.model_json_schema(mode="validation"),
            }
        elif arguments.command == "_execute":
            if arguments.instance not in compiled.instances:
                raise ExecutionError(f"unknown instance: {arguments.instance}")
            value = runtime.run_local(
                arguments.instance,
                no_cache=arguments.no_cache,
                executor="slurm",
            )
        else:
            if arguments.command == "run" and arguments.step:
                if arguments.target:
                    raise GraphError("--target and --step cannot be combined")
                selected = compiled.select(targets=tuple(arguments.step))
            else:
                selected = _selected(compiled, arguments)
            if arguments.command == "plan":
                value = runtime.plan(selected, no_cache=arguments.no_cache)
            elif arguments.command == "run":
                executor = arguments.executor
                if executor == "auto":
                    executor = "slurm" if runtime.compiled.workflow.slurm.account else "local"
                value = (
                    runtime.execute_selected(selected, no_cache=arguments.no_cache)
                    if executor == "local"
                    else SlurmExecutor(runtime).submit(selected, no_cache=arguments.no_cache)
                )
            elif arguments.command == "status":
                if arguments.refresh:
                    value = SlurmExecutor(runtime).refresh(selected)
                else:
                    value = [
                        {
                            "instance_id": item,
                            "status": (runtime.state(item) or {}).get("status", "not-run"),
                            "job_id": (runtime.state(item) or {}).get("slurm_job_id"),
                            "failure": (runtime.state(item) or {}).get("failure"),
                            "cache_explanation": (runtime.state(item) or {}).get(
                                "cache_explanation"
                            ),
                        }
                        for item in selected
                    ]
            elif arguments.command == "logs":
                value = []
                for item in selected:
                    state = runtime.state(item) or {}
                    path = Path(
                        state.get(
                            "log",
                            runtime.paths.logs / f"{canonical_digest(item)}.log",
                        )
                    )
                    try:
                        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
                        text = "\n".join(lines[-max(0, arguments.lines) :])
                    except OSError:
                        text = ""
                    value.append(
                        {
                            "instance_id": item,
                            "status": state.get("status", "not-run"),
                            "log": text,
                        }
                    )
                if not arguments.json:
                    for item in value:
                        print(f"==> {item['instance_id']} ({item['status']}) <==")
                        print(item["log"])
                    return 0
            elif arguments.command == "cancel":
                value = SlurmExecutor(runtime).cancel(selected)
            else:
                parser.error("unknown command")
        _emit(value, arguments.json)
        return 0
    except (
        YamlError,
        GraphError,
        ExecutionError,
        ModuleError,
        LockError,
        OSError,
        ValueError,
    ) as error:
        print(f"iomix-workflow: {error}", file=sys.stderr)
        return 2
