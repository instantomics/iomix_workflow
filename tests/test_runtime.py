from __future__ import annotations

import json
from pathlib import Path

import pytest

from iomix_workflow.graph import compile_workflow
from iomix_workflow.locking import DirectoryLock, LockError
from iomix_workflow.runtime import ExecutionError, Runtime, runtime_paths
from iomix_workflow.yaml_loader import load_workflow


def setup(tmp_path: Path, script: str) -> tuple[Runtime, str]:
    (tmp_path / "write.py").write_text(script, encoding="utf-8")
    workflow = tmp_path / "workflow.yaml"
    workflow.write_text(
        """
schema: iomix://workflow/v1
workflow_id: local
steps:
  write:
    run: {python: write.py}
    outputs:
      result: {path: result.txt}
    resources: {timeout_seconds: 2.0}
""",
        encoding="utf-8",
    )
    compiled = compile_workflow(load_workflow(workflow))
    return Runtime(compiled, runtime_paths(workflow, tmp_path / "output", "local")), "write"


def test_local_staging_publication_cache_and_logs(tmp_path: Path) -> None:
    runtime, instance = setup(
        tmp_path,
        "import json, os\n"
        "r=json.load(open(os.environ['IOMIX_WORKFLOW_REQUEST']))\n"
        "open(r['outputs']['result'],'w').write('ok')\n"
        "print('hello')\n",
    )
    assert runtime.run_local(instance)["status"] == "completed"
    assert (tmp_path / "output" / "result.txt").read_text(encoding="utf-8") == "ok"
    assert runtime.run_local(instance)["status"] == "cached"
    state = runtime.state(instance)
    assert state and state["outputs"]["result"]["sha256"]
    assert "hello" in Path(state["log"]).read_text(encoding="utf-8")


def test_failure_never_creates_valid_cache(tmp_path: Path) -> None:
    runtime, instance = setup(tmp_path, "raise RuntimeError('broken')\n")
    with pytest.raises(ExecutionError, match="exited"):
        runtime.run_local(instance)
    assert runtime.state(instance)["status"] == "failed"  # type: ignore[index]
    assert not (tmp_path / "output" / "result.txt").exists()


def test_missing_declared_output_fails(tmp_path: Path) -> None:
    runtime, instance = setup(tmp_path, "print('no output')\n")
    with pytest.raises(ExecutionError, match="missing"):
        runtime.run_local(instance)
    assert runtime.state(instance)["status"] == "failed"  # type: ignore[index]


def test_shell_tokens_are_passed_via_environment_without_command_injection(tmp_path: Path) -> None:
    attack = "value with 'quotes' ; touch PWNED; printf '%s' \"$HOME\""
    workflow_path = tmp_path / "workflow.yaml"
    workflow_path.write_text(
        "schema: iomix://workflow/v1\nworkflow_id: shell\nparameters:\n"
        f"  attack: {json.dumps(attack)}\nsteps:\n  write:\n"
        "    foreach: {label: [\"a b; $(touch FOREACH_PWNED)\"]}\n"
        "    inputs: {literal: {literal: \"x; touch INPUT_PWNED\"}}\n"
        "    run:\n      shell: >-\n"
        "        printf '%s\\n%s\\n%s\\n%s\\n' ${parameters.attack} '${foreach.label}' "
        "\"prefix:${inputs.literal}:suffix\" ${outputs.result} > ${outputs.result}\n"
        "    outputs: {result: {path: result.txt}}\n",
        encoding="utf-8",
    )
    compiled = compile_workflow(load_workflow(workflow_path))
    runtime = Runtime(compiled, runtime_paths(workflow_path, tmp_path / "output", "shell"))
    result = runtime.run_local(compiled.order[0])
    assert result["status"] == "completed"
    lines = (tmp_path / "output" / "result.txt").read_text(encoding="utf-8").splitlines()
    assert lines[:3] == [
        attack,
        "a b; $(touch FOREACH_PWNED)",
        "prefix:x; touch INPUT_PWNED:suffix",
    ]
    assert not (tmp_path / "PWNED").exists()
    assert not (tmp_path / "FOREACH_PWNED").exists()
    assert not (tmp_path / "INPUT_PWNED").exists()


def test_argv_interpolation_remains_one_direct_argument(tmp_path: Path) -> None:
    (tmp_path / "write.py").write_text(
        "import json,os,sys\nr=json.load(open(os.environ['IOMIX_WORKFLOW_REQUEST']))\n"
        "open(r['outputs']['result'],'w').write(sys.argv[1])\n",
        encoding="utf-8",
    )
    workflow_path = tmp_path / "workflow.yaml"
    workflow_path.write_text(
        "schema: iomix://workflow/v1\nworkflow_id: argv\n"
        "parameters: {value: \"a b; touch PWNED\"}\n"
        "steps:\n  write:\n    run: {argv: [python, write.py, \"${parameters.value}\"]}\n"
        "    outputs: {result: {path: result.txt}}\n",
        encoding="utf-8",
    )
    compiled = compile_workflow(load_workflow(workflow_path))
    runtime = Runtime(compiled, runtime_paths(workflow_path, tmp_path / "output", "argv"))
    runtime.run_local("write")
    assert (tmp_path / "output" / "result.txt").read_text() == "a b; touch PWNED"
    assert not (tmp_path / "PWNED").exists()


def test_tool_identity_includes_executable_digest(tmp_path: Path) -> None:
    tool = tmp_path / "tool"
    tool.write_text("#!/bin/sh\necho v1\n", encoding="utf-8")
    tool.chmod(0o755)
    workflow_path = tmp_path / "workflow.yaml"
    workflow_path.write_text(
        f"""schema: iomix://workflow/v1
workflow_id: tool
tools:
  checked:
    executable: {tool}
    version: {{argv: [{tool}], timeout_seconds: 1.0}}
steps:
  x:
    tools: [checked]
    run: {{argv: [{tool}]}}
    outputs: {{result: {{path: result.txt}}}}
""",
        encoding="utf-8",
    )
    compiled = compile_workflow(load_workflow(workflow_path))
    runtime = Runtime(compiled, runtime_paths(workflow_path, tmp_path / "output", "tool"))
    _, details, _, _ = runtime.fingerprint(compiled.instances["x"])
    identity = details["tools"]["checked"]["executable_identity"]
    assert identity["kind"] == "file"
    assert len(identity["sha256"]) == 64


def test_fingerprint_ignores_submit_worker_placement_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, instance_id = setup(tmp_path, "pass\n")
    instance = runtime.compiled.instances[instance_id]
    first = runtime.fingerprint(instance)[0]
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("TMPDIR", "/different/worker/tmp")
    monkeypatch.setenv("XDG_CACHE_HOME", "/different/cache")
    second = runtime.fingerprint(instance)[0]
    assert first == second


def test_python_and_bash_args_interpolate_as_single_values(tmp_path: Path) -> None:
    (tmp_path / "write.py").write_text(
        "import json,os,sys\nr=json.load(open(os.environ['IOMIX_WORKFLOW_REQUEST']))\n"
        "open(r['outputs']['result'],'w').write(sys.argv[1])\n",
        encoding="utf-8",
    )
    workflow = tmp_path / "workflow.yaml"
    workflow.write_text(
        "schema: iomix://workflow/v1\nworkflow_id: args\n"
        'parameters: {value: "a b; touch PWNED"}\nsteps:\n  write:\n'
        '    run: {python: write.py, args: ["${parameters.value}"]}\n'
        "    outputs: {result: {path: result.txt}}\n",
        encoding="utf-8",
    )
    compiled = compile_workflow(load_workflow(workflow))
    runtime = Runtime(compiled, runtime_paths(workflow, tmp_path / "output", "args"))
    runtime.run_local("write")
    assert (tmp_path / "output" / "result.txt").read_text() == "a b; touch PWNED"
    assert not (tmp_path / "PWNED").exists()


def test_argv_requires_declared_matching_tool_and_detects_replacement(tmp_path: Path) -> None:
    tool = tmp_path / "tool"
    tool.write_text("#!/bin/sh\necho v1\n", encoding="utf-8")
    tool.chmod(0o755)
    workflow = tmp_path / "workflow.yaml"
    workflow.write_text(
        f"""schema: iomix://workflow/v1
workflow_id: executable
tools:
  checked:
    executable: {tool}
    version: {{argv: [{tool}]}}
steps:
  x:
    tools: [checked]
    run: {{argv: [{tool}]}}
    outputs: {{result: {{path: result.txt}}}}
""",
        encoding="utf-8",
    )
    compiled = compile_workflow(load_workflow(workflow))
    runtime = Runtime(compiled, runtime_paths(workflow, tmp_path / "output", "executable"))
    first = runtime.fingerprint(compiled.instances["x"])[0]
    tool.write_text("#!/bin/sh\necho v2\n", encoding="utf-8")
    second = runtime.fingerprint(compiled.instances["x"])[0]
    assert first != second
    undeclared = workflow.read_text().replace("    tools: [checked]\n", "")
    workflow.write_text(undeclared, encoding="utf-8")
    compiled = compile_workflow(load_workflow(workflow))
    runtime = Runtime(compiled, runtime_paths(workflow, tmp_path / "output", "executable"))
    with pytest.raises(ExecutionError, match="declared tool"):
        runtime.fingerprint(compiled.instances["x"])


def test_run_selection_rejects_omitted_dependencies(tmp_path: Path) -> None:
    workflow = tmp_path / "workflow.yaml"
    workflow.write_text(
        "schema: iomix://workflow/v1\nworkflow_id: deps\nsteps:\n"
        '  a: {run: {argv: ["true"]}, outputs: {x: {path: a}}}\n'
        '  b: {needs: [a], run: {argv: ["true"]}, outputs: {x: {path: b}}}\n',
        encoding="utf-8",
    )
    compiled = compile_workflow(load_workflow(workflow))
    runtime = Runtime(compiled, runtime_paths(workflow, tmp_path / "out", "deps"))
    with pytest.raises(ExecutionError, match="omits dependencies"):
        runtime.execute_selected(("b",))


def test_output_lock_is_global_across_workflow_ids(tmp_path: Path) -> None:
    workflow = tmp_path / "workflow.yaml"
    workflow.write_text(
        "schema: iomix://workflow/v1\nworkflow_id: one\nsteps:\n"
        '  x: {run: {argv: ["true"]}, outputs: {x: {path: shared}}}\n',
        encoding="utf-8",
    )
    compiled = compile_workflow(load_workflow(workflow))
    output = tmp_path / "out"
    first = Runtime(compiled, runtime_paths(workflow, output, "one"))
    second = Runtime(compiled, runtime_paths(workflow, output, "two"))
    first_path = {lock.path for lock in first._execution_locks(compiled.instances["x"])}
    second_path = {lock.path for lock in second._execution_locks(compiled.instances["x"])}
    global_root = output.resolve() / ".iomix" / "locks"
    common = next(path for path in first_path & second_path if path.parent == global_root)
    with DirectoryLock(common), pytest.raises(LockError, match="timed out"):
        DirectoryLock(common, acquire_timeout=0.1).acquire()
