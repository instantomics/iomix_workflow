from __future__ import annotations

from pathlib import Path

import pytest

from iomix_workflow.cli import main
from iomix_workflow.graph import compile_workflow
from iomix_workflow.runtime import ExecutionError, Runtime, runtime_paths
from iomix_workflow.slurm import SlurmExecutor
from iomix_workflow.yaml_loader import load_workflow


def fake(path: Path, name: str, body: str) -> str:
    script = path / name
    script.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    script.chmod(0o755)
    return str(script)


def test_slurm_submit_dependency_status_and_cancel(tmp_path: Path) -> None:
    calls = tmp_path / "calls"
    counter = tmp_path / "counter"
    sbatch = fake(
        tmp_path,
        "sbatch",
        f"n=$(($(cat {counter} 2>/dev/null || echo 9)+1)); echo $n > {counter}; "
        f"echo \"$@\" >> {calls}; echo $n\n",
    )
    sacct = fake(tmp_path, "sacct", "echo 'RUNNING|'\n")
    squeue = fake(tmp_path, "squeue", "exit 0\n")
    scancel = fake(tmp_path, "scancel", "exit 0\n")
    workflow_path = tmp_path / "workflow.yaml"
    workflow_path.write_text(
        f"""
schema: iomix://workflow/v1
workflow_id: slurm
slurm:
  bootstrap_argv: [env]
  sbatch: {sbatch}
  sacct: {sacct}
  squeue: {squeue}
  scancel: {scancel}
steps:
  a: {{run: {{shell: "true"}}, outputs: {{x: {{path: a}}}}}}
  b:
    needs: [a]
    inputs: {{source: {{step: a, output: x}}}}
    run: {{shell: "true"}}
    outputs: {{x: {{path: b}}}}
""",
        encoding="utf-8",
    )
    receipt = tmp_path / "receipt with spaces.json"
    receipt.write_text("{}", encoding="utf-8")
    os_environ = __import__("os").environ
    old_receipt = os_environ.get("IOM_ACTIVE_SHELL_RECEIPT")
    os_environ["IOM_ACTIVE_SHELL_RECEIPT"] = str(receipt)
    compiled = compile_workflow(load_workflow(workflow_path))
    runtime = Runtime(compiled, runtime_paths(workflow_path, tmp_path / "out", "slurm"))
    executor = SlurmExecutor(runtime)
    try:
        result = executor.submit(compiled.order)
    finally:
        if old_receipt is None:
            os_environ.pop("IOM_ACTIVE_SHELL_RECEIPT", None)
        else:
            os_environ["IOM_ACTIVE_SHELL_RECEIPT"] = old_receipt
    assert [item["status"] for item in result] == ["submitted", "submitted"]
    lines = calls.read_text(encoding="utf-8").splitlines()
    assert "--dependency=afterok:10" in lines[1]
    # The exact upstream fingerprint is reused only after checking SLURM. The
    # unresolved downstream fingerprint is deliberately submitted again.
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("IOM_ACTIVE_SHELL_RECEIPT", str(receipt))
    try:
        repeated = executor.submit(compiled.order)
    finally:
        monkeypatch.undo()
    assert repeated[0] == {"instance_id": "a", "status": "running", "job_id": "10"}
    assert repeated[1]["status"] == "submitted"
    assert len(calls.read_text(encoding="utf-8").splitlines()) == 3
    assert executor.refresh(("a",))[0]["status"] == "running"
    assert executor.cancel(("a",))[0]["status"] == "cancelled"

    script = next((tmp_path / "out" / ".iomix" / "workflow" / "slurm" / "slurm").glob("*.sh"))
    body = script.read_text(encoding="utf-8")
    assert "IOM_ACTIVE_SHELL_RECEIPT=" in body
    assert str(receipt) in body
    assert __import__("sys").executable not in body


def test_slurm_uses_iom_receipt_bootstrap_by_default(tmp_path: Path, monkeypatch) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("IOM_ACTIVE_SHELL_RECEIPT", str(receipt))
    workflow = tmp_path / "workflow.yaml"
    workflow.write_text(
        "schema: iomix://workflow/v1\nworkflow_id: closed\nsteps:\n"
        '  a: {run: {argv: ["true"]}, outputs: {x: {path: x}}}\n',
        encoding="utf-8",
    )
    compiled = compile_workflow(load_workflow(workflow))
    runtime = Runtime(compiled, runtime_paths(workflow, tmp_path / "out", "closed"))
    script = SlurmExecutor(runtime)._script(compiled.instances["a"])
    assert "iom env exec-receipt" in script.read_text(encoding="utf-8")


def test_slurm_fails_closed_with_explicit_empty_bootstrap(tmp_path: Path, monkeypatch) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("IOM_ACTIVE_SHELL_RECEIPT", str(receipt))
    workflow = tmp_path / "workflow.yaml"
    workflow.write_text(
        "schema: iomix://workflow/v1\nworkflow_id: closed\n"
        "slurm: {bootstrap_argv: []}\nsteps:\n"
        '  a: {run: {argv: ["true"]}, outputs: {x: {path: x}}}\n',
        encoding="utf-8",
    )
    compiled = compile_workflow(load_workflow(workflow))
    runtime = Runtime(compiled, runtime_paths(workflow, tmp_path / "out", "closed"))
    with pytest.raises(ExecutionError, match="bootstrap_argv"):
        SlurmExecutor(runtime)._script(compiled.instances["a"])


def test_cli_check_and_plan_json(tmp_path: Path) -> None:
    workflow = tmp_path / "workflow.yaml"
    workflow.write_text(
        "schema: iomix://workflow/v1\nworkflow_id: cli\nsteps:\n"
        '  a: {run: {argv: ["true"]}, outputs: {x: {path: x}}}\n',
        encoding="utf-8",
    )
    common = ["--workflow", str(workflow), "--output-root", str(tmp_path / "out"), "--json"]
    assert main(["check", *common]) == 0
    assert main(["plan", *common]) == 0


def test_slurm_script_propagates_no_cache(tmp_path: Path, monkeypatch) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("IOM_ACTIVE_SHELL_RECEIPT", str(receipt))
    workflow = tmp_path / "workflow.yaml"
    workflow.write_text(
        "schema: iomix://workflow/v1\nworkflow_id: nocache\n"
        "slurm: {bootstrap_argv: [env]}\nsteps:\n"
        '  a: {run: {argv: ["true"]}, outputs: {x: {path: x}}}\n',
        encoding="utf-8",
    )
    compiled = compile_workflow(load_workflow(workflow))
    runtime = Runtime(compiled, runtime_paths(workflow, tmp_path / "out", "nocache"))
    script = SlurmExecutor(runtime)._script(compiled.instances["a"], no_cache=True)
    assert "--no-cache" in script.read_text(encoding="utf-8")


@pytest.mark.parametrize("scheduler", ("sacct", "squeue"))
def test_scheduler_query_failures_are_explicit(
    tmp_path: Path, monkeypatch, scheduler: str
) -> None:
    sacct_body = "exit 1\n" if scheduler == "sacct" else "exit 0\n"
    squeue_body = "exit 1\n" if scheduler == "squeue" else "exit 0\n"
    workflow = tmp_path / "workflow.yaml"
    workflow.write_text(
        "schema: iomix://workflow/v1\nworkflow_id: scheduler\nslurm:\n"
        f"  sacct: {fake(tmp_path, 'sacct', sacct_body)}\n"
        f"  squeue: {fake(tmp_path, 'squeue', squeue_body)}\nsteps:\n"
        '  a: {run: {argv: ["true"]}, outputs: {x: {path: x}}}\n',
        encoding="utf-8",
    )
    compiled = compile_workflow(load_workflow(workflow))
    runtime = Runtime(compiled, runtime_paths(workflow, tmp_path / "out", "scheduler"))
    runtime._write_state("a", status="submitted", slurm_job_id="1")
    with pytest.raises(ExecutionError, match=scheduler):
        SlurmExecutor(runtime).refresh(("a",))


def test_submit_fails_closed_for_undeclared_argv_tool(tmp_path: Path) -> None:
    workflow = tmp_path / "workflow.yaml"
    workflow.write_text(
        "schema: iomix://workflow/v1\nworkflow_id: undeclared\nsteps:\n"
        '  a: {run: {argv: ["true"]}, outputs: {x: {path: x}}}\n',
        encoding="utf-8",
    )
    compiled = compile_workflow(load_workflow(workflow))
    runtime = Runtime(compiled, runtime_paths(workflow, tmp_path / "out", "undeclared"))
    with pytest.raises(ExecutionError, match="declared tool"):
        SlurmExecutor(runtime).submit(("a",))
