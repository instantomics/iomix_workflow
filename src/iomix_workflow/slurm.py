from __future__ import annotations

import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

from .canonical import canonical_digest
from .graph import StepInstance, interpolate
from .locking import DirectoryLock
from .runtime import ExecutionError, FingerprintDeferred, Runtime

_JOB_ID = re.compile(r"^[0-9]+(?:_[0-9]+)?$")


class SlurmExecutor:
    def __init__(self, runtime: Runtime) -> None:
        self.runtime = runtime
        self.config = runtime.compiled.workflow.slurm

    def _command(self, argv: list[str]) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                argv,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=self.config.command_timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ExecutionError(f"scheduler command failed: {error}") from error

    def _script(self, instance: StepInstance, *, no_cache: bool = False) -> Path:
        if not self.config.bootstrap_argv:
            raise ExecutionError("SLURM requires slurm.bootstrap_argv")
        for value in self.config.bootstrap_argv:
            tokens = set(re.findall(r"\$\{([^{}]+)\}", value))
            if tokens - {"receipt"}:
                raise ExecutionError(
                    "SLURM bootstrap_argv uses unsupported placeholders: "
                    f"{sorted(tokens - {'receipt'})}"
                )
        receipt_value = os.environ.get("IOM_ACTIVE_SHELL_RECEIPT")
        if not receipt_value:
            raise ExecutionError("SLURM requires IOM_ACTIVE_SHELL_RECEIPT")
        receipt = Path(receipt_value).expanduser().resolve()
        if receipt.is_symlink() or not receipt.is_file():
            raise ExecutionError(f"active Iom receipt is not a regular file: {receipt}")
        root = self.runtime.paths.state.parent / "slurm"
        root.mkdir(parents=True, exist_ok=True)
        script = root / f"{canonical_digest(instance.instance_id)}.sh"
        worker = [
            "python",
            "-m",
            "iomix_workflow",
            "_execute",
            "--workflow",
            str(self.runtime.paths.workflow_file),
            "--output-root",
            str(self.runtime.paths.output),
            "--instance",
            instance.instance_id,
        ]
        if self.runtime.deep:
            worker.append("--deep")
        if no_cache:
            worker.append("--no-cache")
        bootstrap = [
            interpolate(value, {"receipt": receipt}) for value in self.config.bootstrap_argv
        ]
        argv = [*bootstrap, *worker]
        body = (
            "#!/bin/bash\nset -euo pipefail\n"
            f"export IOM_ACTIVE_SHELL_RECEIPT={shlex.quote(str(receipt))}\n"
            "exec " + shlex.join(argv) + "\n"
        )
        temporary = script.with_name(f".{script.name}.{os.getpid()}.tmp")
        temporary.write_text(body, encoding="utf-8")
        os.chmod(temporary, 0o700)
        os.replace(temporary, script)
        return script

    def _scheduler_state(self, job_id: str) -> tuple[str | None, str | None]:
        accounting = self._command(
            [self.config.sacct, "-n", "-X", "-j", job_id, "--format=State", "--parsable2"]
        )
        if accounting.returncode:
            detail = accounting.stderr.strip() or accounting.stdout.strip()
            raise ExecutionError(f"sacct failed for job {job_id}: {detail}")
        states = [
            line.strip().split("|", 1)[0]
            for line in accounting.stdout.splitlines()
            if line.strip()
        ]
        if not states:
            queue = self._command([self.config.squeue, "-h", "-j", job_id, "-o", "%T"])
            if queue.returncode:
                detail = queue.stderr.strip() or queue.stdout.strip()
                raise ExecutionError(f"squeue failed for job {job_id}: {detail}")
            states = [line.strip() for line in queue.stdout.splitlines() if line.strip()]
        raw = states[0] if states else None
        return (self._normalize(raw), raw) if raw else (None, None)

    def submit(self, selected: tuple[str, ...], *, no_cache: bool = False) -> list[dict[str, Any]]:
        submitted: dict[str, str] = {}
        result = []
        for instance_id in selected:
            instance = self.runtime.compiled.instances[instance_id]
            submission_identity = self.runtime.submission_identity(instance)
            try:
                fingerprint, _, _, _ = self.runtime.fingerprint(instance)
                hit, explanation = self.runtime.cache_status(instance, fingerprint)
            except FingerprintDeferred as error:
                # A downstream job may be submitted before dependency outputs
                # exist. Its worker computes the exact fingerprint after the
                # static scheduler dependencies succeed.
                fingerprint = None
                hit, explanation = False, f"deferred until dependencies complete: {error}"
            if hit and not no_cache:
                result.append(
                    {
                        "instance_id": instance_id,
                        "status": "cached",
                        "explanation": explanation,
                    }
                )
                continue
            submit_lock = self.runtime.paths.locks / f"{canonical_digest(instance_id)}.submit"
            with DirectoryLock(submit_lock):
                previous = self.runtime.state(instance_id) or {}
                previous_job = previous.get("slurm_job_id")
                reusable = (
                    not no_cache
                    and fingerprint is not None
                    and previous.get("status") in {"submitted", "queued", "running"}
                    and isinstance(previous_job, str)
                    and _JOB_ID.fullmatch(previous_job) is not None
                    and previous.get("fingerprint") == fingerprint
                    and previous.get("submission_identity") == submission_identity
                )
                if reusable:
                    scheduler_status, _ = self._scheduler_state(previous_job)
                    if scheduler_status in {"queued", "running"}:
                        submitted[instance_id] = previous_job
                        result.append(
                            {
                                "instance_id": instance_id,
                                "status": scheduler_status,
                                "job_id": previous_job,
                            }
                        )
                        continue
                resources = instance.step.resources
                argv = [
                    self.config.sbatch,
                    "--parsable",
                    f"--cpus-per-task={resources.cpus}",
                    f"--mem={resources.memory_mb}M",
                    f"--time={resources.time_minutes}",
                    f"--job-name=iw-{instance.step_id[:64]}",
                    "--output="
                    + str(
                        self.runtime.paths.logs
                        / (canonical_digest(instance_id) + ".slurm.log")
                    ),
                ]
                if resources.gpus:
                    argv.append(f"--gpus={resources.gpus}")
                if self.config.account:
                    argv.append(f"--account={self.config.account}")
                if self.config.partition:
                    argv.append(f"--partition={self.config.partition}")
                dependencies = [
                    submitted[item]
                    for item in instance.dependencies
                    if item in submitted
                ]
                if dependencies:
                    argv.append("--dependency=afterok:" + ":".join(dependencies))
                argv.extend(self.config.extra_args)
                argv.append(str(self._script(instance, no_cache=no_cache)))
                response = self._command(argv)
                if response.returncode:
                    raise ExecutionError(
                        f"sbatch failed for {instance_id}: {response.stderr.strip()}"
                    )
                job_id = response.stdout.strip().split(";", 1)[0]
                if not _JOB_ID.fullmatch(job_id):
                    raise ExecutionError(
                        f"sbatch returned invalid job ID: {response.stdout.strip()!r}"
                    )
                submitted[instance_id] = job_id
                written = self.runtime.write_submission_state(
                    instance_id,
                    job_id,
                    expected_updated_at=previous.get("updated_at"),
                    **({"fingerprint": fingerprint} if fingerprint is not None else {}),
                    submission_identity=submission_identity,
                    submitted_at=time.time(),
                    log=str(
                        self.runtime.paths.logs
                        / (canonical_digest(instance_id) + ".slurm.log")
                    ),
                    cache_explanation="submitted because "
                    + ("cache disabled" if no_cache else explanation),
                )
                if written.get("slurm_job_id") != job_id:
                    cancellation = self._command([self.config.scancel, job_id])
                    if cancellation.returncode:
                        raise ExecutionError(
                            f"scancel failed for raced job {job_id}: "
                            f"{cancellation.stderr.strip()}"
                        )
                    result.append(
                        {
                            "instance_id": instance_id,
                            "status": written.get("status", "unknown"),
                            "explanation": "worker completed while submission was in progress",
                        }
                    )
                    continue
                result.append(
                    {"instance_id": instance_id, "status": "submitted", "job_id": job_id}
                )
        return result

    @staticmethod
    def _normalize(state: str) -> str:
        state = state.upper().split("+", 1)[0]
        if state in {"PENDING", "CONFIGURING", "REQUEUED", "RESIZING"}:
            return "queued"
        if state in {"RUNNING", "COMPLETING", "SUSPENDED"}:
            return "running"
        if state == "COMPLETED":
            return "completed"
        if state in {"CANCELLED", "PREEMPTED"}:
            return "cancelled"
        return "failed"

    def refresh(self, instance_ids: tuple[str, ...]) -> list[dict[str, Any]]:
        records = []
        for instance_id in instance_ids:
            record = self.runtime.state(instance_id) or {}
            job_id = record.get("slurm_job_id")
            if not isinstance(job_id, str) or not _JOB_ID.fullmatch(job_id):
                records.append(
                    {
                        "instance_id": instance_id,
                        "status": record.get("status", "not-run"),
                    }
                )
                continue
            scheduler_status, raw = self._scheduler_state(job_id)
            status = scheduler_status or record.get("status", "unknown")
            # A worker terminal state is authoritative and must not be downgraded.
            if record.get("status") in {"completed", "failed"}:
                status = record["status"]
            elif status != record.get("status"):
                self.runtime._write_state(instance_id, status=status, scheduler_state=raw)
            records.append({"instance_id": instance_id, "status": status, "job_id": job_id})
        return records

    def cancel(self, instance_ids: tuple[str, ...]) -> list[dict[str, Any]]:
        jobs = []
        owners = []
        for instance_id in instance_ids:
            record = self.runtime.state(instance_id) or {}
            job_id = record.get("slurm_job_id")
            if isinstance(job_id, str) and _JOB_ID.fullmatch(job_id) and record.get("status") in {
                "submitted",
                "queued",
                "running",
            }:
                jobs.append(job_id)
                owners.append(instance_id)
        if jobs:
            response = self._command([self.config.scancel, *jobs])
            if response.returncode:
                raise ExecutionError(f"scancel failed: {response.stderr.strip()}")
        result = []
        for instance_id, job_id in zip(owners, jobs, strict=True):
            self.runtime._write_state(instance_id, status="cancelled", cancelled_at=time.time())
            result.append({"instance_id": instance_id, "status": "cancelled", "job_id": job_id})
        return result
