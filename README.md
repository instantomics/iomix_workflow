# iomix_workflow

`iomix_workflow` is a small first-party execution engine for reproducible source
preparation and task-owned data operations. It deliberately supports a static,
inspectable workflow rather than runtime workflow generation, and it has no
dependency on Iomix.

Reproducibility is based on validated canonical workflow definitions, SHA-256
input identities, repository scripts and policies, project and lock files,
Python identity, activated modules, external-tool probes, and declared execution
settings. File metadata is only a local shortcut for reusing a previously
computed digest; deep verification always reads the bytes. Successful outputs
are staged, validated, and atomically published before cache state becomes
valid. Operational state and logs explain execution and cache decisions but are
not scientific evidence.

The engine executes Python, Bash, direct argv, and explicitly requested shell
steps locally or as separate dependency-linked SLURM jobs. Source and task
integration, artifact-manifest interpretation, CAS admission, scientific
validation, and human review remain responsibilities of their owning packages.

The generated JSON Schema and command help are the authoritative references for
the workflow format and command surface.

SLURM submission is detached by definition. It requires an active Iom receipt
and uses `iom env exec-receipt` to materialize the exact environment on the
allocation before executing a worker. Sites may replace `slurm.bootstrap_argv`
with another finite receipt-aware bootstrap.
