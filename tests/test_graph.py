from __future__ import annotations

import pytest

from iomix_workflow.graph import GraphError, compile_workflow, resolve_step_output
from iomix_workflow.models import StepOutputInput, Workflow
from iomix_workflow.yaml_loader import load_yaml


def workflow(text: str) -> Workflow:
    from iomix_workflow.yaml_loader import _freeze_sequences

    return Workflow.model_validate(_freeze_sequences(load_yaml(text)), strict=True)


def test_cartesian_expansion_shared_axis_and_target_closure() -> None:
    compiled = compile_workflow(
        workflow(
            """
schema: iomix://workflow/v1
workflow_id: expanded
steps:
  produce:
    foreach: {sample: [a, b], lane: [1, 2]}
    run: {argv: [make, "${foreach.sample}", "${foreach.lane}"]}
    outputs: {data: {path: "raw/${foreach.sample}-${foreach.lane}.txt"}}
  consume:
    foreach: {sample: [a, b]}
    needs: [produce]
    run: {argv: [consume, "${foreach.sample}"]}
    outputs: {result: {path: "done/${foreach.sample}.txt"}}
"""
        )
    )
    assert len(compiled.instances) == 6
    consumers = [item for item in compiled.instances.values() if item.step_id == "consume"]
    assert all(len(item.dependencies) == 2 for item in consumers)
    assert len(compiled.select(targets=("consume",))) == 6
    assert len(compiled.select(steps=("consume",))) == 2


def test_instance_identity_is_independent_of_foreach_mapping_order() -> None:
    template = """
schema: iomix://workflow/v1
workflow_id: order
steps:
  x:
    foreach: {%s}
    run: {argv: [x]}
    outputs: {data: {path: "${foreach.a}-${foreach.b}"}}
"""
    first = compile_workflow(workflow(template % "a: [1], b: [2]"))
    second = compile_workflow(workflow(template % "b: [2], a: [1]"))
    assert first.order == second.order


def test_step_output_must_resolve_exactly_one() -> None:
    compiled = compile_workflow(
        workflow(
            """
schema: iomix://workflow/v1
workflow_id: ambiguity
steps:
  source:
    foreach: {lane: [1, 2]}
    run: {argv: [x]}
    outputs: {data: {path: "${foreach.lane}.txt"}}
  sink:
    run: {argv: [x]}
    inputs: {data: {step: source, output: data}}
    outputs: {done: {path: done.txt}}
"""
        )
    )
    sink = next(item for item in compiled.instances.values() if item.step_id == "sink")
    with pytest.raises(GraphError, match="exactly one"):
        resolve_step_output(compiled, sink, StepOutputInput(step="source", output="data"))


@pytest.mark.parametrize(
    "body, message",
    [
        (
            "a: {needs: [b], run: {argv: [x]}, outputs: {x: {path: a}}}\n"
            "  b: {needs: [a], run: {argv: [x]}, outputs: {x: {path: b}}}",
            "cycle",
        ),
        (
            "a: {run: {argv: [x]}, outputs: {x: {path: same}}}\n"
            "  b: {run: {argv: [x]}, outputs: {x: {path: same}}}",
            "collision",
        ),
        ("a: {run: {argv: ['${unknown.x}']}, outputs: {x: {path: x}}}", "undeclared"),
    ],
)
def test_static_graph_validation(body: str, message: str) -> None:
    text = f"schema: iomix://workflow/v1\nworkflow_id: bad\nsteps:\n  {body}\n"
    with pytest.raises(GraphError, match=message):
        compile_workflow(workflow(text))


def test_shared_axis_matching_uses_strict_json_scalar_identity() -> None:
    compiled = compile_workflow(
        workflow(
            """
schema: iomix://workflow/v1
workflow_id: strict
steps:
  source:
    foreach: {value: [1, 1.0, true]}
    run: {argv: [x]}
    outputs: {data: {path: "source/${foreach.value}"}}
  sink:
    foreach: {value: [1, 1.0, true]}
    needs: [source]
    run: {argv: [x]}
    outputs: {data: {path: "sink/${foreach.value}"}}
"""
        )
    )
    consumers = [item for item in compiled.instances.values() if item.step_id == "sink"]
    assert all(len(item.dependencies) == 1 for item in consumers)


def test_duplicate_artifact_ids_are_rejected_across_outputs() -> None:
    artifact = (
        "artifact: {artifact_id: same, role: data, media_type: text/plain, "
        "schema_id: text, schema_version: 1}"
    )
    text = (
        "schema: iomix://workflow/v1\nworkflow_id: artifacts\nsteps:\n"
        f"  a: {{run: {{argv: [x]}}, outputs: {{x: {{path: a, {artifact}}}}}}}\n"
        f"  b: {{run: {{argv: [x]}}, outputs: {{x: {{path: b, {artifact}}}}}}}\n"
    )
    with pytest.raises(GraphError, match="duplicate artifact ID"):
        compile_workflow(workflow(text))
