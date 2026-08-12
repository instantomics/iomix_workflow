from __future__ import annotations

import pytest
from pydantic import ValidationError

from iomix_workflow.models import Workflow
from iomix_workflow.yaml_loader import YamlError, _freeze_sequences, load_yaml

BASE = """\
schema: iomix://workflow/v1
workflow_id: example
steps:
  make:
    run:
      argv: [tool, x]
    outputs:
      result:
        path: result.txt
"""


def test_workflow_is_strict_frozen_and_has_schema() -> None:
    workflow = Workflow.model_validate(_freeze_sequences(load_yaml(BASE)), strict=True)
    assert workflow.schema_ == "iomix://workflow/v1"
    assert Workflow.model_json_schema()["additionalProperties"] is False
    with pytest.raises(ValidationError):
        workflow.workflow_id = "changed"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        Workflow.model_validate(
            _freeze_sequences(
                load_yaml(BASE.replace("outputs:", 'resources: {cpus: "1"}\n    outputs:'))
            ),
            strict=True,
        )


def test_yaml_sequences_validate_as_frozen_tuples() -> None:
    workflow = Workflow.model_validate(
        _freeze_sequences(
            load_yaml(
                BASE.replace(
                    "run:\n      argv: [tool, x]", "needs: []\n    run:\n      argv: [tool, x]"
                )
            )
        ),
        strict=True,
    )
    assert workflow.steps["make"].run.argv == ("tool", "x")


@pytest.mark.parametrize(
    "document",
    [
        "---\na: 1\n---\nb: 2\n",
        "a: &x 1\nb: *x\n",
        "a: !!str x\n",
        "%YAML 1.2\n---\na: 1\n",
        "a: 1\na: 2\n",
        "1: value\n",
        "a: .nan\n",
        "a: 0x10\n",
        "a: yes\n",
        "a: ~\n",
        "base: {x: 1}\nother: {<<: {x: 1}}\n",
    ],
)
def test_conservative_yaml_rejects_unsafe_or_non_json_forms(document: str) -> None:
    with pytest.raises(YamlError):
        load_yaml(document)


def test_yaml_limits_bytes_and_depth() -> None:
    with pytest.raises(YamlError, match="bytes"):
        load_yaml(b"x" * (2 * 1024 * 1024 + 1))
    nested = "value: " + "[" * 34 + "0" + "]" * 34
    with pytest.raises(YamlError, match="depth"):
        load_yaml(nested)


@pytest.mark.parametrize("values", ("[1, 1]", "[1.0, 1.0]", "[true, true]", "[null, null]"))
def test_foreach_rejects_duplicate_canonical_values(values: str) -> None:
    document = BASE.replace(
        "run:\n      argv: [tool, x]",
        f"foreach: {{value: {values}}}\n    run:\n      argv: [tool, x]",
    )
    with pytest.raises(ValidationError, match="duplicate canonical"):
        Workflow.model_validate(_freeze_sequences(load_yaml(document)), strict=True)


def test_foreach_json_scalar_types_have_distinct_identity() -> None:
    document = BASE.replace(
        "run:\n      argv: [tool, x]",
        "foreach: {value: [1, 1.0, true]}\n    run:\n      argv: [tool, x]",
    )
    workflow = Workflow.model_validate(_freeze_sequences(load_yaml(document)), strict=True)
    assert workflow.steps["make"].foreach["value"] == (1, 1.0, True)


@pytest.mark.parametrize("path", (".", ".iomix", ".iomix/state", "x/.iomix/y"))
def test_outputs_reject_root_and_internal_namespace(path: str) -> None:
    document = BASE.replace("path: result.txt", f"path: {path}")
    with pytest.raises(ValidationError, match="normalized"):
        Workflow.model_validate(_freeze_sequences(load_yaml(document)), strict=True)


@pytest.mark.parametrize("value", ("null", '""'))
def test_interpolated_outputs_reject_empty_foreach_values(value: str) -> None:
    document = BASE.replace(
        "run:\n      argv: [tool, x]",
        f"foreach: {{value: [{value}]}}\n    run:\n      argv: [tool, x]",
    ).replace("path: result.txt", 'path: "${foreach.value}"')
    workflow = Workflow.model_validate(_freeze_sequences(load_yaml(document)), strict=True)
    from iomix_workflow.graph import GraphError, compile_workflow

    with pytest.raises(GraphError, match="normalized"):
        compile_workflow(workflow)
