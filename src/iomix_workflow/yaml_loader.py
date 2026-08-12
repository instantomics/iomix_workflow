from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode
from yaml.tokens import AliasToken, AnchorToken, DirectiveToken, TagToken

from .models import Workflow

MAX_BYTES = 2 * 1024 * 1024
MAX_DEPTH = 32
MAX_NODES = 100_000
_SCALAR_TAGS = {
    "tag:yaml.org,2002:null",
    "tag:yaml.org,2002:bool",
    "tag:yaml.org,2002:int",
    "tag:yaml.org,2002:float",
    "tag:yaml.org,2002:str",
}
_JSON_INTEGER = re.compile(r"^-?(?:0|[1-9][0-9]*)$")
_JSON_FLOAT = re.compile(r"^-?(?:0|[1-9][0-9]*)\.[0-9]+(?:[eE][+-]?[0-9]+)?$")


class YamlError(ValueError):
    pass


def _freeze_sequences(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_freeze_sequences(item) for item in value)
    if isinstance(value, dict):
        return {key: _freeze_sequences(item) for key, item in value.items()}
    return value


def _construct(node: Node, depth: int, counter: list[int]) -> Any:
    counter[0] += 1
    if counter[0] > MAX_NODES:
        raise YamlError(f"YAML exceeds {MAX_NODES} nodes")
    if depth > MAX_DEPTH:
        raise YamlError(f"YAML exceeds depth {MAX_DEPTH}")
    if isinstance(node, ScalarNode):
        if node.tag not in _SCALAR_TAGS:
            raise YamlError(f"YAML scalar tag is not permitted: {node.tag}")
        if node.tag.endswith(":str"):
            return node.value
        if node.tag.endswith(":null"):
            if node.value.lower() != "null":
                raise YamlError("only the JSON spelling 'null' is permitted")
            return None
        if node.tag.endswith(":bool"):
            if node.value.lower() not in {"true", "false"}:
                raise YamlError("only JSON boolean spellings are permitted")
            return node.value.lower() == "true"
        if node.tag.endswith(":int"):
            if not _JSON_INTEGER.fullmatch(node.value):
                raise YamlError("only JSON decimal integers are permitted")
            return int(node.value, 10)
        if node.tag.endswith(":float"):
            if not _JSON_FLOAT.fullmatch(node.value):
                raise YamlError("only JSON decimal floats are permitted")
            value = float(node.value)
            if not math.isfinite(value):
                raise YamlError("non-finite YAML floats are not permitted")
            return value
    if isinstance(node, SequenceNode):
        if node.tag != "tag:yaml.org,2002:seq":
            raise YamlError(f"YAML sequence tag is not permitted: {node.tag}")
        return [_construct(child, depth + 1, counter) for child in node.value]
    if isinstance(node, MappingNode):
        if node.tag != "tag:yaml.org,2002:map":
            raise YamlError(f"YAML mapping tag is not permitted: {node.tag}")
        result: dict[str, Any] = {}
        for key_node, value_node in node.value:
            if not isinstance(key_node, ScalarNode) or key_node.tag != "tag:yaml.org,2002:str":
                raise YamlError("YAML mapping keys must be strings")
            key = key_node.value
            if key == "<<":
                raise YamlError("YAML merge keys are not permitted")
            if key in result:
                raise YamlError(f"duplicate YAML mapping key: {key!r}")
            result[key] = _construct(value_node, depth + 1, counter)
        return result
    raise YamlError(f"unsupported YAML node: {type(node).__name__}")


def load_yaml(data: bytes | str, *, source: str = "<input>") -> Any:
    raw = data.encode("utf-8") if isinstance(data, str) else data
    if len(raw) > MAX_BYTES:
        raise YamlError(f"{source}: YAML exceeds {MAX_BYTES} bytes")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise YamlError(f"{source}: YAML must be UTF-8") from error
    try:
        for token in yaml.scan(text):
            if isinstance(token, (DirectiveToken, AnchorToken, AliasToken, TagToken)):
                raise YamlError(
                    f"{source}: YAML directives, tags, anchors, and aliases are forbidden"
                )
        documents = list(yaml.compose_all(text, Loader=yaml.SafeLoader))
    except yaml.YAMLError as error:
        raise YamlError(f"{source}: invalid YAML: {error}") from error
    if len(documents) != 1 or documents[0] is None:
        raise YamlError(f"{source}: exactly one non-empty YAML document is required")
    return _construct(documents[0], 0, [0])


def load_workflow(path: str | Path) -> Workflow:
    workflow_path = Path(path)
    try:
        value = load_yaml(workflow_path.read_bytes(), source=str(workflow_path))
        return Workflow.model_validate(_freeze_sequences(value), strict=True)
    except OSError as error:
        raise YamlError(f"cannot read workflow {workflow_path}: {error}") from error
    except ValidationError as error:
        raise YamlError(f"invalid workflow {workflow_path}: {error}") from error
