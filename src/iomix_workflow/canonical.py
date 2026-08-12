from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> bytes:
    """Return the unique UTF-8 JSON representation used for package identities."""
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()
