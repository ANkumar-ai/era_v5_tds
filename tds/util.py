"""Deterministic hashing + canonical serialization helpers.

Every hash in the system flows through here so that the *same content* always
produces the *same hash*, on any machine, in any Python process. That property
is what makes the whole run auditable and replayable.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any


def canonical_json(obj: Any) -> str:
    """Serialize with sorted keys and no incidental whitespace.

    Canonical form => byte-identical output for equal objects => stable hashes.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_hex(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def hash_obj(obj: Any, prefix: str = "sha256") -> str:
    """Content hash of an arbitrary JSON-able object."""
    return f"{prefix}_{sha256_hex(canonical_json(obj))[:16]}"


def hash_tokens(tokens: list[int]) -> str:
    """Hash of a token stream (order matters)."""
    return "sha256_" + sha256_hex(",".join(str(t) for t in tokens))[:16]


def batch_hash(batch_id: str, sequences: list[dict]) -> str:
    """Stable hash over a packed batch.

    Includes batch id, token ids, span boundaries and all masks so that ANY
    change to what the model sees changes the hash. This is the object replay
    and resume are checked against.
    """
    payload = {
        "batch_id": batch_id,
        "sequences": [
            {
                "tokens": s["tokens"],
                "loss_mask": s["loss_mask"],
                "attention_segments": s["segment_ids"],
                "position_ids": s["position_ids"],
                "doc_spans": s["doc_spans"],
                "source_shard": s["source_shard"],
            }
            for s in sequences
        ],
    }
    return "sha256_" + sha256_hex(canonical_json(payload))


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def append_jsonl(path: str, obj: Any) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(canonical_json(obj) + "\n")


def read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
