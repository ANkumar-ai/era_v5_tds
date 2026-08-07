"""The two-way ledger -- the crown jewel of Session 6.

A checkpoint without a data position is incomplete. The ledger is the accounting
book that binds model state to data state, in both directions:

  Consumption ledger (incoming): what the model SAW.
      batch id, step, branch, phase, and for every packed sequence its source
      shard, token span and a content hash. This is what audit/replay reconstruct
      against.

  Learning ledger (outgoing): what the model LEARNED.
      per-batch and per-shard average loss, perplexity, loss delta, a gradient
      proxy, the OPUS score and the model phase. This is the data Rohan says no
      lab shares -- it can only be generated while the model trains, and it is
      what lets a future run drop an already-learned shard.

Both are append-only JSONL so an offset (a line count) is a precise, restorable
position -- the ledger_offset stored in every checkpoint.
"""
from __future__ import annotations

import os

from .util import append_jsonl, read_jsonl, canonical_json, sha256_hex


class Ledger:
    def __init__(self, consumption_path: str, learning_path: str, opus_path: str):
        self.consumption_path = consumption_path
        self.learning_path = learning_path
        self.opus_path = opus_path
        for p in (consumption_path, learning_path, opus_path):
            open(p, "w", encoding="utf-8").close()   # fresh run

    # ---- append ------------------------------------------------------
    def record_consumption(self, rec: dict) -> None:
        append_jsonl(self.consumption_path, rec)

    def record_learning(self, rec: dict) -> None:
        append_jsonl(self.learning_path, rec)

    def record_opus(self, rec: dict) -> None:
        append_jsonl(self.opus_path, rec)

    # ---- offsets (restorable positions) ------------------------------
    def offsets(self) -> dict:
        return {
            "consumption": _line_count(self.consumption_path),
            "learning": _line_count(self.learning_path),
            "opus": _line_count(self.opus_path),
        }

    def truncate_to(self, offsets: dict) -> None:
        """Roll every ledger back to a checkpoint's offsets.

        This is what makes crash recovery exact: on resume we discard any rows
        written after the last checkpoint, so the ledger and the model state
        agree on the same data position -- no skipped or repeated batch.
        """
        _truncate(self.consumption_path, offsets["consumption"])
        _truncate(self.learning_path, offsets["learning"])
        _truncate(self.opus_path, offsets["opus"])

    # ---- read back ---------------------------------------------------
    def consumption(self) -> list[dict]:
        return read_jsonl(self.consumption_path)

    def learning(self) -> list[dict]:
        return read_jsonl(self.learning_path)

    def opus(self) -> list[dict]:
        return read_jsonl(self.opus_path)


def _line_count(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def _truncate(path: str, n: int) -> None:
    lines = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                lines.append(line)
    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(lines[:n])
