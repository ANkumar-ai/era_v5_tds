"""Checkpoints bound to a data position.

"A checkpoint without a data position is incomplete." A checkpoint here saves
everything needed to resume the run *exactly*:

  - step               the optimizer step about to run next
  - branch_id          which run branch this belongs to (for forks)
  - model_state        the bigram counts + running-loss accumulators
  - cursors            per-lane dataloader position (what has been consumed)
  - realized_tokens    per-lane token tallies (to evaluate protected floors)
  - ledger_offsets     line offsets into every ledger (the data position)
  - lr                 scheduler state (learning-rate value)
  - seed               so the deterministic scheduler reproduces the stream

Because the dataloader is a pure function of (seed, step, model, cursors), these
fields are sufficient to regenerate the next batch identically.
"""
from __future__ import annotations

import os

from .util import write_json, read_json, ensure_dir


class CheckpointStore:
    def __init__(self, directory: str):
        self.dir = ensure_dir(directory)

    def path(self, step: int, branch_id: str) -> str:
        return os.path.join(self.dir, f"ckpt_{branch_id}_step{step:06d}.json")

    def save(self, *, step, branch_id, model, cursors, realized_tokens,
             ledger_offsets, lr, seed, parent=None) -> str:
        payload = {
            "step": step,
            "branch_id": branch_id,
            "parent_checkpoint": parent,
            "model_state": model.state_dict(),
            "cursors": cursors,
            "realized_tokens": realized_tokens,
            "ledger_offsets": ledger_offsets,
            "lr": lr,
            "seed": seed,
            "tokens_seen": model.tokens_seen,
        }
        p = self.path(step, branch_id)
        write_json(p, payload)
        return p

    def load(self, path: str) -> dict:
        return read_json(path)

    def list_for_branch(self, branch_id: str) -> list[tuple[int, str]]:
        out = []
        for f in os.listdir(self.dir):
            if f.startswith(f"ckpt_{branch_id}_step") and f.endswith(".json"):
                step = int(f.split("step")[1].split(".")[0])
                out.append((step, os.path.join(self.dir, f)))
        return sorted(out)

    def latest_at_or_before(self, branch_id: str, step: int) -> tuple[int, str] | None:
        cands = [(s, p) for s, p in self.list_for_branch(branch_id) if s <= step]
        return cands[-1] if cands else None
