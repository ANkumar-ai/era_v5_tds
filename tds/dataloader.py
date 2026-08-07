"""Deterministic dataloader / mixture scheduler.

This is the program Rohan says is "freely provided by Python" but which we must
make bullet-proof. The key correctness property: the stream of batches is a PURE
FUNCTION of (seed, branch, step, model-state, cursors). Nothing depends on wall
clock or Python's RNG internal state -- because Rohan warns that a Python seed is
only reproducible on the same machine/session ("turn the machine off and on,
you'll get a different value"). We derive every random choice from a SHA-256 of
(seed, branch, step, slot) instead, so resume and replay are identical anywhere.

Per step the loader:
  1. picks a capability lane by the stage's mixture weights (hash-deterministic),
  2. pulls the next packed sequence from that lane's queue (round-robin cursor;
     repetition is allowed, as in real runs),
  3. asks OPUS to accept / reject / defer (protected floors can override),
  4. assembles the accepted sequences into a global batch.

Rejected/deferred candidates still advance the cursor (they are "seen and thrown
away"), and every decision is returned so the ledger can record the full audit
trail.
"""
from __future__ import annotations

from .util import sha256_hex
from .packing import pack_shard
from .opus import Opus, ACCEPT, REJECT, DEFER


def _hash_float(*parts) -> float:
    key = ":".join(str(p) for p in parts)
    return (int(sha256_hex(key)[:12], 16) % 1_000_000) / 1_000_000.0


class DataLoader:
    def __init__(self, shards, schedule: dict, opus: Opus, seed: int):
        self.seed = seed
        self.opus = opus
        self.lanes = sorted({s.lane for s in shards})
        # pack every shard at each sequence length used by the schedule
        seq_lens = sorted({st["sequence_length"] for st in schedule["stages"]})
        self.queues: dict[tuple[str, int], list[dict]] = {}
        for sl in seq_lens:
            for lane in self.lanes:
                q = []
                for shard in shards:
                    if shard.lane == lane:
                        q.extend(pack_shard(shard, sl))
                if q:
                    self.queues[(lane, sl)] = q
        self.cursors: dict[str, int] = {}

    # ---- state (for checkpoint) --------------------------------------
    def cursor_state(self) -> dict:
        return dict(self.cursors)

    def load_cursor_state(self, state: dict) -> None:
        self.cursors = dict(state)

    # ---- packing utilisation stats (static, from the queues) ---------
    def packing_stats(self) -> dict:
        real = pad = 0
        for q in self.queues.values():
            for s in q:
                real += s["n_tokens_real"]
                pad += s["n_pad"]
        tot = real + pad
        return {"real_tokens": real, "pad_tokens": pad,
                "packing_utilisation": (real / tot) if tot else 0.0}

    # ---- one global batch --------------------------------------------
    def batch_for_step(self, step: int, branch_id: str, stage: dict, model,
                       realized_tokens: dict, batch_seqs: int,
                       max_tries: int = 8) -> tuple[list[dict], list[dict], dict]:
        seq_len = stage["sequence_length"]
        mixture = stage["mixture"]
        accepted: list[dict] = []
        opus_records: list[dict] = []
        slot = tries = 0
        cap = batch_seqs * max_tries

        def shares():
            tot = sum(realized_tokens.values()) or 1
            return {l: realized_tokens.get(l, 0) / tot for l in self.lanes}

        while len(accepted) < batch_seqs and tries < cap:
            tries += 1
            lane = self._pick_lane(branch_id, step, slot, mixture)
            slot += 1
            q = self.queues.get((lane, seq_len))
            if not q:
                continue
            key = f"{lane}@{seq_len}"
            idx = self.cursors.get(key, 0)
            cand = q[idx % len(q)]
            self.cursors[key] = idx + 1

            d = self.opus.decide(cand, model, stage, shares())
            d = {**d, "step": step, "slot": slot - 1, "branch": branch_id,
                 "n_cand_tokens": cand["n_tokens_real"], "n_cand_loss": cand["n_loss"]}
            opus_records.append(d)
            if d["decision"] == ACCEPT:
                accepted.append(cand)
                realized_tokens[lane] = realized_tokens.get(lane, 0) + cand["n_loss"]

        info = {
            "requested": batch_seqs,
            "accepted": len(accepted),
            "candidates_examined": len(opus_records),
            "underfilled": len(accepted) < batch_seqs,
        }
        return accepted, opus_records, info

    def _pick_lane(self, branch_id, step, slot, mixture: dict) -> str:
        r = _hash_float(self.seed, branch_id, step, slot)
        acc = 0.0
        items = sorted(mixture.items())
        for lane, w in items:
            acc += w
            if r <= acc:
                return lane
        return items[-1][0]
