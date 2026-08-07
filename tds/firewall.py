"""Evaluation / validation firewall (the 'never-train' bouncer).

Test material must never enter a loss-bearing batch. The firewall works at two
layers:

  1. Manifest layer -- a shard is refused admission if its manifest says
     never_train, eval_overlap_status == 'test_material', or contains_canary.
  2. Batch layer (defense in depth) -- every packed sequence about to train is
     scanned for the canary token subsequence, so contamination cannot slip
     through even if a manifest were wrong.

If a benchmark score ever jumps suspiciously, the run can be audited against
these fingerprints.
"""
from __future__ import annotations

from .corpus import CANARY
from .shards import Shard
from .tokenizer import Tokenizer


class Firewall:
    def __init__(self, tok: Tokenizer):
        # tokenized canary fingerprint used for batch-level scanning
        self.canary_ids = tok.encode(CANARY, add_eos=False)

    def admit_shards(self, shards: list[Shard], logger=None) -> tuple[list[Shard], list[dict]]:
        admitted, blocked = [], []
        for s in shards:
            m = s.manifest
            reasons = []
            if m["never_train"]:
                reasons.append("never_train=true")
            if m["eval_overlap_status"] == "test_material":
                reasons.append("eval_overlap=test_material")
            if m["contains_canary"]:
                reasons.append("canary_present")
            if reasons:
                rec = {"shard_id": s.shard_id, "reasons": reasons,
                       "content_hash": m["content_hash"]}
                blocked.append(rec)
                if logger:
                    logger.check(True, "eval_shard_blocked",
                                 f"{s.shard_id} ({', '.join(reasons)})")
            else:
                admitted.append(s)
        return admitted, blocked

    def _contains_subseq(self, tokens: list[int]) -> bool:
        n, m = len(tokens), len(self.canary_ids)
        if m == 0:
            return False
        for i in range(n - m + 1):
            if tokens[i:i + m] == self.canary_ids:
                return True
        return False

    def scan_batch(self, sequences: list[dict]) -> bool:
        """Return True if the batch is CLEAN (no canary). Loss-bearing tokens
        only -- a canary token with loss_mask 0 still counts as contamination,
        so we scan all real tokens."""
        for seq in sequences:
            real = [t for t, s in zip(seq["tokens"], seq["segment_ids"]) if s != -1]
            if self._contains_subseq(real):
                return False
        return True
