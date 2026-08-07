"""OPUS data selection: accept / reject / defer / protected-floor override.

From Session 5, OPUS is not a black box. You keep a ghost copy of the model,
run it against a golden proxy (held-out benchmark material, never trained), and
find which candidate batches would most move the weights that are still bad for
the benchmark. You KEEP high-signal batches and THROW AWAY the ones the model is
already comfortable with. Rohan is explicit: push for high loss / high gradient,
reject the easy wins. Scarce protected lanes (indic, agentic, reasoning) bypass
the selector via an always-on floor, because an English-heavy proxy would starve
them.

This implementation makes the "already comfortable" test REAL: it asks the live
model for its current loss on the candidate and rejects when that loss is well
below the model's running average -- i.e. the model already knows this. The
proxy-alignment term is a deterministic function of shard content (a stand-in
for gradient alignment against the golden-proxy direction), so decisions are
reproducible.
"""
from __future__ import annotations

from .util import sha256_hex

# decision constants
ACCEPT, REJECT, DEFER = "accept", "reject", "defer"


def _proxy_alignment(seq: dict, seed: int) -> float:
    """Deterministic stand-in for gradient alignment with the proxy direction.

    Real OPUS computes how much a candidate would move the benchmark-critical
    weights. Here we derive a stable pseudo-alignment in [0,1) from the shard
    content + seed. Same content -> same score, every run.
    """
    key = f"{seed}:{seq['source_shard']}:{seq['tokens']}"
    h = int(sha256_hex(key)[:8], 16)
    return (h % 10_000) / 10_000.0


class Opus:
    def __init__(self, seed: int, accept_align: float = 0.55,
                 defer_align: float = 0.35, comfortable_ratio: float = 0.6):
        self.seed = seed
        self.accept_align = accept_align
        self.defer_align = defer_align
        # reject when current loss < comfortable_ratio * running average loss
        self.comfortable_ratio = comfortable_ratio

    def decide(self, seq: dict, model, stage: dict,
               realized_shares: dict) -> dict:
        lane = seq["lane"]
        align = _proxy_alignment(seq, self.seed)
        cur_loss = model.eval_sequence(seq)["avg_loss"]
        avg = model.running_avg_loss
        floors = stage.get("protected_floors", {})
        mix = stage.get("mixture", {})

        base = {"shard": seq["source_shard"], "lane": lane,
                "align_score": round(align, 4), "cur_loss": round(cur_loss, 4),
                "model_avg_loss": round(avg, 4), "override": False}

        # 1) protected-floor override: scarce lane below its floor -> force in,
        #    regardless of what the selector thinks.
        if lane in floors and realized_shares.get(lane, 0.0) < floors[lane]:
            return {**base, "decision": ACCEPT, "override": True,
                    "reason": "protected_floor_override"}

        # 2) stage mismatch: lane not active in this curriculum stage -> defer
        if mix.get(lane, 0.0) <= 0.0:
            return {**base, "decision": DEFER, "reason": "stage_mismatch"}

        # 3) already comfortable: model's loss here is far below its average,
        #    so it has effectively learned this -> reject as wasted compute.
        if model._loss_n > 0 and cur_loss < self.comfortable_ratio * avg:
            return {**base, "decision": REJECT, "reason": "already_comfortable_low_loss"}

        # 4) proxy-alignment gate (keep high-signal, defer marginal, drop low)
        if align >= self.accept_align:
            return {**base, "decision": ACCEPT, "reason": "high_gradient_alignment"}
        if align >= self.defer_align:
            return {**base, "decision": DEFER, "reason": "below_proxy_threshold"}
        return {**base, "decision": REJECT, "reason": "low_gradient_alignment"}
