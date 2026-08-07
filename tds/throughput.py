"""Throughput & packing efficiency -- reconstructable from the ledgers.

The metric that matters (Rohan): useful *loss-bearing* tokens per second at the
target mixture. The colour code from the Throughput Lab:

    green  = useful loss-bearing tokens actually trained
    amber  = tokens prepared but rejected/deferred by OPUS  (wasted preparation)
    red    = padding / packing waste (positions the GPU computes but learns nothing)
    gray   = GPU time lost waiting on the loader

Every number here is recomputed from the consumption + opus ledgers, so the
claims can be reconstructed by re-reading those files -- no unverifiable figures.
"""
from __future__ import annotations

from .util import write_json


def build(ledger, seconds: float, loader_fraction: float, out_path: str) -> dict:
    cons = ledger.consumption()
    opus = ledger.opus()

    green = sum(sq["n_loss"] for c in cons for sq in c["sequences"])          # loss-bearing
    accepted_real = sum(sq["n_tokens_real"] for c in cons for sq in c["sequences"])
    red_padding = sum(sq["n_pad"] for c in cons for sq in c["sequences"])     # padding waste

    amber = sum(d["n_cand_tokens"] for d in opus if d["decision"] != "accept")
    raw_examined = sum(d["n_cand_tokens"] for d in opus)

    n_accept = sum(1 for d in opus if d["decision"] == "accept")
    n_reject = sum(1 for d in opus if d["decision"] == "reject")
    n_defer = sum(1 for d in opus if d["decision"] == "defer")
    n_total = max(1, len(opus))

    packed_positions = accepted_real + red_padding
    report = {
        "wall_seconds": round(seconds, 4),
        "useful_loss_tokens_per_sec": round(green / seconds, 1) if seconds else 0,
        "raw_tokens_per_sec": round(raw_examined / seconds, 1) if seconds else 0,
        "tokens": {
            "raw_examined": raw_examined,
            "green_useful_loss_bearing": green,
            "accepted_real": accepted_real,
            "amber_opus_rejected_deferred": amber,
            "red_padding_waste": red_padding,
        },
        "packing_utilisation": round(accepted_real / packed_positions, 4) if packed_positions else 0,
        "opus": {
            "candidates": len(opus),
            "accept_rate": round(n_accept / n_total, 4),
            "reject_rate": round(n_reject / n_total, 4),
            "defer_rate": round(n_defer / n_total, 4),
        },
        "gpu_idle_loader_wait_fraction": round(loader_fraction, 4),
        "colour_breakdown_pct": _colours(green, amber, red_padding, loader_fraction),
    }
    write_json(out_path, report)
    return report


def _colours(green, amber, red, loader_fraction):
    # token-based green/amber/red, scaled into the compute-time budget with a
    # gray slice for measured loader wait.
    tok_total = max(1, green + amber + red)
    busy = max(0.0, 1.0 - loader_fraction)
    return {
        "green": round(busy * green / tok_total * 100, 1),
        "amber": round(busy * amber / tok_total * 100, 1),
        "red": round(busy * red / tok_total * 100, 1),
        "gray": round(loader_fraction * 100, 1),
    }
