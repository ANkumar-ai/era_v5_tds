"""Curriculum stages, lane weights and protected floors -> executable schedule.

Session 5 produced a *plan*; this compiles it into something the dataloader can
execute step by step. The schedule knows, for each stage: the token range, the
sequence length, the target lane mixture, and the protected floors (hard
minimums a lane may never drop below). The compiler checks that every weighted
lane actually has verified supply; if not it warns and suggests a remedy
(repeat / synthesize / reduce share / postpone) instead of silently starving.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Stage:
    name: str
    token_start: int
    token_end: int
    sequence_length: int
    mixture: dict            # lane -> target share (sums ~1.0)
    protected_floors: dict   # lane -> hard minimum share
    warmup_tokens: int = 0


def default_schedule(seq_len_foundation=16, seq_len_mid=16) -> list[Stage]:
    """Two tiny curriculum stages (scaled down from the ~trillion-token real
    ranges). Foundation is web/code heavy; the mid stage tilts toward reasoning
    while holding indic and agentic above their floors."""
    return [
        Stage(
            name="foundation",
            token_start=0, token_end=600,
            sequence_length=seq_len_foundation,
            mixture={"general_web": 0.34, "code": 0.22, "math_science": 0.14,
                     "reasoning": 0.12, "indic": 0.12, "agentic": 0.06},
            protected_floors={"indic": 0.08, "agentic": 0.03, "reasoning": 0.05},
            warmup_tokens=32,
        ),
        Stage(
            name="reasoning-heavy-midtrain",
            token_start=600, token_end=1200,
            sequence_length=seq_len_mid,
            mixture={"general_web": 0.24, "code": 0.20, "math_science": 0.16,
                     "reasoning": 0.22, "indic": 0.12, "agentic": 0.06},
            protected_floors={"indic": 0.08, "agentic": 0.03, "reasoning": 0.10},
            warmup_tokens=0,
        ),
    ]


def compile_schedule(stages: list[Stage], available_lanes: set[str],
                     logger=None) -> dict:
    """Validate supply and normalise weights. Returns a compiled, serialisable
    schedule plus any warnings/remedies."""
    warnings = []
    compiled = []
    for st in stages:
        norm = _normalise(st.mixture)
        for lane, w in st.mixture.items():
            if w > 0 and lane not in available_lanes:
                remedy = "reduce_share_or_synthesize"
                warnings.append({"stage": st.name, "lane": lane,
                                 "issue": "no_verified_supply", "remedy": remedy})
                if logger:
                    logger.event(f"[WARN] stage {st.name}: lane '{lane}' has no "
                                 f"verified supply -> suggest {remedy}")
        # floors must not exceed target share
        for lane, floor in st.protected_floors.items():
            if floor > norm.get(lane, 0) + 1e-9:
                warnings.append({"stage": st.name, "lane": lane,
                                 "issue": "floor_above_target"})
        compiled.append({
            "name": st.name,
            "token_start": st.token_start,
            "token_end": st.token_end,
            "sequence_length": st.sequence_length,
            "mixture": norm,
            "protected_floors": st.protected_floors,
            "warmup_tokens": st.warmup_tokens,
        })
    return {"stages": compiled, "warnings": warnings}


def _normalise(mixture: dict) -> dict:
    total = sum(mixture.values())
    if total == 0:
        return dict(mixture)
    return {k: v / total for k, v in mixture.items()}


def stage_at(schedule: dict, token_pos: int) -> dict:
    for st in schedule["stages"]:
        if st["token_start"] <= token_pos < st["token_end"]:
            return st
    return schedule["stages"][-1]
