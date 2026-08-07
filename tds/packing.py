"""Packing policies + per-sequence training metadata.

Hardware wants fixed-length windows; natural text does not come that way.
Padding wastes compute (and artificially lowers loss); packing fills windows
with useful tokens. HOW you pack depends on the data:

    concat-and-chop     -> pretraining (general_web, math, reasoning): boundaries
                           are engineering boundaries, chopping mid-doc is fine.
    structure-preserving-> code / agentic: a function or reasoning trace must
                           stay intact, never split across sequences.
    best-fit            -> minimise padding for lanes with few/odd-length docs
                           (indic) by bin-packing docs into windows.

Every packed sequence carries the three things a batch needs beyond token ids:
    loss_mask      1 = learn this token, 0 = ignore (pad, or the prompt half of
                   an SFT/agentic example where only the response is scored).
    segment_ids    document id per token; two tokens may attend only within the
                   same segment => block-diagonal attention, no cross-doc leak.
                   Pad tokens get segment -1.
    position_ids   token order, reset to 0 at each document boundary (RoPE-style).
"""
from __future__ import annotations

from .tokenizer import PAD, EOS
from .shards import Shard

# which policy each lane uses
LANE_POLICY = {
    "general_web": "concat_and_chop",
    "math_science": "concat_and_chop",
    "reasoning": "concat_and_chop",
    "code": "structure_preserving",
    "agentic": "structure_preserving",
    "indic": "best_fit",
}


def policy_for(lane: str) -> str:
    return LANE_POLICY.get(lane, "concat_and_chop")


def _token_doc_map(shard: Shard) -> list[int]:
    """Map each token position in the shard to the index of its document."""
    m = [0] * len(shard.tokens)
    for di, span in enumerate(shard.doc_spans):
        for p in range(span["start"], span["end"]):
            m[p] = di
    return m


def _response_positions(shard: Shard) -> set[int]:
    """Token positions that belong to an SFT 'response' (loss-bearing) span."""
    resp = set()
    for span in shard.doc_spans:
        rt = span.get("response_token_span")
        if rt:
            resp.update(range(rt[0], rt[1]))
    return resp


def _empty_seq(seq_len, shard, policy):
    return {
        "tokens": [PAD] * seq_len,
        "loss_mask": [0] * seq_len,
        "segment_ids": [-1] * seq_len,
        "position_ids": [0] * seq_len,
        "doc_spans": [],
        "source_shard": shard.shard_id,
        "lane": shard.lane,
        "policy": policy,
    }


def _finalize(seq, seq_len):
    seq["n_tokens_real"] = sum(1 for s in seq["segment_ids"] if s != -1)
    seq["n_pad"] = seq_len - seq["n_tokens_real"]
    seq["n_loss"] = sum(seq["loss_mask"])
    return seq


def pack_shard(shard: Shard, seq_len: int) -> list[dict]:
    policy = policy_for(shard.lane)
    if policy == "concat_and_chop":
        return _pack_concat(shard, seq_len)
    if policy == "structure_preserving":
        return _pack_structured(shard, seq_len)
    if policy == "best_fit":
        return _pack_best_fit(shard, seq_len)
    raise ValueError(policy)


# --------------------------------------------------------------------------
def _pack_concat(shard: Shard, seq_len: int) -> list[dict]:
    """Join docs (EOS already between them), cut fixed windows sequentially."""
    tokens = shard.tokens
    dmap = _token_doc_map(shard)
    seqs = []
    for base in range(0, len(tokens), seq_len):
        window = tokens[base: base + seq_len]
        seq = _empty_seq(seq_len, shard, "concat_and_chop")
        cur_seg = -1
        cur_doc = None
        pos = 0
        seg_counter = 0
        span_starts = {}
        for i, t in enumerate(window):
            doc_idx = dmap[base + i]
            if doc_idx != cur_doc:
                cur_doc = doc_idx
                seg_counter += 0 if cur_seg == -1 else 1
                cur_seg = seg_counter
                pos = 0
                span_starts[doc_idx] = i
            seq["tokens"][i] = t
            seq["segment_ids"][i] = cur_seg
            seq["position_ids"][i] = pos
            seq["loss_mask"][i] = 0 if t == PAD else 1  # pretraining: learn all real
            pos += 1
        # record provenance spans present in this window
        seen = {}
        for i in range(len(window)):
            d = dmap[base + i]
            seen.setdefault(d, [i, i])
            seen[d][1] = i
        seq["doc_spans"] = [
            {"doc_id": shard.doc_spans[d]["doc_id"], "seq_start": s, "seq_end": e + 1}
            for d, (s, e) in sorted(seen.items())
        ]
        seqs.append(_finalize(seq, seq_len))
    return seqs


def _pack_structured(shard: Shard, seq_len: int) -> list[dict]:
    """One document per sequence; never split a trace. Response-only loss when
    the doc marks a response span (SFT/agentic)."""
    resp = _response_positions(shard)
    seqs = []
    for di, span in enumerate(shard.doc_spans):
        doc_tokens = shard.tokens[span["start"]: span["end"]]
        has_resp = span.get("response_token_span") is not None
        # chunk only if a single doc exceeds the window (keeps as few splits as
        # possible); POC docs fit in one window.
        for off in range(0, len(doc_tokens), seq_len):
            chunk = doc_tokens[off: off + seq_len]
            seq = _empty_seq(seq_len, shard, "structure_preserving")
            for i, t in enumerate(chunk):
                global_pos = span["start"] + off + i
                seq["tokens"][i] = t
                seq["segment_ids"][i] = 0
                seq["position_ids"][i] = off + i
                if t == PAD:
                    seq["loss_mask"][i] = 0
                elif has_resp:
                    seq["loss_mask"][i] = 1 if global_pos in resp else 0
                else:
                    seq["loss_mask"][i] = 1
            seq["doc_spans"] = [{"doc_id": span["doc_id"], "seq_start": 0,
                                 "seq_end": len(chunk)}]
            seqs.append(_finalize(seq, seq_len))
    return seqs


def _pack_best_fit(shard: Shard, seq_len: int) -> list[dict]:
    """Sort docs by length desc and greedily pack into the emptiest-fitting
    window (bin packing) to minimise padding waste."""
    # pre-split any document longer than the window into window-sized pieces
    docs = []
    for span in shard.doc_spans:
        toks = shard.tokens[span["start"]: span["end"]]
        if len(toks) <= seq_len:
            docs.append((span, toks))
        else:
            for off in range(0, len(toks), seq_len):
                docs.append((span, toks[off: off + seq_len]))
    docs.sort(key=lambda x: -len(x[1]))
    bins: list[list[tuple]] = []  # each bin: list of (span, tokens)
    used: list[int] = []
    for span, toks in docs:
        placed = False
        # first-fit-decreasing
        for bi in range(len(bins)):
            if used[bi] + len(toks) <= seq_len:
                bins[bi].append((span, toks))
                used[bi] += len(toks)
                placed = True
                break
        if not placed:
            bins.append([(span, toks)])
            used.append(len(toks))

    seqs = []
    for b in bins:
        seq = _empty_seq(seq_len, shard, "best_fit")
        i = 0
        for seg_id, (span, toks) in enumerate(b):
            start_i = i
            for j, t in enumerate(toks):
                seq["tokens"][i] = t
                seq["segment_ids"][i] = seg_id
                seq["position_ids"][i] = j
                seq["loss_mask"][i] = 0 if t == PAD else 1
                i += 1
            seq["doc_spans"].append({"doc_id": span["doc_id"],
                                     "seq_start": start_i, "seq_end": i})
        seqs.append(_finalize(seq, seq_len))
    return seqs


# --------------------------------------------------------------------------
def attention_allowed(segment_ids: list[int], q: int, k: int) -> bool:
    """True if query position q may attend to key position k.

    Causal (k <= q) AND same non-pad segment. Used by tests to prove no
    cross-document attention leaks through a packed window.
    """
    if segment_ids[q] == -1 or segment_ids[k] == -1:
        return False
    return k <= q and segment_ids[q] == segment_ids[k]
