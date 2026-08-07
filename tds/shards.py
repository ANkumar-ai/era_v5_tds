"""Immutable tokenized shards + their manifests.

A shard is a frozen, tokenized brick of data. It is *immutable*: its identity
is its content hash. Change a single token and it becomes a different shard
with a different hash and its own lineage. The manifest is the contract the
dataloader trusts -- Sessions 1-5 (cleaning, dedup, screening) speak to the
training loop only through this document.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

from .corpus import CANARY
from .tokenizer import Tokenizer
from .util import hash_obj, hash_tokens, sha256_hex, canonical_json


@dataclass
class Shard:
    shard_id: str
    lane: str
    tokens: list[int]
    doc_spans: list[dict]          # [{doc_id, start, end, response_span?}]
    never_train: bool
    manifest: dict

    @property
    def token_count(self) -> int:
        return len(self.tokens)


def _cleaning_pipeline_hash(lane: str) -> str:
    # Stands in for the Session 1-4 cleaning lineage. Deterministic per lane.
    return "clean_" + sha256_hex(f"pipeline::{lane}::v5")[:12]


def build_shards(docs: list[dict], tok: Tokenizer,
                 target_tokens: int = 60) -> list[Shard]:
    """Group documents by lane and tokenize into small immutable shards.

    target_tokens keeps shards tiny for the POC; real shards are ~128M tokens.
    """
    by_lane: dict[str, list[dict]] = {}
    for d in docs:
        by_lane.setdefault(d["lane"], []).append(d)

    shards: list[Shard] = []
    for lane in sorted(by_lane):
        buf_tokens: list[int] = []
        buf_spans: list[dict] = []
        idx = 0
        lane_docs = by_lane[lane]

        def flush(force=False):
            nonlocal buf_tokens, buf_spans, idx
            if buf_tokens and (force or len(buf_tokens) >= target_tokens):
                shards.append(_make_shard(lane, idx, buf_tokens, buf_spans, tok))
                idx += 1
                buf_tokens, buf_spans = [], []

        for d in lane_docs:
            start = len(buf_tokens)
            ids = tok.encode(d["text"], add_eos=True)
            buf_tokens.extend(ids)
            span = {
                "doc_id": d["doc_id"],
                "start": start,
                "end": len(buf_tokens),
                "source": d["source"],
                "license_tier": d["license_tier"],
                "never_train": d["never_train"],
                "canary": d["canary"],
            }
            if d.get("response_span"):
                # translate word-level response span to token span within the doc.
                # our tokenizer is ~word-level, so word idx ~= token idx here.
                ws, we = d["response_span"]
                span["response_token_span"] = [start + ws, start + we]
            buf_spans.append(span)
            # structure-preserving lanes: one doc per shard to keep traces intact.
            # other lanes: split into ~target_tokens shards once the buffer fills.
            if lane in ("agentic", "code"):
                flush(force=True)
            else:
                flush()
        flush(force=True)
    return shards


def _make_shard(lane, idx, tokens, spans, tok: Tokenizer) -> Shard:
    shard_id = f"v5_{lane}_shard_{idx:03d}"
    content_hash = hash_tokens(tokens)
    never_train = any(s["never_train"] for s in spans)
    has_canary = any(s["canary"] for s in spans)
    manifest = {
        "shard_id": shard_id,
        "capability_lane": lane,
        "token_count": len(tokens),
        "tokenizer_hash": tok.tokenizer_hash,
        "content_hash": content_hash,
        "cleaning_pipeline_hash": _cleaning_pipeline_hash(lane),
        "dedup_status": "passed",
        "pii_screen_status": "screened",
        # eval lanes are flagged as overlapping/never-train by upstream screening
        "eval_overlap_status": "test_material" if never_train else "clear",
        "never_train": never_train,
        "contains_canary": has_canary,
        "license_tier": _lane_license(spans),
        "doc_ids": [s["doc_id"] for s in spans],
        "parent_manifest_ids": [],
    }
    # bind the manifest to its own content
    manifest["manifest_hash"] = hash_obj(manifest, prefix="mf")
    return Shard(shard_id, lane, tokens, spans, never_train, manifest)


def _lane_license(spans) -> str:
    tiers = {s["license_tier"] for s in spans}
    if "restricted" in tiers or "unknown" in tiers:
        return "restricted"
    return "safe"


def modify_shard(shard: Shard, tok: Tokenizer) -> Shard:
    """Return a NEW shard with one token changed.

    Demonstrates immutability: the modified shard gets a fresh content hash and
    records the original as a parent (lineage), rather than mutating in place.
    """
    new_tokens = list(shard.tokens)
    new_tokens[0] = (new_tokens[0] + 1)  # flip a single token
    new = _make_shard(shard.lane + "_mod", 0, new_tokens, shard.doc_spans, tok)
    new.manifest["parent_manifest_ids"] = [shard.manifest["manifest_hash"]]
    new.manifest["manifest_hash"] = hash_obj(
        {k: v for k, v in new.manifest.items() if k != "manifest_hash"}, prefix="mf")
    return new
