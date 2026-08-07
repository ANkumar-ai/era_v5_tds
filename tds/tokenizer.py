"""Frozen, deterministic word-level tokenizer.

A tokenizer is what gives token IDs meaning. If the vocabulary changes, every
token ID silently changes meaning -- so the tokenizer is *frozen* (built once,
hashed) and every shard manifest records the tokenizer_hash. A shard tokenized
with a different tokenizer is a different object.
"""
from __future__ import annotations

from typing import Optional

from .util import canonical_json, sha256_hex

# Reserved special tokens. Fixed IDs -- part of the frozen contract.
PAD = 0   # padding filler (never carries loss)
EOS = 1   # end-of-context marker between documents
UNK = 2   # out-of-vocabulary
BOS = 3   # begin-of-sequence
_SPECIAL = {"<PAD>": PAD, "<EOS>": EOS, "<UNK>": UNK, "<BOS>": BOS}

# Punctuation peeled off word edges. Includes Latin + Indic danda/double-danda
# and common quotes, so Telugu/Devanagari conjuncts+matras stay intact as one
# word (a Unicode \w class would split on combining marks and shatter Brahmic
# scripts -- exactly the failure mode Session 4 warned about).
_PUNCT = set(".,!?;:\"'()[]{}<>/\\|@#$%^&*_+=~`।॥“”‘’—–…-")


def _pretok(text: str) -> list[str]:
    """Whitespace-delimited, script-agnostic pre-tokenizer.

    Splits on whitespace (words are space-separated in Telugu, Hindi and
    English alike), then peels leading/trailing punctuation into their own
    tokens. Brahmic clusters are never broken internally.
    """
    toks: list[str] = []
    for w in text.lower().split():
        i, j = 0, len(w)
        while i < j and w[i] in _PUNCT:      # leading punctuation
            toks.append(w[i]); i += 1
        tail = []
        while j > i and w[j - 1] in _PUNCT:  # trailing punctuation
            tail.append(w[j - 1]); j -= 1
        if i < j:
            toks.append(w[i:j])
        toks.extend(reversed(tail))
    return toks


class Tokenizer:
    def __init__(self, vocab: dict[str, int]):
        self.vocab = vocab
        self.inv = {i: t for t, i in vocab.items()}
        # The hash freezes the vocabulary. Any change => different hash.
        self.tokenizer_hash = "tok_" + sha256_hex(canonical_json(vocab))[:12]

    # ----- construction -------------------------------------------------
    @classmethod
    def build(cls, documents: list[str], max_vocab: int | None = None) -> "Tokenizer":
        """Build a vocab deterministically from a corpus.

        Words are added in a stable order (frequency desc, then lexical) so the
        vocabulary -- and therefore the tokenizer_hash -- is identical every run.
        `max_vocab` caps the vocabulary (rare words fall back to <UNK>), which
        bounds memory and the ln(V) cold-start on a large real corpus.
        """
        counts: dict[str, int] = {}
        for doc in documents:
            for tok in _pretok(doc):
                counts[tok] = counts.get(tok, 0) + 1
        ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        if max_vocab is not None:
            ordered = ordered[:max(0, max_vocab - len(_SPECIAL))]
        vocab = dict(_SPECIAL)
        nxt = len(vocab)
        for tok, _ in ordered:
            if tok not in vocab:
                vocab[tok] = nxt
                nxt += 1
        return cls(vocab)

    def fertility(self, documents: list[str]) -> float:
        """Mean tokens per whitespace word -- the tokenizer-fertility metric."""
        words = toks = 0
        for d in documents:
            w = d.split()
            words += len(w)
            toks += len(self.encode(d, add_eos=False))
        return (toks / words) if words else 0.0

    # ----- use ----------------------------------------------------------
    def encode(self, text: str, add_eos: bool = True) -> list[int]:
        ids = [self.vocab.get(t, UNK) for t in _pretok(text)]
        if add_eos:
            ids.append(EOS)
        return ids

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.inv.get(i, "<UNK>") for i in ids if i not in (PAD,))

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def to_manifest(self) -> dict:
        return {
            "tokenizer_hash": self.tokenizer_hash,
            "vocab_size": self.vocab_size,
            "special_tokens": _SPECIAL,
        }
