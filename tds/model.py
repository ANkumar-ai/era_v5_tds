"""A tiny, fully-deterministic model that *genuinely learns*.

The assignment says training may be "fake" -- the point is the data system, not
the GPU. But a fake constant loss would make the learning ledger meaningless.
So instead of faking loss, we use a real (if tiny) model: a Laplace-smoothed
bigram language model implemented in pure Python.

Two properties make it perfect for this POC:

  1. It has NO randomness. Given the same tokens in the same order it produces
     the same loss and the same counts on any machine -- which is what makes
     replay bit-identical.
  2. Its cold-start loss is exactly ln(vocab_size), matching the theory Rohan
     walks through in the session (11.78 for a 131k vocab, because an untrained
     model over V tokens guesses 1/V). As it sees data, loss genuinely drops,
     so "this shard is already learned" (low loss vs. the running average) is a
     real, measured signal -- not a hardcoded number.
"""
from __future__ import annotations

import math

from .tokenizer import PAD


class BigramModel:
    def __init__(self, vocab_size: int, alpha: float = 0.1):
        self.V = vocab_size
        self.alpha = alpha                      # Laplace smoothing
        # context token -> {next token -> count}
        self.counts: dict[int, dict[int, int]] = {}
        self.totals: dict[int, int] = {}
        self.tokens_seen = 0
        self._loss_sum = 0.0                    # for running average loss
        self._loss_n = 0

    # ----- probability / loss -----------------------------------------
    def token_loss(self, prev: int, tok: int) -> float:
        """Cross-entropy loss (nats) of predicting `tok` given context `prev`.

        With no observations P = alpha/(alpha*V) = 1/V  ->  loss = ln(V).
        """
        ctx = self.counts.get(prev)
        c = ctx.get(tok, 0) if ctx else 0
        total = self.totals.get(prev, 0)
        p = (c + self.alpha) / (total + self.alpha * self.V)
        return -math.log(p)

    def cold_start_loss(self) -> float:
        return math.log(self.V)

    @property
    def running_avg_loss(self) -> float:
        if self._loss_n == 0:
            return self.cold_start_loss()
        return self._loss_sum / self._loss_n

    # ----- evaluation (no state change) -------------------------------
    def eval_sequence(self, seq: dict) -> dict:
        """Per-token loss over the loss-masked tokens of one packed sequence.

        Context is the previous token *within the same segment* (so packed
        documents never leak context across the attention boundary). Returns
        per-token losses plus aggregate loss and perplexity.
        """
        toks, mask, seg = seq["tokens"], seq["loss_mask"], seq["segment_ids"]
        losses, token_ids = [], []
        for i in range(len(toks)):
            if mask[i] != 1:
                continue
            prev = toks[i - 1] if (i > 0 and seg[i - 1] == seg[i]) else PAD
            losses.append(self.token_loss(prev, toks[i]))
            token_ids.append(toks[i])
        avg = sum(losses) / len(losses) if losses else 0.0
        return {
            "n_loss_tokens": len(losses),
            "avg_loss": avg,
            "perplexity": math.exp(avg) if losses else 0.0,
            "token_losses": losses,
            "token_ids": token_ids,
            "max_token_loss": max(losses) if losses else 0.0,
        }

    # ----- update (the "gradient step") -------------------------------
    def train_sequence(self, seq: dict) -> dict:
        """Evaluate, then learn from the loss-masked tokens (the SGD analogue).

        Learning = incrementing bigram counts for the tokens that carry loss.
        For an SFT/agentic sequence only the response tokens carry loss, so only
        the response bigrams are learned -- exactly the masking rule.

        `loss_delta` (loss before minus a re-eval after) is what the learning
        ledger stores to show the shard actually taught the model something.
        """
        before = self.eval_sequence(seq)
        toks, mask, seg = seq["tokens"], seq["loss_mask"], seq["segment_ids"]
        grad_mass = 0
        for i in range(len(toks)):
            if mask[i] != 1:
                continue
            prev = toks[i - 1] if (i > 0 and seg[i - 1] == seg[i]) else PAD
            self.counts.setdefault(prev, {})
            self.counts[prev][toks[i]] = self.counts[prev].get(toks[i], 0) + 1
            self.totals[prev] = self.totals.get(prev, 0) + 1
            self.tokens_seen += 1
            grad_mass += 1
        # fold this sequence's loss into the running average (model "phase")
        self._loss_sum += before["avg_loss"] * before["n_loss_tokens"]
        self._loss_n += before["n_loss_tokens"]
        after = self.eval_sequence(seq)
        return {
            "avg_loss": before["avg_loss"],
            "perplexity": before["perplexity"],
            "max_token_loss": before["max_token_loss"],
            "loss_delta": before["avg_loss"] - after["avg_loss"],
            "grad_norm_proxy": float(grad_mass),   # tokens that produced an update
            "n_loss_tokens": before["n_loss_tokens"],
            "token_losses": before["token_losses"],
            "token_ids": before["token_ids"],
        }

    # ----- (de)serialization for checkpoints --------------------------
    def state_dict(self) -> dict:
        # canonical, sorted -> identical bytes for identical model state
        counts = [[p, sorted(d.items())] for p, d in sorted(self.counts.items())]
        return {
            "vocab_size": self.V,
            "alpha": self.alpha,
            "counts": counts,
            "totals": sorted(self.totals.items()),
            "tokens_seen": self.tokens_seen,
            "loss_sum": self._loss_sum,
            "loss_n": self._loss_n,
        }

    def load_state_dict(self, sd: dict) -> None:
        self.V = sd["vocab_size"]
        self.alpha = sd["alpha"]
        self.counts = {p: dict(d) for p, d in sd["counts"]}
        self.totals = {p: t for p, t in sd["totals"]}
        self.tokens_seen = sd["tokens_seen"]
        self._loss_sum = sd["loss_sum"]
        self._loss_n = sd["loss_n"]

    def phase(self, schedule_total_tokens: int) -> str:
        """Coarse training phase from progress, used to tag ledger rows."""
        frac = self.tokens_seen / max(1, schedule_total_tokens)
        if frac < 0.25:
            return "early"
        if frac < 0.6:
            return "mid"
        if frac < 0.9:
            return "late"
        return "anneal"
