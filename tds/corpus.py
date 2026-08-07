"""A tiny multi-lane corpus with provenance and metadata.

Deliberately small (the goal is proof, not scale) but rich enough to exercise
every policy: multiple capability lanes, an SFT-style agentic lane with an
explicit response span, a low-supply lane to trigger a mixture warning + floor
override, and eval/benchmark documents that MUST be blocked by the firewall
(never_train + a canary string).

Each document carries provenance so the ledger can always answer
"where did this token come from".
"""
from __future__ import annotations

# A canary string is a unique fingerprint deliberately embedded in held-out
# eval data. If it ever shows up in a training batch, contamination happened.
CANARY = "CANARY_ERAV5_7f3a91_DO_NOT_TRAIN"


def _doc(doc_id, lane, source, license_tier, text, never_train=False,
         response_span=None, canary=False):
    return {
        "doc_id": doc_id,
        "lane": lane,
        "source": source,
        "license_tier": license_tier,   # safe | restricted | unknown
        "text": text,
        "never_train": never_train,
        "response_span": response_span,  # (word_start, word_end) for SFT loss mask
        "canary": canary,
    }


def build_corpus(indic_docs: list[dict] | None = None) -> list[dict]:
    """Assemble the multi-lane corpus.

    The `indic` lane is the REAL data (Sangraha verified/tel) when `indic_docs`
    is provided; otherwise it falls back to the bundled Telugu fixture. The other
    lanes are small synthetic stubs so the mixture / curriculum / OPUS / floor
    machinery has multiple lanes to schedule -- the Indic lane is the star, the
    rest exercise the system.
    """
    docs: list[dict] = []

    # ---- general_web -------------------------------------------------
    web = [
        "the river flows past the old town and the market opens at dawn every day",
        "clouds gather over the valley while farmers harvest the golden wheat fields",
        "a new library opened downtown offering books music and quiet reading rooms",
        "travelers crossed the bridge as the evening lights reflected on the water",
        "the festival filled the square with music dancing food and bright lanterns",
        "scientists observed the migration of birds across the northern coast this spring",
    ]
    for i, t in enumerate(web):
        docs.append(_doc(f"web_{i:03d}", "general_web", "commoncrawl_sample", "safe", t))

    # ---- code (structure-preserving; must not be split mid-function) -
    code = [
        "def add numbers a b return a plus b end function computes the sum of two",
        "for item in list process item append result to output then return output list",
        "class stack push pop peek uses an internal array to store the elements safely",
        "function binary search array target while low leq high compute mid compare values",
    ]
    for i, t in enumerate(code):
        docs.append(_doc(f"code_{i:03d}", "code", "github_permissive", "safe", t))

    # ---- reasoning ---------------------------------------------------
    reasoning = [
        "if all birds can fly and a penguin is a bird then reconsider the premise carefully",
        "step one define the variables step two write the equation step three solve for x",
        "assume the opposite is true derive a contradiction therefore the claim must hold",
    ]
    for i, t in enumerate(reasoning):
        docs.append(_doc(f"reason_{i:03d}", "reasoning", "curated_reasoning", "safe", t))

    # ---- math_science ------------------------------------------------
    math = [
        "the derivative of x squared is two x and the integral of two x is x squared plus c",
        "force equals mass times acceleration and energy equals mass times speed of light squared",
    ]
    for i, t in enumerate(math):
        docs.append(_doc(f"math_{i:03d}", "math_science", "textbook_openstax", "safe", t))

    # ---- indic: the REAL Sangraha verified/tel lane (or bundled fixture) ----
    if indic_docs is None:
        from .datasource import fixture_records, records_to_indic_docs
        indic_docs = records_to_indic_docs(fixture_records())
    docs.extend(indic_docs)

    # ---- agentic (SFT-style: only the RESPONSE span carries loss) ----
    # words: 0 user 1 asks 2 how 3 to 4 sort 5 a 6 list 7 assistant 8 use ...
    agentic_text = ("user asks how to sort a list assistant use a stable sort "
                    "compare pairs and swap until ordered then return the list")
    resp_start = agentic_text.split().index("use")  # response begins here
    resp_end = len(agentic_text.split())
    docs.append(_doc("agent_000", "agentic", "agent_traces", "safe",
                     agentic_text, response_span=(resp_start, resp_end)))
    agentic_text2 = ("user asks to reverse a string assistant iterate from the end "
                     "to the start collecting characters then join them together")
    rs2 = agentic_text2.split().index("iterate")
    docs.append(_doc("agent_001", "agentic", "agent_traces", "safe",
                     agentic_text2, response_span=(rs2, len(agentic_text2.split()))))

    # ---- restricted license (OPUS will treat cautiously) -------------
    docs.append(_doc("web_restricted_000", "general_web", "scraped_unknown",
                     "restricted",
                     "premium article content behind a paywall with unclear reuse terms here"))

    # ---- EVAL / NEVER-TRAIN (firewall MUST block these) --------------
    docs.append(_doc("eval_mmlu_000", "eval_holdout", "benchmark_mmlu", "safe",
                     "which of the following best describes the capital city choose one option",
                     never_train=True))
    docs.append(_doc("eval_canary_000", "eval_holdout", "benchmark_private", "safe",
                     f"held out question {CANARY} the answer key must never be trained on",
                     never_train=True, canary=True))
    return docs
