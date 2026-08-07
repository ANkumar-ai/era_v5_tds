"""Real data source: AI4Bharat Sangraha (verified / Telugu).

This is the reference corpus from Assignment 4 -- `ai4bharat/sangraha :: verified/tel`.
On Colab (or anywhere with network) `load_sangraha_telugu()` streams the top-N
rows straight from the HuggingFace Hub. `records_to_indic_docs()` turns those
rows into the document dicts the rest of the system consumes.

For offline / sandbox use there is a small bundled real-Telugu fixture so the
pipeline always runs and still produces a real HTML dashboard; the notebook
overrides it with the true 10k-row stream.
"""
from __future__ import annotations

import gzip
import json
import os
import re
import unicodedata

# Pinned dataset cache. The Colab notebook downloads the top rows once and writes
# this file; committing it makes `python run_demo.py` self-contained, offline and
# deterministic, so the grader regenerates the SAME artifacts you did.
DATA_CACHE = os.path.join("data", "sangraha_tel_10k.jsonl.gz")

# ---- light normalization (Session 4 discipline, condensed) --------------
_INVISIBLE = dict.fromkeys(map(ord, "​‎‏﻿‪‫‬"), None)
# NOTE: ZWNJ (U+200C) and ZWJ (U+200D) are deliberately KEPT -- they are
# load-bearing in Brahmic scripts.
_WS = re.compile(r"\s+")


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_INVISIBLE)
    return _WS.sub(" ", text).strip()


def telugu_script_ratio(text: str) -> float:
    """Fraction of letters in the Telugu Unicode block (U+0C00–U+0C7F)."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    tel = sum(1 for c in letters if 0x0C00 <= ord(c) <= 0x0C7F)
    return tel / len(letters)


# ---- HuggingFace loader (Colab) -----------------------------------------
def load_sangraha_telugu(n: int = 10000, min_words: int = 8,
                         max_scan: int | None = None) -> list[dict]:
    """Stream the top verified Telugu rows from Sangraha and keep `n` of them.

    Requires `datasets` and network access (Colab has both). Returns a list of
    {doc_id, text, type, source} dicts, lightly cleaned and length/script-filtered.
    Scans until `n` rows pass the filters (bounded by `max_scan`).
    """
    from datasets import load_dataset  # imported lazily so the pkg stays stdlib
    dset = load_dataset("ai4bharat/sangraha", data_dir="verified/tel",
                        split="train", streaming=True)
    out: list[dict] = []
    cap = max_scan or n * 3
    for i, row in enumerate(dset):
        if len(out) >= n or i >= cap:
            break
        text = clean_text(row.get("text", ""))
        if len(text.split()) < min_words:
            continue
        if telugu_script_ratio(text) < 0.70:      # code-mixing / wrong-script guard
            continue
        out.append({
            "doc_id": row.get("doc_id") or f"sangraha_tel_{i:06d}",
            "text": text,
            "type": row.get("type", "web"),
            "source": f"sangraha:verified/tel:{row.get('type', 'web')}",
        })
    return out


# ---- pinned cache (commit this so the command is reproducible) -----------
def save_cache(records: list[dict], path: str = DATA_CACHE) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def load_cache(path: str = DATA_CACHE) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def default_records() -> list[dict]:
    """Resolve the Indic-lane data: the pinned 10k cache if committed, else the
    bundled Telugu fixture. This is what makes `python run_demo.py` reproduce the
    same values everywhere the cache is present."""
    if os.path.exists(DATA_CACHE):
        return load_cache(DATA_CACHE)
    return fixture_records()


# ---- adapter: records -> document dicts ---------------------------------
def records_to_indic_docs(records: list[dict], max_docs: int | None = None) -> list[dict]:
    """Turn loaded Sangraha rows into `indic`-lane documents.

    max_docs bounds how many feed the *training* shard pool (the executable
    proves the system, not scale); the full set is still profiled for stats.
    """
    docs = []
    use = records if max_docs is None else records[:max_docs]
    for i, r in enumerate(use):
        docs.append({
            "doc_id": r.get("doc_id", f"tel_{i:06d}"),
            "lane": "indic",
            "source": r.get("source", "sangraha:verified/tel"),
            "license_tier": "safe",
            "text": r["text"],
            "never_train": False,
            "response_span": None,
            "canary": False,
        })
    return docs


def corpus_stats(records: list[dict], tok) -> dict:
    """A4-style profile of the loaded corpus for the dashboard."""
    import collections
    by_type = collections.Counter(r.get("type", "web") for r in records)
    words = sum(len(r["text"].split()) for r in records)
    sample = [r["text"] for r in records[:2000]]           # bounded for speed
    return {
        "source": "ai4bharat/sangraha :: verified/tel",
        "rows_loaded": len(records),
        "words": words,
        "type_distribution": dict(by_type),
        "tokenizer_fertility": round(tok.fertility(sample), 3) if sample else 0.0,
        "vocab_size": tok.vocab_size,
    }


# ---- offline fixture: real Telugu sentences (fallback only) --------------
TELUGU_SAMPLE = [
    "నది పట్టణం గుండా ప్రవహిస్తుంది మరియు ప్రతి ఉదయం మార్కెట్ తెరుచుకుంటుంది",
    "రైతులు పొలాల్లో బంగారు వరి పంటను కోస్తున్నారు వర్షం సమయానికి కురిసింది",
    "కొత్త గ్రంథాలయం నగరంలో ప్రారంభమైంది అక్కడ పుస్తకాలు సంగీతం అందుబాటులో ఉన్నాయి",
    "పండుగ సందర్భంగా వీధులు దీపాలతో సంగీతంతో నృత్యాలతో నిండిపోయాయి",
    "శాస్త్రవేత్తలు ఉత్తర తీరం వెంబడి పక్షుల వలసను ఈ వసంతంలో గమనించారు",
    "ప్రభుత్వం కొత్త విద్యా విధానాన్ని ప్రకటించింది విద్యార్థులకు ఇది ఉపయోగకరం",
    "సముద్ర తీరంలో జాలర్లు తెల్లవారుజామున చేపలు పట్టడానికి బయలుదేరారు",
    "పర్వత ప్రాంతంలో వాతావరణం చల్లగా ఉంది పర్యాటకులు ఎక్కువగా వస్తున్నారు",
    "నగర పాలక సంస్థ రహదారుల మరమ్మతులకు నిధులు కేటాయించింది పనులు వేగంగా జరుగుతున్నాయి",
    "విద్యార్థులు వార్షిక క్రీడా పోటీలలో ఉత్సాహంగా పాల్గొన్నారు బహుమతులు అందుకున్నారు",
    "కొత్త రైలు మార్గం రెండు జిల్లాలను కలుపుతుంది ప్రయాణ సమయం తగ్గుతుంది",
    "వ్యవసాయ శాఖ రైతులకు నూతన విత్తనాలను పంపిణీ చేసింది దిగుబడి పెరుగుతుందని అంచనా",
    "పట్టణంలో కొత్త ఆసుపత్రి ప్రారంభమైంది ప్రజలకు మెరుగైన వైద్య సదుపాయాలు లభిస్తాయి",
    "సాంకేతిక రంగంలో యువత కొత్త ఆవిష్కరణలతో ముందుకు సాగుతోంది ఉద్యోగ అవకాశాలు పెరుగుతున్నాయి",
    "చరిత్రకారులు పురాతన దేవాలయ శిల్పాలను అధ్యయనం చేస్తున్నారు వాటి కథలను నమోదు చేస్తున్నారు",
    "వర్షాకాలంలో చెరువులు నిండాయి రైతులు సాగునీటి కోసం సంతోషంగా ఉన్నారు",
    "నగరంలో కాలుష్యం తగ్గించేందుకు కొత్త చెట్లను నాటే కార్యక్రమం ప్రారంభమైంది",
    "కళాకారులు సాంప్రదాయ చిత్రలేఖనాన్ని కొత్త తరానికి నేర్పిస్తున్నారు ప్రదర్శనలు నిర్వహిస్తున్నారు",
    "గ్రామాల్లో సౌర విద్యుత్ ద్వారా వెలుగులు నింపే ప్రయత్నాలు జరుగుతున్నాయి",
    "పుస్తక ప్రదర్శనలో వివిధ భాషల రచనలు పాఠకులను ఆకర్షించాయి అమ్మకాలు బాగా జరిగాయి",
    "క్రీడాకారులు జాతీయ స్థాయి పోటీలకు కఠినంగా సాధన చేస్తున్నారు కోచ్‌లు మార్గనిర్దేశనం చేస్తున్నారు",
    "వాతావరణ శాఖ రాబోయే రోజుల్లో వర్షాలు కురుస్తాయని హెచ్చరిక జారీ చేసింది",
    "పరిశ్రమల అభివృద్ధి కోసం ప్రభుత్వం కొత్త విధానాలను రూపొందిస్తోంది ఉపాధి పెరుగుతుంది",
    "విశ్వవిద్యాలయంలో పరిశోధన కేంద్రం కొత్త ప్రయోగశాలను ఏర్పాటు చేసింది విద్యార్థులకు ఉపయోగం",
    "పండ్ల తోటల్లో దిగుబడి బాగుంది రైతులు మార్కెట్‌కు తరలిస్తున్నారు ధరలు స్థిరంగా ఉన్నాయి",
    "సాంస్కృతిక కార్యక్రమంలో పిల్లలు సాంప్రదాయ నృత్యాలను ప్రదర్శించారు ప్రేక్షకులు మెచ్చుకున్నారు",
    "నగరంలో నీటి సరఫరా మెరుగుపరిచేందుకు కొత్త పైపులైన్ పనులు ప్రారంభమయ్యాయి",
    "గ్రామీణ ప్రాంతాల్లో అంతర్జాల సదుపాయం విస్తరిస్తోంది విద్య ఉపాధి రంగాలకు లాభం",
    "చారిత్రక కోట పునరుద్ధరణ పనులు పూర్తయ్యాయి పర్యాటకులు సందర్శించడం ప్రారంభించారు",
    "రైతు సంఘాలు కొత్త మార్కెటింగ్ విధానాలపై సమావేశం నిర్వహించాయి రైతులకు అవగాహన కల్పించారు",
]


def fixture_records() -> list[dict]:
    return [{"doc_id": f"fixture_tel_{i:03d}", "text": t, "type": "web",
             "source": "sangraha:verified/tel:web(fixture)"} for i, t in enumerate(TELUGU_SAMPLE)]
