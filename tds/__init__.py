"""ERA V5 - Training Data Execution System (POC).

A small-but-complete, deterministic, auditable data system that proves the
full path:

    documents -> tokenized shards -> manifests -> mixture schedule -> packing
    -> batches -> training -> consumption ledger -> learning ledger
    -> checkpoint -> crash -> resume -> replay -> fork -> audit

Everything is pure-Python standard library so `python run_demo.py` runs
anywhere with no external dependencies.
"""

SEED = 20250804          # single global seed => full reproducibility
SCHEMA_VERSION = "era-v5-tds-1.0"

__all__ = ["SEED", "SCHEMA_VERSION"]
