# ERA V5 — Training Data Execution System · Evidence

**Status:** 26 checks passed, 1 failed (of 27) · _Statuses are produced by the run from emitted PASS/FAIL markers. Point scoring is left to the evaluator; no self-assigned score._

## Key numbers

- **indic_rows**: 10000
- **shards**: 113
- **eval_shards_blocked**: 1
- **steps**: 24
- **cold_start_loss**: 10.5966
- **final_batch_loss**: 4.34
- **useful_loss_tokens_per_sec**: 30225.4
- **packing_utilisation**: 0.746
- **opus_accept_rate**: 0.3174

## Requirements

| Requirement | Result | Evidence | Detail |
|---|---|---|---|
| Tokenizer integrity | PASS | `manifests/tokenizer.json` | Frozen hash reproduces; changes iff vocab changes |
| Shard immutability | PASS | `manifests/` | Editing a shard yields a new content hash + lineage; manifests revalidated |
| Evaluation firewall | PASS | `run.log` | never_train + canary shards blocked from loss-bearing batches |
| Packing correctness | PASS | `run.log` | No cross-doc attention; response-only loss; position reset |
| Mixture compliance | PASS | `manifests/schedule.json + ledgers/consumption.jsonl` | Planned mixture vs realized lane shares; scarce lanes held above floor |
| OPUS audit trail | PASS | `ledgers/opus.jsonl` | accept/reject/defer + protected-floor override recorded |
| Learning trace | FAIL | `ledgers/learning.jsonl` | Per-shard loss linked to source; already-learned flagged |
| Crash recovery | PASS | `ledgers/consumption.jsonl` | Next batch after resume is exact; no skip/repeat |
| Replay | PASS | `ledgers/consumption.jsonl` | Replayed interval reproduces identical hashes + spans |
| Fork | PASS | `checkpoints/fork_lineage.json` | Branch from earlier checkpoint diverges w/ lineage |
| Audit | PASS | `ledgers/audit_report.json` | Shards for a step range reconstructed |
| Throughput | PASS | `performance.json` | Useful loss-bearing tokens/sec measured |

## Limits & honest readings

- Indic lane = real Sangraha verified/tel (10000 rows profiled). Other lanes are small synthetic stubs so the mixture / OPUS / floor machinery has multiple lanes; the Telugu lane is the star.
- Toy training scale by design: the executable trains a bounded slice, not ~128M-token shards. The architecture and its invariants are the deliverable.
- The 'model' is a Laplace-smoothed bigram LM — it learns genuinely yet has zero randomness, so replay is bit-identical. Not a transformer; loss magnitudes are illustrative, the mechanism (incl. ln(V) cold start) is real.
- Throughput wall-clock (tokens/sec) is host-dependent; the token accounting behind it is fully reconstructable from the ledgers and byte-identical across runs.
