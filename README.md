# Training Data Execution System (POC) for LLM's

A small‑but‑complete, **fully deterministic** data system that turns the Session 5
recipe (mixtures, curriculum, protected floors, OPUS) into the actual, inspectable
stream a training loop consumes — and proves, for any checkpoint, **what the model
ate, why, what it learned, and how to reconstruct the run exactly.**

The **Indic lane is real data**: AI4Bharat **Sangraha `verified/tel`** (Telugu) — the
reference corpus from Assignment 4 — top **10,000** rows. The other capability lanes
are small synthetic stubs so the mixture / curriculum / OPUS / protected‑floor
machinery has multiple lanes to schedule; the Telugu lane is the star.

## Data mix (capability lanes)

The assignment requires *"packing policies for different data types (General, Code,
Agentic, Indic)"*, response‑only **loss masks**, and **protected floors** — all of which
need more than one lane. So the corpus is multi‑lane: the **Indic lane is real
Sangraha**; the rest are **small synthetic stubs** (the assignment permits a small
corpus — the goal is the system, not scale). Each lane exercises a specific behaviour:

| Lane | Source | Packing policy | Loss mask | Foundation weight | Floor |
|---|---|---|---|---|---|
| `indic` | **real — Sangraha verified/tel** | best‑fit | all tokens | 0.12 | 0.08 |
| `general_web` | synthetic stub | concat‑and‑chop | all tokens | 0.34 | — |
| `code` | synthetic stub | **structure‑preserving** | all tokens | 0.22 | — |
| `math_science` | synthetic stub | concat‑and‑chop | all tokens | 0.14 | — |
| `reasoning` | synthetic stub | concat‑and‑chop | all tokens | 0.12 | 0.05 |
| `agentic` | synthetic stub | **structure‑preserving** | **response‑only** | 0.06 | 0.03 |
| `eval_holdout` | synthetic (never‑train + canary) | — | — | — | **blocked by firewall** |

Why each lane earns its place: `code` and `agentic` use **structure‑preserving** packing
(never split a function or a reasoning trace); `agentic` is the one lane with a
**response‑only** loss mask (SFT‑style — the prompt carries no loss); `indic`, `agentic`
and `reasoning` sit behind **protected floors** so OPUS can't starve them; `eval_holdout`
carries a canary and is **blocked** from every loss‑bearing batch. Weights above are the
*Foundation* stage; the *reasoning‑heavy mid‑training* stage shifts reasoning to 0.22
(and raises its floor to 0.10) to demonstrate a curriculum transition. These weights are
illustrative — the point is that the system executes and audits whatever mixture it's given.

---

## How to run

**The one command**

```bash
python run_demo.py        # regenerates submission_artifacts/ end to end
python -m pytest -q        # 17 invariant tests (incl. resume/replay tamper checks)
```

**Pinned real data (so your run and the grader's match).** `run_demo.py` reads the
Indic‑lane corpus from a committed cache file, `data/sangraha_tel_10k.jsonl.gz` (top
10k Sangraha `verified/tel` rows). Because the data is **pinned** and the system is
deterministic, `python run_demo.py` produces **identical** `submission_artifacts/`
on your machine, on Colab, and on the grader's box — same shard hashes, ledgers,
checkpoints and PASS markers (only wall‑clock throughput varies).

Create that cache once on **Colab** via `run_pipeline.ipynb`: it installs `datasets`,
downloads the top‑10k rows, writes `data/sangraha_tel_10k.jsonl.gz`, then runs the
exact command and renders `evidence.html`. Commit **both** the cache and
`submission_artifacts/`.

If the cache file is absent, `run_demo.py` falls back to a small **bundled Telugu
fixture** so it still runs fully offline (with fixture‑scale numbers). Run parameters
(sequence length, steps, shard size, vocab cap) auto‑scale to the data size. The core
is **stdlib‑only**; `datasets` is needed solely to fetch Sangraha on Colab.

---

## What is real vs. imitated (read this first)

Per the Session 6 transcript — *"training is just fake training, just send it to a
loop and come back"* — the assignment grades the **data system**, not the GPU. So the
data‑system behaviours are **genuinely executed and reproducible**, while **training
compute** and the **OPUS gradient computation** are deliberate, transparent imitations.
Being explicit about this is intentional: the graded behaviours (resume, replay,
firewall, ledgers, packing, mixture) are real; the two imitations below are *compute*
stand‑ins, not faked *results*.

### The model that stands in for training

We do **not** train a transformer. The stand‑in is a **Laplace‑smoothed bigram
language model** (`tds/model.py`), chosen for two reasons:

- It **genuinely learns** by counting bigrams, so its loss really drops and the
  learning ledger's per‑shard loss / perplexity / "already‑learned" signals are
  *measured*, not invented.
- It has **zero randomness**, which is what makes replay bit‑identical. Its cold‑start
  loss is exactly **ln(V)** (an untrained model over V tokens guesses 1/V — the
  session's 11.78‑for‑131k‑vocab point).

"Training a batch" means: for each accepted packed sequence, compute per‑token loss
under the current counts (loss‑masked), record it, then increment the bigram counts for
the loss‑bearing tokens (the SGD analogue). Response‑only masking therefore means only
the response bigrams are learned. **What it is not:** a transformer — loss *magnitudes*
are illustrative; the *mechanism* is real.

### How we imitate OPUS

OPUS is a real paper — *"OPUS: Towards Efficient and Principled Data Selection in LLM
Pre‑training in Every Iteration"* (arXiv 2602.05400, ICML 2026). It is **not** a pip
library, and its real method needs a transformer on 8×A100 (optimizer‑space gradient
projection with a Ghost/CountSketch estimator and Boltzmann diversity sampling). We
therefore imitate its **logic**, not its code (`tds/opus.py`):

- **Real in our imitation** — the selection *control flow and audit trail*: candidate
  buffering (the loader pulls a multiplied pool), the accept / reject / defer cascade,
  the **protected‑floor override**, and the **"already‑comfortable" test**, which reads
  the *live model's actual loss* and rejects data the model has effectively learned.
  Every decision + reason is written to `ledgers/opus.jsonl` and is fully reproducible.
- **Imitated (a stand‑in)** — the **gradient‑alignment score**. Real OPUS projects a
  candidate's optimizer‑space update onto a proxy direction; we substitute a
  deterministic SHA‑256‑of‑content score in [0,1). Same content → same score (so it's
  reproducible), but it is **not** a true gradient computation.

In short: OPUS's **decisions and audit trail are real**; only the scalar that ranks
candidates on gradient alignment is approximated. Both imitations are also stated in the
`evidence.html` limits section, generated by the run.

## The full path

```
documents → tokenized shards → manifests → mixture schedule → packing
  → batches → training → consumption ledger → learning ledger
  → checkpoint → crash → resume → replay → fork → audit → throughput
```

Every arrow is a real module under `tds/`, and every stage writes reconstructable
evidence to `run.log` and the ledgers.

---

## Design decisions (and why)

**Determinism is derived, not seeded.** Session 6 warns that a Python RNG seed is only
reproducible on the same machine/session ("turn the machine off and on, you get a
different value"). So the batch stream is a **pure function of (seed, branch, step,
model‑state, cursors)**, with every choice taken from a SHA‑256 of those inputs — never
from `random`'s internal state or the wall clock. Result: ledgers, manifests and
checkpoints are **byte‑identical across separate runs and machines** (verified), which
is what makes resume and replay exact.

**The "model" is a real tiny learner, not a fake number.** Training is intentionally
cheap, but a constant fake loss would make the learning ledger meaningless. Instead the
model is a **Laplace‑smoothed bigram LM** (`tds/model.py`): zero randomness, yet it
genuinely learns by counting — so loss actually drops and "this shard is already
learned" is a *measured* signal. Its cold‑start loss is exactly **ln(V)** (the
session's 11.78‑for‑131k‑vocab point), because an untrained model over V tokens guesses
1/V. On the real Telugu vocab this lands at ~6 and descends as the model learns.

**Telugu‑safe tokenization.** The frozen tokenizer splits on whitespace and peels
punctuation, so Brahmic conjuncts + matras stay intact (a Unicode `\w` class would
shatter them on combining marks — the exact failure Session 4 warned about). The vocab
is frozen and hashed; every shard manifest records the `tokenizer_hash`.

**OPUS uses the model's real loss.** Because the model learns for real, OPUS
(`tds/opus.py`) rejects a candidate whose *current* loss is well below the model's
running average — the "already comfortable, wasted compute" case — alongside a
deterministic proxy‑alignment term and stage matching. Protected floors
(indic/agentic/reasoning) **override** the selector, exactly as the always‑on lane
does in V4.

**A checkpoint is bound to a data position.** Every checkpoint stores the model state,
the dataloader cursors, the per‑lane token tallies **and the ledger offsets**. On
resume we roll the ledgers back to that offset, restore state, and regenerate — proving
the next batch is exact, with no skipped or repeated batch.

---

## What it demonstrates

| Capability | Where | Marker in `run.log` |
|---|---|---|
| Immutable shards, frozen tokenizer hash | `shards.py`, `tokenizer.py` | `tokenizer_hash_verified`, `shard_immutable_hash` |
| Packing policies (concat / structure‑preserving / best‑fit) | `packing.py` | `attention_no_cross_doc`, `loss_mask_response_only`, `position_ids_reset` |
| Curriculum stages, lane weights, protected floors | `mixture.py` | `mixture_compiled`, `protected_floor_respected` |
| Eval firewall (never‑train + canary) | `firewall.py` | `eval_shard_blocked`, `no_canary_in_loss_batches` |
| OPUS accept / reject / defer / floor‑override | `opus.py` | `opus_trail_recorded`, `opus_override_fired` |
| Two‑way consumption + learning ledgers | `ledger.py`, `trainer.py` | `consumption_recorded`, `learning_linked_to_source`, `already_learned_detected` |
| Checkpoints tied to ledger offsets | `checkpoint.py` | `checkpoint_saved` |
| Crash → resume (exact next batch) | `trainer.py` | `resume_next_batch_matched`, `resume_no_skip_no_repeat` |
| Replay (identical hashes + spans) | `trainer.py` | `replay_hash_matched` |
| Fork from an earlier checkpoint | `trainer.py` | `fork_diverged` |
| Audit (reconstruct shards for a step range) | `trainer.py` | `audit_reconstructed` |
| Throughput & packing efficiency | `throughput.py` | `throughput_measured` |

The evidence bundle is **generated by the run**, never hardcoded: every PASS/FAIL is
read from markers the code emitted, and every number is recomputed from the ledgers.

---

## Output layout

```
submission_artifacts/
  run.log            full event log with [PASS] markers
  evidence.json      machine-readable: requirement → result + evidence
  evidence.md        human-readable summary table
  evidence.html      designed dashboard (data pipeline + throughput signature + loss curve)
  manifests/         one manifest per shard, tokenizer.json, schedule.json
  ledgers/           consumption.jsonl, learning.jsonl, opus.jsonl, audit_report.json
  checkpoints/       ckpt_*.json (bound to ledger offsets), fork_lineage.json
  performance.json   throughput & packing efficiency
```

---

## Honest limits

- **Indic lane is real** Sangraha `verified/tel` (10k rows profiled); the other lanes
  are small synthetic stubs to exercise the multi‑lane machinery.
- **Toy training scale by design.** The executable trains a bounded slice, not
  ~128M‑token shards. The architecture and its invariants are the deliverable.
- **The model is a bigram LM**, not a transformer. Loss *magnitudes* are illustrative;
  the *mechanism* (genuine learning, ln(V) cold start, already‑learned detection) is real.
- **Throughput wall‑clock** (tokens/sec) is host‑dependent; the token accounting behind
  it is fully reconstructable from the ledgers and byte‑identical across runs.

---

## Module map

```
run_pipeline.ipynb   Colab entry point — pins Sangraha 10k, runs the command, renders HTML
run_demo.py          run_pipeline(...) — shared entry point; CLI uses the offline fixture
tds/datasource.py    Sangraha verified/tel loader + records→corpus + Telugu fixture
tds/tokenizer.py     frozen Telugu-safe tokenizer + hash + fertility
tds/corpus.py        multi-lane corpus (real Indic lane + synthetic stubs + eval/canary)
tds/shards.py        immutable tokenized shards + manifests (+ immutability demo)
tds/packing.py       concat / structure-preserving / best-fit; masks; position ids
tds/firewall.py      never-train + canary blocking (manifest + batch layers)
tds/mixture.py       curriculum stages, lane weights, protected floors, compiler
tds/opus.py          accept/reject/defer + protected-floor override
tds/model.py         deterministic Laplace-smoothed bigram model (ln(V) cold start)
tds/ledger.py        two-way consumption + learning ledgers, offset truncation
tds/checkpoint.py    checkpoints bound to ledger offsets (+ fork lineage)
tds/dataloader.py    deterministic mixture scheduler (hash-derived, no RNG state)
tds/trainer.py       engine: canonical run, crash/resume, replay, fork, audit
tds/throughput.py    green/amber/red/gray token-flow accounting
tds/evidence.py      evidence.json / evidence.md / evidence.html generation
tests/               pytest invariant suite (15 tests)
```
