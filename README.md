# Behavioural Anomaly Detection for Cybersecurity

Detects intrusion and compromised-credential activity in access logs by modelling what
"normal" looks like **per entity**, classifies the anomaly type, and gives a SOC analyst an
explainable risk score. All data is synthetic and generated here; labels are used for
evaluation only and are physically separated from anything the detector can read.

**Read `REPORT.md` for the full write-up and `RESULTS.md` for the chronological log of what
was measured, what broke, and what is still wrong.**

---

## Quick start

```bash
pip install -r requirements.txt

# Generate the two datasets. ~20s each. Seed 3 is the eval seed: it carries all six
# attack types (seed 2 structurally lacks impossible_travel by design).
python src/generator.py --seed 1 --entities 200 --days 60 --delta 0.5
python src/generator.py --seed 3 --entities 200 --days 60 --delta 0.5

# The leakage gate comes FIRST. Nothing downstream means anything until it passes.
python src/audit.py --tag seed1_delta05
python src/audit.py --tag seed3_delta05

python tests/test_evaluate.py        # 13 hand-computed metric tests
python src/pipeline.py --explain 3   # headline metrics + sample alerts

streamlit run dashboard/app.py       # the analyst console
```

A fresh clone works with nothing but `streamlit run`. The dashboard prefers the precomputed
bundles in `demo/` — real full-scale results, ten selectable fit/eval pairs (the δ sweep and
the five holdout seeds), loaded instantly and using almost no memory. That is also what the
deployed app serves, because a cloud container cannot hold the 173 MB of datasets and
re-scoring a container-sized pair would be a different, easier experiment rather than a
smaller one. With `data/` populated it re-scores live instead, which is the fully
interactive path.

## What each module does

| file | role |
|---|---|
| `src/generator.py` | Synthetic log generator. One emission path for benign and attack traffic. |
| `src/world.py` | The simulated org: roles, resources, geography, IP pools, devices. |
| `src/audit.py` | **Adversarial leakage gate.** Must pass before any result is meaningful. |
| `src/features.py` | L0 rules + L1 per-entity baselines, streaming, bounded state. |
| `src/sequence.py` | L2 n-gram surprisal over a typed token stream. |
| `src/detector.py` | L4 fusion, calibration, thresholds, L5 type rules, L6 explanation. |
| `src/evaluate.py` | Alert grouping, incident matching, imbalance-aware metrics. |
| `src/pipeline.py` | End-to-end run. |
| `src/classify.py` | Anomaly-type confusion matrix. |
| `src/ablation.py` | Layer ablation — does each layer earn its keep? |
| `src/drift.py` | Drift adaptation and baseline-poisoning experiment. |
| `src/stream.py` | `score_event()` + latency benchmark. |
| `src/experiments.py` | Difficulty sweep + frozen-config holdout protocol. |
| `src/export_demo.py` | Score one fit/eval pair at full scale, write a `demo/` bundle. |
| `src/export_all.py` | Every bundle the dashboard offers, and why only those pairs. |
| `dashboard/app.py` | Analyst console (6 tabs). |

## Two things to know before reading the numbers

**`data/` is gitignored.** It is 173 MB and fully reproducible — `--seed N` regenerates any
dataset byte-identically. `figures/*.csv` and `*.png` **are** committed, because they are
the record of what was measured.

**If you change the generator, delete `data/` and re-run everything.** Three separate leaks
in this project were introduced by fixes that were individually correct; the audit caught
all three, but only because stale datasets were deleted rather than reused.

## Headline result

Fitted on seed 1, evaluated on five held-out seeds under a hash-frozen config:

| metric | dev | holdout (5 seeds) |
|---|---|---|
| Incident recall | 0.817 | **0.837** |
| PR-AUC | 0.643 | 0.613 |
| R-Precision | 0.615 | 0.570 |

PR-AUC 0.613 against a random baseline of ~0.02. The pre-registered expectation, written
before any number existed, was 0.5–0.8 — a score near 0.99 would have meant the benchmark
was leaking, not that the detector was good.
