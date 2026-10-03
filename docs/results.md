# NexusAI Evaluation Report

**Dataset:** held-out test set (stratified, seed=42)
**Total Queries:** 114
**Split:** Stratified shuffled 60/40, seed 42

## Summary

| Metric | Value |
|---|---|
| Routing Accuracy | 90.6% |
| Resolution Rate | 81.9% |
| Direct Answer Rate (simple, n=84) | 86.9% |
| Out-of-Scope Handoff Recall (n=8) | 87.5% |
| Wrong-Domain Answer Rate (in-domain) | 3.2% |
| False Handoff Rate (in-domain) | 2.1% |
| Sensitive Recall | 100.0% |
| Clarify Precision | 62.5% |
| Macro-F1 | 0.901 |
| p50 Latency | 8.8ms |
| p95 Latency | 13.9ms |

## Per-Domain F1 (with support counts)

| Domain | F1 | Support |
|---|---|---|
| it | 0.914 | 16 |
| hr | 0.846 | 15 |
| finance | 0.938 | 16 |
| facilities | 0.97 | 16 |
| academics | 0.909 | 16 |
| admissions | 0.828 | 15 |

## Per-Type Breakdown

| Type | Correct | Total | Accuracy |
|---|---|---|---|
| ambiguous | 8 | 10 | 80.0% |
| multi-topic | 7 | 8 | 87.5% |
| out-of-scope | 7 | 8 | 87.5% |
| sensitive | 4 | 4 | 100.0% |
| simple | 77 | 84 | 91.7% |

## Out-of-Domain Trade-off

Held-out operating point compared with the Phase 0 router (same split, same queries):

| Router | Out-of-Scope Handoff Recall | Direct Answer Rate |
|---|---|---|
| Phase 0 (no out-of-domain gate) | 75.0% | 90.5% |
| Phase 1A (out-of-domain gate) | 87.5% | 86.9% |

Train-split sweep of `other_threshold` (other thresholds fixed at the selected values; out-of-scope features cross-fitted over 4 folds):

| other_threshold | Handoff Recall | Direct Answer | Wrong-Domain Answer | False Handoff | Clarify |
|---|---|---|---|---|---|
| 0.3 | 100.0% | 92.8% | 0.7% | 3.7% | 3.0% |
| 0.4 | 75.0% | 93.6% | 0.7% | 1.5% | 4.4% |
| 0.5 | 75.0% | 94.4% | 0.7% | 0.0% | 5.2% |
| 0.6 | 75.0% | 94.4% | 0.7% | 0.0% | 5.2% |
| 0.7 | 75.0% | 94.4% | 0.7% | 0.0% | 5.2% |
| 0.8 | 75.0% | 94.4% | 0.7% | 0.0% | 5.2% |
| 1.01 | 75.0% | 94.4% | 0.7% | 0.0% | 5.2% |

Selected on train: `other_threshold=0.3`, `sim_floor=0.02`, `answer_floor=0.0`, `margin=0.04`, `pair_margin=0.08`, `cal_low=0.0`

## Fresh Set

Not run: `tests/fresh_eval.json` does not exist.

## Methodology

- Split: stratified by query type and expected domain, shuffled with seed 42 (60% train / 40% held-out).
- `tests/tune.py` fits the calibration and every decision threshold on the train split only.
- Out-of-domain gate: a classifier `other` class trained on generic non-campus text plus train-split out-of-scope queries, a word-level similarity floor (`sim_floor`), and an `answer_floor` on the raw score.
- Overlapping pairs (finance/facilities, finance/academics, hr/it, admissions/academics) need `pair_margin` instead of `margin` to answer.
- `cal_answer` is fixed at 0.55. Direct Answer Rate = simple held-out queries routed to the correct domain with action `answer`.
- These numbers come from a single run on the held-out split and were not used for tuning.
