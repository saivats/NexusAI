# NexusAI Evaluation Report

**Dataset:** held-out test set (stratified, seed=42)
**Total Queries:** 114
**Split:** Stratified shuffled 60/40, seed 42

## Summary

| Metric | Value |
|---|---|
| Routing Accuracy | 89.6% |
| Resolution Rate | 84.0% |
| Direct Answer Rate | 90.5% |
| Handoff Accuracy | 75.0% |
| Sensitive Recall | 100.0% |
| Clarify Precision | 100.0% |
| Macro-F1 | 0.901 |
| p50 Latency | 4.3ms |
| p95 Latency | 7.4ms |

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
| out-of-scope | 6 | 8 | 75.0% |
| sensitive | 4 | 4 | 100.0% |
| simple | 77 | 84 | 91.7% |

## Methodology

- Split: stratified by query type and expected domain, shuffled with seed 42 (60% train / 40% held-out).
- Calibration (logistic fit on top score and top-2 margin) and thresholds (`margin`, `cal_low`) are fitted by `tests/tune.py` on the train split only.
- `cal_answer` is fixed at 0.55; the router answers when calibrated confidence >= 0.55 and the top-2 margin >= `margin`.
- Direct Answer Rate = share of simple held-out queries routed to the correct domain with action `answer` (not `clarify`).
- These numbers come from a single run on the held-out split and were not used for tuning.
