import contextlib
import io
import json
import sys
from dataclasses import replace
from itertools import product
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from sklearn.linear_model import LogisticRegression

from tests.eval import load_queries, split_queries
from app.decision import (
    ACTION_ANSWER,
    ACTION_HANDOFF,
    DEFAULT_OVERLAP_PAIRS,
    DecisionThresholds,
    decide,
    normalize_pairs,
)
from app.router import DATA_DIR, DomainRouter, load_out_of_scope_texts

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"
TRAIN_OOS_PATH = DATA_DIR / "out_of_scope_train.json"
SUMMARY_PATH = Path(__file__).resolve().parent / "tuning_summary.json"

FIXED_CAL_ANSWER = 0.55
CROSS_FIT_FOLDS = 4
TARGET_HANDOFF_RECALL = 0.90
TARGET_DIRECT_RATE = 0.85
WRONG_ANSWER_PENALTY = 2.0

GRID = {
    "other_threshold": [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.01],
    "sim_floor": [0.0, 0.02, 0.04, 0.06, 0.08, 0.10],
    "answer_floor": [0.0, 0.10, 0.12, 0.14, 0.16],
    "margin": [0.02, 0.04, 0.06, 0.08],
    "pair_margin": [0.06, 0.08, 0.10, 0.12],
    "cal_low": [0.0, 0.20, 0.30, 0.40, 0.50],
}


def _is_in_domain(query):
    return query["type"] in ("simple", "ambiguous") and query["expected_domain"] not in ("none", "sensitive")


def _is_out_of_scope(query):
    return query["expected_domain"] == "none"


def export_train_out_of_scope(train_set):
    texts = sorted({q["query"] for q in train_set if _is_out_of_scope(q)})
    with open(TRAIN_OOS_PATH, "w", encoding="utf-8") as f:
        json.dump(texts, f, indent=2, ensure_ascii=False)
    print(f"Exported {len(texts)} train-split out-of-scope examples to {TRAIN_OOS_PATH.name}")
    return texts


def _silent_router(out_of_scope_texts):
    with contextlib.redirect_stdout(io.StringIO()):
        return DomainRouter(out_of_scope_texts=out_of_scope_texts)


def collect_features(train_set, generic_texts, train_oos_texts):
    full_router = _silent_router(generic_texts + train_oos_texts)
    records = []
    for q in train_set:
        if _is_in_domain(q):
            _, _, features = full_router.analyze(q["query"])
            records.append({"query": q, "features": features})

    oos_queries = [q for q in train_set if _is_out_of_scope(q)]
    for fold in range(CROSS_FIT_FOLDS):
        held = [q for i, q in enumerate(oos_queries) if i % CROSS_FIT_FOLDS == fold]
        held_texts = {q["query"] for q in held}
        fold_router = _silent_router(generic_texts + [t for t in train_oos_texts if t not in held_texts])
        for q in held:
            _, _, features = fold_router.analyze(q["query"])
            records.append({"query": q, "features": features})
    return full_router, records


def fit_calibration(records):
    rows = [r for r in records if _is_in_domain(r["query"])]
    features = [[r["features"].top_score, r["features"].margin] for r in rows]
    labels = [1 if r["features"].top_domain == r["query"]["expected_domain"] else 0 for r in rows]
    if len(set(labels)) < 2:
        raise ValueError("Calibration needs both correct and incorrect routing examples in the train split")
    model = LogisticRegression(C=10.0, max_iter=1000)
    model.fit(features, labels)
    coef_score, coef_margin = (float(c) for c in model.coef_[0])
    calibration = {
        "coef_score": round(coef_score, 4),
        "coef_margin": round(coef_margin, 4),
        "intercept": round(float(model.intercept_[0]), 4),
    }
    print(f"Calibration fitted on {len(labels)} in-domain train examples ({sum(labels)} routed correctly): {calibration}")
    return calibration


def recalibrate(records, router):
    for record in records:
        f = record["features"]
        record["features"] = replace(f, calibrated=router.calibrate_confidence(f.top_score, f.margin))


def simulate(records, thresholds):
    simple_total = simple_direct = 0
    in_domain_total = wrong_answers = false_handoffs = clarified = 0
    oos_total = oos_handed_off = 0
    for record in records:
        query, features = record["query"], record["features"]
        action, _ = decide(features, thresholds)
        if _is_out_of_scope(query):
            oos_total += 1
            oos_handed_off += action == ACTION_HANDOFF
            continue
        in_domain_total += 1
        correct_domain = features.top_domain == query["expected_domain"]
        if action == ACTION_ANSWER and not correct_domain:
            wrong_answers += 1
        elif action == ACTION_HANDOFF:
            false_handoffs += 1
        elif action != ACTION_ANSWER:
            clarified += 1
        if query["type"] == "simple":
            simple_total += 1
            simple_direct += action == ACTION_ANSWER and correct_domain
    return {
        "handoff_recall": oos_handed_off / oos_total if oos_total else 0.0,
        "direct_rate": simple_direct / simple_total if simple_total else 0.0,
        "wrong_answer_rate": wrong_answers / in_domain_total if in_domain_total else 0.0,
        "false_handoff_rate": false_handoffs / in_domain_total if in_domain_total else 0.0,
        "clarify_rate": clarified / in_domain_total if in_domain_total else 0.0,
    }


def _objective(metrics):
    return metrics["direct_rate"] + metrics["handoff_recall"] - WRONG_ANSWER_PENALTY * metrics["wrong_answer_rate"]


def _is_feasible(metrics):
    return metrics["handoff_recall"] >= TARGET_HANDOFF_RECALL and metrics["direct_rate"] >= TARGET_DIRECT_RATE


def _selection_key(params, metrics):
    return (
        _is_feasible(metrics),
        round(_objective(metrics), 6),
        params["pair_margin"],
        params["margin"],
        -abs(params["other_threshold"] - 0.5),
    )


def search_thresholds(records, overlap_pairs):
    names = list(GRID)
    best = None
    for values in product(*(GRID[name] for name in names)):
        params = dict(zip(names, values))
        thresholds = DecisionThresholds(cal_answer=FIXED_CAL_ANSWER, overlap_pairs=overlap_pairs, **params)
        metrics = simulate(records, thresholds)
        key = _selection_key(params, metrics)
        if best is None or key > best[0]:
            best = (key, params, metrics)
    _, params, metrics = best
    return params, metrics


def tradeoff_curve(records, best_params, overlap_pairs):
    rows = []
    for other_threshold in GRID["other_threshold"]:
        params = dict(best_params, other_threshold=other_threshold)
        metrics = simulate(records, DecisionThresholds(cal_answer=FIXED_CAL_ANSWER, overlap_pairs=overlap_pairs, **params))
        rows.append({"other_threshold": other_threshold, **{k: round(v * 100, 1) for k, v in metrics.items()}})
    return rows


def _format_metrics(metrics):
    return ", ".join(f"{name}={value * 100:.1f}%" for name, value in metrics.items())


def save_config(calibration, params):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    routing = config.setdefault("routing", {})
    for obsolete in ("t_high", "t_low"):
        routing.pop(obsolete, None)
    routing["cal_answer"] = FIXED_CAL_ANSWER
    routing.update({name: value for name, value in params.items()})
    routing["overlap_pairs"] = [f"{a}|{b}" for a, b in DEFAULT_OVERLAP_PAIRS]
    routing["calibration"] = calibration
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"Config updated: {CONFIG_PATH.name}")


def save_summary(params, metrics, curve, record_count):
    summary = {
        "split": "train (stratified, seed=42)",
        "records": record_count,
        "cross_fit_folds_for_out_of_scope": CROSS_FIT_FOLDS,
        "selected": params,
        "train_metrics": {k: round(v * 100, 1) for k, v in metrics.items()},
        "tradeoff_other_threshold": curve,
    }
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Tuning summary saved: {SUMMARY_PATH.name}")


def tune():
    train_set, _ = split_queries(load_queries(), train_ratio=0.6, seed=42)
    print(f"Tuning on train split only ({len(train_set)} queries)\n")

    train_oos_texts = export_train_out_of_scope(train_set)
    generic_texts = load_out_of_scope_texts(("out_of_scope_generic.json",))
    router, records = collect_features(train_set, generic_texts, train_oos_texts)

    calibration = fit_calibration(records)
    router.set_calibration(**calibration)
    recalibrate(records, router)

    overlap_pairs = normalize_pairs(DEFAULT_OVERLAP_PAIRS)
    params, metrics = search_thresholds(records, overlap_pairs)
    print(f"\nSelected: {params}")
    print(f"Train metrics: {_format_metrics(metrics)}  feasible={_is_feasible(metrics)}")

    curve = tradeoff_curve(records, params, overlap_pairs)
    print("\nTrade-off on train (other_threshold sweep, other params fixed):")
    print(f"  {'other_thr':>9} {'handoff':>8} {'direct':>7} {'wrong':>6} {'false_ho':>8} {'clarify':>8}")
    for row in curve:
        print(f"  {row['other_threshold']:>9.2f} {row['handoff_recall']:>7.1f}% {row['direct_rate']:>6.1f}% "
              f"{row['wrong_answer_rate']:>5.1f}% {row['false_handoff_rate']:>7.1f}% {row['clarify_rate']:>7.1f}%")

    save_config(calibration, params)
    save_summary(params, metrics, curve, len(records))
    return calibration, params


if __name__ == "__main__":
    tune()
