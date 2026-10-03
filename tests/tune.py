import contextlib
import io
import sys
from itertools import product
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from sklearn.linear_model import LogisticRegression

from tests.eval import load_queries, split_queries, evaluate
from app.router import get_router

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
FIXED_CAL_ANSWER = 0.55
MARGIN_CANDIDATES = [0.02, 0.04, 0.06, 0.08, 0.10]
CAL_LOW_CANDIDATES = [0.20, 0.30, 0.35, 0.40, 0.45, 0.48, 0.50, 0.52]


def _calibration_examples(router, train_set):
    features, labels = [], []
    for q in train_set:
        if q["type"] in ("multi-topic", "sensitive"):
            continue
        scores = router.score_domains(q["query"])
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_domain, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        features.append([top_score, top_score - second_score])
        labels.append(1 if top_domain == q["expected_domain"] else 0)
    return features, labels


def fit_calibration(router, train_set):
    features, labels = _calibration_examples(router, train_set)
    if len(set(labels)) < 2:
        raise ValueError("Calibration needs both correct and incorrect routing examples in the train split")
    model = LogisticRegression(C=10.0, max_iter=1000)
    model.fit(features, labels)
    coef_score, coef_margin = (float(c) for c in model.coef_[0])
    intercept = float(model.intercept_[0])
    print(f"Calibration fitted on {len(labels)} train examples ({sum(labels)} correct)")
    print(f"  coef_score={coef_score:.3f} coef_margin={coef_margin:.3f} intercept={intercept:.3f}")
    return {"coef_score": round(coef_score, 4), "coef_margin": round(coef_margin, 4), "intercept": round(intercept, 4)}


def _composite_score(results):
    return (
        results["routing_accuracy"] * 0.35
        + results["resolution_rate"] * 0.2
        + results.get("direct_answer_rate", 0) * 0.25
        + results.get("handoff_accuracy", 0) * 0.1
        + results.get("macro_f1", 0) * 100 * 0.1
    )


def search_thresholds(router, train_set):
    best_score, best_params, best_results = -1.0, None, None
    for margin, cal_low in product(MARGIN_CANDIDATES, CAL_LOW_CANDIDATES):
        router.update_thresholds(margin=margin, cal_answer=FIXED_CAL_ANSWER, cal_low=cal_low)
        with contextlib.redirect_stdout(io.StringIO()):
            results, _ = evaluate(train_set, label="train")
        score = _composite_score(results)
        print(f"  margin={margin:.2f} cal_low={cal_low:.2f} -> score={score:.2f} "
              f"routing={results['routing_accuracy']}% direct={results['direct_answer_rate']}% "
              f"handoff={results['handoff_accuracy']}%")
        if score > best_score:
            best_score, best_params, best_results = score, {"margin": margin, "cal_low": cal_low}, results
    return best_params, best_results


def save_config(calibration, thresholds):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    routing = config.setdefault("routing", {})
    routing["margin"] = thresholds["margin"]
    routing["cal_answer"] = FIXED_CAL_ANSWER
    routing["cal_low"] = thresholds["cal_low"]
    routing["calibration"] = calibration
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"Config updated: {CONFIG_PATH}")


def tune():
    queries = load_queries()
    train_set, _ = split_queries(queries, train_ratio=0.6, seed=42)
    print(f"Tuning on train split only ({len(train_set)} queries)\n")

    router = get_router()
    calibration = fit_calibration(router, train_set)
    router.set_calibration(**calibration)

    print("\nThreshold search:")
    thresholds, results = search_thresholds(router, train_set)
    router.update_thresholds(cal_answer=FIXED_CAL_ANSWER, **thresholds)

    print(f"\nBest thresholds: {thresholds}")
    print(f"Train metrics: routing={results['routing_accuracy']}% resolution={results['resolution_rate']}% "
          f"direct={results['direct_answer_rate']}% handoff={results['handoff_accuracy']}%")
    save_config(calibration, thresholds)
    return calibration, thresholds


if __name__ == "__main__":
    tune()
