import json
import sys
import io
from pathlib import Path
from collections import defaultdict
from itertools import product

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.eval import load_queries, split_queries, evaluate

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def tune():
    queries = load_queries()
    train_set, _ = split_queries(queries, train_ratio=0.6)

    print(f"Tuning on train set ({len(train_set)} queries)...")
    print("Testing threshold combinations via cross-validation...\n")

    from app.router import get_router
    router = get_router()

    t_high_candidates = [0.35, 0.38, 0.40, 0.42, 0.45, 0.48]
    t_low_candidates = [0.10, 0.12, 0.15, 0.18]
    margin_candidates = [0.05, 0.08, 0.10, 0.12]

    best_score = 0
    best_params = None

    for t_high, t_low, margin in product(t_high_candidates, t_low_candidates, margin_candidates):
        if t_high <= t_low:
            continue

        router.update_thresholds(t_high, t_low, margin)

        results, _ = evaluate(train_set, label=f"t_high={t_high} t_low={t_low} margin={margin}")

        score = results["routing_accuracy"] * 0.5 + results["resolution_rate"] * 0.3 + results.get("sensitive_recall", 0) * 0.2

        if score > best_score:
            best_score = score
            best_params = {"t_high": t_high, "t_low": t_low, "margin": margin}
            print(f"\n  ★ New best: t_high={t_high}, t_low={t_low}, margin={margin} → score={score:.1f}")
            print(f"    routing={results['routing_accuracy']:.1f}%, resolution={results['resolution_rate']:.1f}%\n")

    print(f"\n{'=' * 60}")
    print(f"Best parameters: {best_params}")
    print(f"Best composite score: {best_score:.1f}")
    print(f"{'=' * 60}")

    if best_params:
        router.update_thresholds(best_params["t_high"], best_params["t_low"], best_params["margin"])

        import yaml
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        config["routing"]["t_high"] = best_params["t_high"]
        config["routing"]["t_low"] = best_params["t_low"]
        config["routing"]["margin"] = best_params["margin"]
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"\nConfig updated: {CONFIG_PATH}")

    return best_params


if __name__ == "__main__":
    best = tune()
    print(f"\nTuning complete. Best: {best}")
