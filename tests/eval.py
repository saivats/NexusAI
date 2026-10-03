import json
import sys
import io
import re
import time
import random
from pathlib import Path
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.router import get_router
from app.retrieval import get_retriever

QUERIES_PATH = Path(__file__).resolve().parent / "queries.json"
RESULTS_PATH = Path(__file__).resolve().parent / "results.json"
REPORT_PATH = Path(__file__).resolve().parent.parent / "docs" / "results.md"
SPLIT_PATTERN = re.compile(
    r"\band\s+also\b|\balso\b|\bplus\b|\band\b|[;]|[,]\s*(?:and\b|also\b|plus\b|\d+[.)])|"
    r"[,](?=\s+(?:how|what|when|where|why|can|do|is|are|i\s))|\?\s*|(?:^|\n)\s*\d+[.)]\s*",
    re.IGNORECASE,
)

DOMAINS = ["it", "hr", "finance", "facilities", "academics", "admissions", "none", "sensitive"]
MIN_HELD_OUT_SIMPLE_PER_DOMAIN = 6
MIN_HELD_OUT_PER_DOMAIN_WARNING = 5


def load_queries():
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _stratification_key(q):
    qtype = q["type"]
    if qtype in ("multi-topic",):
        return f"type:{qtype}"
    domain = q.get("expected_domain", "unknown")
    return f"{qtype}:{domain}"


def split_queries(queries, train_ratio=0.6, seed=42):
    rng = random.Random(seed)

    by_stratum = defaultdict(list)
    for q in queries:
        by_stratum[_stratification_key(q)].append(q)

    train_set, test_set = [], []
    for key in sorted(by_stratum.keys()):
        items = list(by_stratum[key])
        rng.shuffle(items)
        split_idx = max(1, int(len(items) * train_ratio))
        if len(items) - split_idx < 1:
            split_idx = max(0, len(items) - 1)
        train_set.extend(items[:split_idx])
        test_set.extend(items[split_idx:])

    return train_set, test_set


def _check_held_out_coverage(test_set):
    simple_per_domain = defaultdict(int)
    all_per_domain = defaultdict(int)
    for q in test_set:
        if q["type"] == "multi-topic":
            continue
        domain = q.get("expected_domain", "unknown")
        all_per_domain[domain] += 1
        if q["type"] == "simple":
            simple_per_domain[domain] += 1

    warnings = []
    for d in DOMAINS:
        if d in ("none", "sensitive"):
            continue
        count = simple_per_domain.get(d, 0)
        if count < MIN_HELD_OUT_SIMPLE_PER_DOMAIN:
            warnings.append(f"  ⚠ Domain '{d}' has only {count} simple held-out queries (minimum: {MIN_HELD_OUT_SIMPLE_PER_DOMAIN})")
        total = all_per_domain.get(d, 0)
        if total < MIN_HELD_OUT_PER_DOMAIN_WARNING:
            warnings.append(f"  ⚠ Domain '{d}' has only {total} total held-out queries (recommended minimum: {MIN_HELD_OUT_PER_DOMAIN_WARNING})")

    return simple_per_domain, all_per_domain, warnings


def evaluate(queries, label=""):
    router = get_router()
    retriever = get_retriever()

    routing_correct = 0
    routing_total = 0
    resolution_correct = 0
    resolution_total = 0
    handoff_correct = 0
    handoff_total = 0
    sensitive_correct = 0
    sensitive_total = 0
    clarify_correct = 0
    clarify_total = 0

    simple_answered_directly = 0
    simple_total = 0

    confusion = defaultdict(lambda: defaultdict(int))
    type_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    per_domain_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "support": 0})
    latencies = []

    print(f"\n{'=' * 80}")
    print(f"NexusAI Evaluation Report{' — ' + label if label else ''}")
    print(f"{'=' * 80}\n")

    for q in queries:
        query = q["query"]
        expected_domain = q["expected_domain"]
        expected_faq = q["expected_faq"]
        query_type = q["type"]

        if query_type == "multi-topic":
            _evaluate_multi_topic(router, retriever, q, type_stats, latencies)
            continue

        start = time.perf_counter()

        if query_type == "sensitive":
            sensitive_total += 1
            routing_total += 1
            is_sensitive = router.is_sensitive(query)
            latencies.append((time.perf_counter() - start) * 1000)

            if is_sensitive:
                sensitive_correct += 1
                routing_correct += 1
                confusion["sensitive"]["sensitive"] += 1
            else:
                route = router.route(query)
                confusion["sensitive"][route["top_domain"]] += 1

            status = "✓" if is_sensitive else "✗"
            print(f"  {status} [{query_type:12s}] \"{query[:50]}\"")
            if not is_sensitive:
                print(f"    Expected: sensitive, Got: routed normally")

            type_stats[query_type]["total"] += 1
            if is_sensitive:
                type_stats[query_type]["correct"] += 1
            continue

        if expected_domain == "none":
            route = router.route(query)
            latencies.append((time.perf_counter() - start) * 1000)
            handoff_total += 1
            routing_total += 1
            is_handoff = route["action"] == "handoff"
            if is_handoff:
                handoff_correct += 1
                routing_correct += 1
                confusion["none"]["none"] += 1
            else:
                confusion["none"][route["top_domain"]] += 1

            status = "✓" if is_handoff else "✗"
            print(f"  {status} [{query_type:12s}] \"{query[:50]}\"")
            if not is_handoff:
                print(f"    Expected: handoff, Got: {route['action']} → {route['top_domain']} ({route['confidence']:.3f})")

            type_stats[query_type]["total"] += 1
            if is_handoff:
                type_stats[query_type]["correct"] += 1
            continue

        route = router.route(query)
        latencies.append((time.perf_counter() - start) * 1000)
        predicted_domain = route["top_domain"]
        routing_total += 1

        if expected_domain not in ("none", "sensitive"):
            per_domain_stats[expected_domain]["support"] += 1

        domain_correct = predicted_domain == expected_domain
        if domain_correct:
            routing_correct += 1
            per_domain_stats[expected_domain]["tp"] += 1
        else:
            per_domain_stats[expected_domain]["fn"] += 1
            per_domain_stats[predicted_domain]["fp"] += 1

        confusion[expected_domain][predicted_domain] += 1

        if route["action"] == "clarify":
            clarify_total += 1
            if domain_correct or route.get("second_domain") == expected_domain:
                clarify_correct += 1

        if query_type == "simple":
            simple_total += 1
            if domain_correct and route["action"] == "answer":
                simple_answered_directly += 1

        faq_correct = False
        if domain_correct and route["action"] in ("answer", "clarify"):
            result = retriever.retrieve(query, predicted_domain)
            if result and result["faq_id"] == expected_faq:
                faq_correct = True

        resolution_total += 1
        if faq_correct:
            resolution_correct += 1

        status = "✓" if domain_correct else "✗"
        print(f"  {status} [{query_type:12s}] \"{query[:50]}\"")
        if not domain_correct:
            print(f"    Route: expected={expected_domain}, got={predicted_domain} ({route['confidence']:.3f})")
        elif not faq_correct:
            result = retriever.retrieve(query, predicted_domain)
            got_faq = result["faq_id"] if result else "N/A"
            print(f"    FAQ: expected={expected_faq}, got={got_faq} (action={route['action']})")

        type_stats[query_type]["total"] += 1
        if domain_correct:
            type_stats[query_type]["correct"] += 1

    routing_acc = routing_correct / routing_total * 100 if routing_total else 0
    resolution_rate = resolution_correct / resolution_total * 100 if resolution_total else 0
    handoff_acc = handoff_correct / handoff_total * 100 if handoff_total else 0
    sensitive_recall = sensitive_correct / sensitive_total * 100 if sensitive_total else 0
    clarify_precision = clarify_correct / clarify_total * 100 if clarify_total else 0
    direct_answer_rate = simple_answered_directly / simple_total * 100 if simple_total else 0

    p50 = sorted(latencies)[len(latencies) // 2] if latencies else 0
    p95 = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0

    domain_f1s = {}
    domain_supports = {}
    for d in ["it", "hr", "finance", "facilities", "academics", "admissions"]:
        s = per_domain_stats[d]
        precision = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) > 0 else 0
        recall = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        domain_f1s[d] = round(f1, 3)
        domain_supports[d] = s["support"]

    macro_f1 = sum(domain_f1s.values()) / len(domain_f1s) if domain_f1s else 0

    print(f"\n{'=' * 80}")
    print("Summary")
    print(f"{'=' * 80}")
    print(f"  Routing Accuracy:       {routing_correct}/{routing_total} = {routing_acc:.1f}%")
    print(f"  Resolution Rate:        {resolution_correct}/{resolution_total} = {resolution_rate:.1f}%")
    print(f"  Direct Answer Rate:     {simple_answered_directly}/{simple_total} = {direct_answer_rate:.1f}% (simple queries answered, not clarified)")
    print(f"  Handoff Accuracy:       {handoff_correct}/{handoff_total} = {handoff_acc:.1f}%")
    print(f"  Sensitive Recall:       {sensitive_correct}/{sensitive_total} = {sensitive_recall:.1f}%")
    print(f"  Clarify Precision:      {clarify_correct}/{clarify_total} = {clarify_precision:.1f}%")
    print(f"  Macro-F1:               {macro_f1:.3f}")
    print(f"  p50 Latency:            {p50:.1f}ms")
    print(f"  p95 Latency:            {p95:.1f}ms")
    print()

    print("Per-Domain F1 (with support counts):")
    for d, f1 in domain_f1s.items():
        support = domain_supports.get(d, 0)
        warning = " ⚠ LOW SUPPORT" if support < MIN_HELD_OUT_PER_DOMAIN_WARNING else ""
        print(f"  {d:15s}: F1={f1:.3f}  support={support}{warning}")
    print()

    print("Per-Type Breakdown:")
    for t, stats in sorted(type_stats.items()):
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] else 0
        print(f"  {t:15s}: {stats['correct']}/{stats['total']} = {acc:.1f}%")
    print()

    all_domains = ["it", "hr", "finance", "facilities", "academics", "admissions", "none", "sensitive"]
    present_domains = [d for d in all_domains if any(confusion[d][p] > 0 for p in all_domains) or any(confusion[e][d] > 0 for e in all_domains)]
    print("Confusion Matrix (rows=expected, cols=predicted):")
    header = f"{'':>12s}" + "".join(f"{d:>12s}" for d in present_domains)
    print(header)
    for expected in present_domains:
        row = f"{expected:>12s}"
        for predicted in present_domains:
            count = confusion[expected][predicted]
            row += f"{count:>12d}"
        print(row)

    print(f"\n{'=' * 80}")
    if routing_acc >= 90:
        print(f"  ✓ PASS: Routing accuracy {routing_acc:.1f}% >= 90%")
    else:
        print(f"  ✗ FAIL: Routing accuracy {routing_acc:.1f}% < 90%")

    if resolution_rate >= 85:
        print(f"  ✓ PASS: Resolution rate {resolution_rate:.1f}% >= 85%")
    else:
        print(f"  ✗ FAIL: Resolution rate {resolution_rate:.1f}% < 85%")

    if direct_answer_rate >= 70:
        print(f"  ✓ PASS: Direct answer rate {direct_answer_rate:.1f}% >= 70%")
    else:
        print(f"  ⚠ LOW: Direct answer rate {direct_answer_rate:.1f}% < 70%")
    print(f"{'=' * 80}")

    results = {
        "label": label,
        "routing_accuracy": round(routing_acc, 1),
        "resolution_rate": round(resolution_rate, 1),
        "direct_answer_rate": round(direct_answer_rate, 1),
        "handoff_accuracy": round(handoff_acc, 1),
        "sensitive_recall": round(sensitive_recall, 1),
        "clarify_precision": round(clarify_precision, 1),
        "macro_f1": round(macro_f1, 3),
        "p50_latency_ms": round(p50, 1),
        "p95_latency_ms": round(p95, 1),
        "per_domain_f1": domain_f1s,
        "per_domain_support": domain_supports,
        "per_type": {t: {"correct": s["correct"], "total": s["total"]} for t, s in type_stats.items()},
        "confusion": {e: dict(p) for e, p in confusion.items()},
        "total_queries": len(queries),
    }

    return results, routing_acc >= 90 and resolution_rate >= 85


def _evaluate_multi_topic(router, retriever, q, type_stats, latencies):
    query = q["query"]
    parts = q.get("parts", [])

    start = time.perf_counter()
    message_parts = SPLIT_PATTERN.split(query)
    cleaned_parts = [p.strip() for p in message_parts if p and p.strip() and len(p.strip()) > 3]
    latencies.append((time.perf_counter() - start) * 1000)

    all_correct = True
    part_results = []

    for expected_part in parts:
        exp_domain = expected_part["expected_domain"]
        found = False

        for text_part in cleaned_parts:
            route = router.route(text_part)
            if route["top_domain"] == exp_domain:
                found = True
                break

        if not found:
            all_correct = False

        part_results.append((exp_domain, found))

    status = "✓" if all_correct else "✗"
    print(f"  {status} [multi-topic ] \"{query[:50]}\"")
    if not all_correct:
        for domain, routed in part_results:
            if not routed:
                print(f"    Missing domain: {domain}")

    type_stats["multi-topic"]["total"] += 1
    if all_correct:
        type_stats["multi-topic"]["correct"] += 1


def write_results(results):
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")


def write_report(results):
    lines = [
        "# NexusAI Evaluation Report\n",
        f"**Dataset:** {results.get('label', 'held-out test set')}",
        f"**Total Queries:** {results['total_queries']}",
        f"**Split:** Stratified shuffled 60/40, seed 42\n",
        "## Summary\n",
        "| Metric | Value |",
        "|---|---|",
        f"| Routing Accuracy | {results['routing_accuracy']}% |",
        f"| Resolution Rate | {results['resolution_rate']}% |",
        f"| Direct Answer Rate | {results['direct_answer_rate']}% |",
        f"| Handoff Accuracy | {results['handoff_accuracy']}% |",
        f"| Sensitive Recall | {results['sensitive_recall']}% |",
        f"| Clarify Precision | {results['clarify_precision']}% |",
        f"| Macro-F1 | {results['macro_f1']} |",
        f"| p50 Latency | {results['p50_latency_ms']}ms |",
        f"| p95 Latency | {results['p95_latency_ms']}ms |",
        "",
        "## Per-Domain F1 (with support counts)\n",
        "| Domain | F1 | Support |",
        "|---|---|---|",
    ]
    for d, f1 in results.get("per_domain_f1", {}).items():
        support = results.get("per_domain_support", {}).get(d, 0)
        lines.append(f"| {d} | {f1} | {support} |")

    lines.extend([
        "",
        "## Per-Type Breakdown\n",
        "| Type | Correct | Total | Accuracy |",
        "|---|---|---|---|",
    ])
    for t, s in sorted(results.get("per_type", {}).items()):
        acc = s["correct"] / s["total"] * 100 if s["total"] else 0
        lines.append(f"| {t} | {s['correct']} | {s['total']} | {acc:.1f}% |")

    lines.extend([
        "",
        "## Methodology\n",
        "- Split: stratified by query type and expected domain, shuffled with seed 42 (60% train / 40% held-out).",
        "- Calibration (logistic fit on top score and top-2 margin) and thresholds (`margin`, `cal_low`) are fitted by `tests/tune.py` on the train split only.",
        "- `cal_answer` is fixed at 0.55; the router answers when calibrated confidence >= 0.55 and the top-2 margin >= `margin`.",
        "- Direct Answer Rate = share of simple held-out queries routed to the correct domain with action `answer` (not `clarify`).",
        "- These numbers come from a single run on the held-out split and were not used for tuning.",
        "",
    ])

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report saved to {REPORT_PATH}")


if __name__ == "__main__":
    queries = load_queries()
    train_set, test_set = split_queries(queries, train_ratio=0.6, seed=42)

    simple_per_domain, all_per_domain, warnings = _check_held_out_coverage(test_set)
    print(f"Evaluating on held-out test set ({len(test_set)} queries)...")
    print(f"  Train: {len(train_set)}, Test: {len(test_set)}")
    print(f"  Held-out simple per domain: {dict(simple_per_domain)}")
    if warnings:
        print("\n  Coverage warnings:")
        for w in warnings:
            print(w)
    print()

    results, passed = evaluate(test_set, label="held-out test set (stratified, seed=42)")
    write_results(results)
    write_report(results)

    sys.exit(0 if passed else 1)
