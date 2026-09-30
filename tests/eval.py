import json
import sys
import io
from pathlib import Path
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.router import get_router
from app.retrieval import get_retriever


QUERIES_PATH = Path(__file__).resolve().parent / "queries.json"


def load_queries():
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate():
    router = get_router()
    retriever = get_retriever()
    queries = load_queries()

    routing_correct = 0
    routing_total = 0
    resolution_correct = 0
    resolution_total = 0
    handoff_correct = 0
    handoff_total = 0

    confusion = defaultdict(lambda: defaultdict(int))

    type_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    print("=" * 80)
    print("NexusAI Evaluation Report")
    print("=" * 80)
    print()

    for q in queries:
        query = q["query"]
        expected_domain = q["expected_domain"]
        expected_faq = q["expected_faq"]
        query_type = q["type"]

        if query_type == "multi-topic":
            evaluate_multi_topic(router, retriever, q, type_stats)
            continue

        if expected_domain == "none":
            route = router.route(query)
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
        predicted_domain = route["top_domain"]
        routing_total += 1

        domain_correct = predicted_domain == expected_domain
        if domain_correct:
            routing_correct += 1

        confusion[expected_domain][predicted_domain] += 1

        faq_correct = False
        if domain_correct and route["action"] == "answer":
            result = retriever.retrieve(query, predicted_domain)
            if result and result["faq_id"] == expected_faq:
                faq_correct = True

        resolution_total += 1
        if faq_correct:
            resolution_correct += 1

        status = "✓" if domain_correct else "✗"
        faq_status = "✓" if faq_correct else "✗"
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

    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)

    routing_acc = routing_correct / routing_total * 100 if routing_total else 0
    resolution_rate = resolution_correct / resolution_total * 100 if resolution_total else 0
    handoff_rate = handoff_correct / handoff_total * 100 if handoff_total else 0

    print(f"  Routing Accuracy:       {routing_correct}/{routing_total} = {routing_acc:.1f}%")
    print(f"  Resolution Rate:        {resolution_correct}/{resolution_total} = {resolution_rate:.1f}%")
    print(f"  Handoff Accuracy:       {handoff_correct}/{handoff_total} = {handoff_rate:.1f}%")
    print()

    print("Per-Type Breakdown:")
    for t, stats in sorted(type_stats.items()):
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] else 0
        print(f"  {t:15s}: {stats['correct']}/{stats['total']} = {acc:.1f}%")
    print()

    domains = ["it", "hr", "finance", "facilities", "none"]
    print("Confusion Matrix (rows=expected, cols=predicted):")
    header = f"{'':>12s}" + "".join(f"{d:>12s}" for d in domains)
    print(header)
    for expected in domains:
        row = f"{expected:>12s}"
        for predicted in domains:
            count = confusion[expected][predicted]
            row += f"{count:>12d}"
        print(row)

    print()
    print("=" * 80)

    if routing_acc >= 90:
        print(f"  ✓ PASS: Routing accuracy {routing_acc:.1f}% >= 90%")
    else:
        print(f"  ✗ FAIL: Routing accuracy {routing_acc:.1f}% < 90%")

    print("=" * 80)
    return routing_acc >= 90


def evaluate_multi_topic(router, retriever, q, type_stats):
    query = q["query"]
    parts = q.get("parts", [])

    import re
    split_pattern = re.compile(r"\band\b|,|\balso\b|\?")
    message_parts = split_pattern.split(query)
    cleaned_parts = [p.strip() for p in message_parts if p.strip() and len(p.strip()) > 3]

    all_correct = True
    part_results = []

    for expected_part in parts:
        exp_domain = expected_part["expected_domain"]
        exp_faq = expected_part["expected_faq"]
        found = False

        for text_part in cleaned_parts:
            route = router.route(text_part)
            if route["top_domain"] == exp_domain:
                result = retriever.retrieve(text_part, exp_domain)
                faq_match = result and result["faq_id"] == exp_faq
                part_results.append((exp_domain, True, faq_match))
                found = True
                break

        if not found:
            part_results.append((exp_domain, False, False))
            all_correct = False

    status = "✓" if all_correct else "✗"
    print(f"  {status} [multi-topic ] \"{query[:50]}\"")
    if not all_correct:
        for domain, routed, faq_ok in part_results:
            if not routed:
                print(f"    Missing domain: {domain}")

    type_stats["multi-topic"]["total"] += 1
    if all_correct:
        type_stats["multi-topic"]["correct"] += 1


if __name__ == "__main__":
    success = evaluate()
    sys.exit(0 if success else 1)
