import sys
import re
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestNormalizer:
    def test_lowercase_and_strip(self):
        from app.normalizer import normalize_light
        assert normalize_light("  HELLO World  ") == "hello world"

    def test_typo_correction(self):
        from app.normalizer import normalize_light
        result = normalize_light("I forgot my pasword")
        assert "password" in result

    def test_wifi_typo(self):
        from app.normalizer import normalize_light
        result = normalize_light("wfii not working")
        assert "wifi" in result

    def test_punctuation_removal(self):
        from app.normalizer import normalize_light
        result = normalize_light("Hello! How are you?")
        assert "!" not in result
        assert "?" not in result

    def test_synonym_expansion(self):
        from app.normalizer import normalize_query
        result = normalize_query("wifi not working")
        assert "wireless" in result or "internet" in result or "network" in result

    def test_empty_input(self):
        from app.normalizer import normalize_light
        assert normalize_light("") == ""
        assert normalize_light("   ") == ""


class TestSplitter:
    def test_single_query(self):
        from app.main import _split_multi_topic
        parts = _split_multi_topic("How do I reset my password?")
        assert len(parts) >= 1

    def test_and_split(self):
        from app.main import _split_multi_topic
        parts = _split_multi_topic("reset password and check payslip")
        assert len(parts) == 2

    def test_also_split(self):
        from app.main import _split_multi_topic
        parts = _split_multi_topic("reset password also check payslip")
        assert len(parts) == 2

    def test_semicolon_split(self):
        from app.main import _split_multi_topic
        parts = _split_multi_topic("reset password; check payslip; pay fees")
        assert len(parts) >= 2

    def test_question_mark_split(self):
        from app.main import _split_multi_topic
        parts = _split_multi_topic("how to reset password? when is fee deadline?")
        assert len(parts) == 2

    def test_short_parts_filtered(self):
        from app.main import _split_multi_topic
        parts = _split_multi_topic("a and b")
        assert len(parts) >= 1


class TestRouter:
    def test_it_query_routes_correctly(self):
        from app.router import get_router
        router = get_router()
        result = router.route("How do I connect to campus WiFi?")
        assert result["top_domain"] == "it"

    def test_hr_query_routes_correctly(self):
        from app.router import get_router
        router = get_router()
        result = router.route("How do I apply for annual leave?")
        assert result["top_domain"] == "hr"

    def test_finance_query_routes_correctly(self):
        from app.router import get_router
        router = get_router()
        result = router.route("When is the tuition fee deadline?")
        assert result["top_domain"] == "finance"

    def test_wifi_not_working_routes_to_it(self):
        from app.router import get_router
        router = get_router()
        result = router.route("hey why is wifi not working")
        assert result["top_domain"] == "it"
        assert result["action"] != "handoff"

    def test_sensitive_detection(self):
        from app.router import get_router
        router = get_router()
        assert router.is_sensitive("I am being harassed by a senior")
        assert router.is_sensitive("mental health crisis help")
        assert router.is_sensitive("there is bullying in class")

    def test_not_sensitive_normal_query(self):
        from app.router import get_router
        router = get_router()
        assert not router.is_sensitive("How do I reset my password?")
        assert not router.is_sensitive("When is the fee deadline?")

    def test_sensitive_route_action(self):
        from app.router import get_router
        router = get_router()
        result = router.route("I am being harassed")
        assert result["action"] == "sensitive"

    def test_confidence_calibration(self):
        from app.router import get_router
        router = get_router()
        raw = 0.3
        calibrated = router.calibrate_confidence(raw)
        assert 0 <= calibrated <= 1

    def test_confidence_labels(self):
        from app.router import get_router
        router = get_router()
        assert router.confidence_label(0.8) == "High"
        assert router.confidence_label(0.5) == "Medium"
        assert router.confidence_label(0.2) == "Low"

    def test_score_domains_returns_all(self):
        from app.router import get_router
        router = get_router()
        scores = router.score_domains("test query")
        assert "it" in scores
        assert "hr" in scores
        assert "finance" in scores
        assert "facilities" in scores
        assert "academics" in scores
        assert "admissions" in scores

    def test_internet_slow_hostel_routes_to_it(self):
        from app.router import get_router
        router = get_router()
        result = router.route("internet is slow in hostel")
        assert result["top_domain"] == "it"

    def test_sick_leaves_routes_to_hr(self):
        from app.router import get_router
        router = get_router()
        result = router.route("how many sick leaves do i get")
        assert result["top_domain"] == "hr"

    def test_refund_routes_to_finance(self):
        from app.router import get_router
        router = get_router()
        result = router.route("can i get a refund")
        assert result["top_domain"] == "finance"

    def test_hall_ticket_routes_to_academics(self):
        from app.router import get_router
        router = get_router()
        result = router.route("exam hall ticket kab milega")
        assert result["top_domain"] == "academics"

    def test_calibrated_confidence_drives_action(self):
        from app.router import get_router
        router = get_router()
        result = router.route("How do I connect to campus WiFi?")
        assert result["calibrated_confidence"] >= 0.55
        assert result["action"] == "answer"

    def test_calibration_monotonic(self):
        from app.router import get_router
        router = get_router()
        assert router.calibrate_confidence(0.5, 0.2) > router.calibrate_confidence(0.2, 0.2)
        assert router.calibrate_confidence(0.4, 0.3) > router.calibrate_confidence(0.4, 0.0)


KNOWN_FAILURES = [
    ("hey why is wifi not working", "it"),
    ("internet is slow in hostel", "it"),
    ("how many sick leaves do i get", "hr"),
    ("can i get a refund", "finance"),
    ("exam hall ticket kab milega", "academics"),
]


@pytest.mark.parametrize("query,expected_domain", KNOWN_FAILURES)
def test_known_failures_answered_directly(query, expected_domain):
    from app.router import get_router
    result = get_router().route(query)
    assert result["top_domain"] == expected_domain
    assert result["action"] == "answer"


OUT_OF_DOMAIN_QUERIES = [
    "what's the weather going to be like in noida tomorrow",
    "is it raining outside right now",
    "hiii",
    "good afternoon",
    "kjhsdf poiuy mnbvc",
    "qqqq wwww eeee",
    "who is the richest person in the world",
    "recommend me a sci-fi movie to watch tonight",
]


@pytest.mark.parametrize("query", OUT_OF_DOMAIN_QUERIES)
def test_out_of_domain_hands_off(query):
    from app.router import get_router
    result = get_router().route(query)
    assert result["action"] == "handoff"
    assert result["out_of_domain"] is True


def _features(top, second, margin, top_score=0.4, calibrated=0.9):
    from app.decision import RoutingFeatures
    return RoutingFeatures(
        top_domain=top, second_domain=second, top_score=top_score, margin=margin,
        calibrated=calibrated, max_similarity=0.3, other_probability=0.01,
    )


OVERLAP_PAIRS = [
    ("finance", "facilities"),
    ("facilities", "finance"),
    ("finance", "academics"),
    ("hr", "it"),
    ("it", "hr"),
    ("admissions", "academics"),
]


@pytest.mark.parametrize("top,second", OVERLAP_PAIRS)
def test_overlapping_pair_close_call_clarifies(top, second):
    from app.decision import DecisionThresholds, decide
    thresholds = DecisionThresholds(margin=0.04, pair_margin=0.08)
    action, reason = decide(_features(top, second, margin=0.06), thresholds)
    assert action == "clarify"
    assert reason == "close_overlapping_pair"


def test_non_overlapping_pair_same_margin_answers():
    from app.decision import DecisionThresholds, decide
    thresholds = DecisionThresholds(margin=0.04, pair_margin=0.08, answer_floor=0.1)
    action, _ = decide(_features("it", "facilities", margin=0.06), thresholds)
    assert action == "answer"


def test_answer_requires_raw_score_floor_even_with_large_margin():
    from app.decision import DecisionThresholds, decide
    thresholds = DecisionThresholds(margin=0.04, answer_floor=0.2)
    action, reason = decide(_features("it", "facilities", margin=0.3, top_score=0.15), thresholds)
    assert action == "clarify"
    assert reason == "weak_raw_score"


def test_other_class_probability_forces_handoff():
    from app.decision import DecisionThresholds, RoutingFeatures, decide
    features = RoutingFeatures("it", "hr", 0.5, 0.3, 0.95, 0.3, 0.9)
    action, reason = decide(features, DecisionThresholds(other_threshold=0.5))
    assert action == "handoff"
    assert reason == "out_of_domain_classifier"


def test_router_uses_configured_overlap_guard():
    from app.router import get_router
    assert frozenset(("hr", "it")) in get_router().thresholds.overlap_pairs


class TestRetrieval:
    def test_retrieve_returns_result(self):
        from app.retrieval import get_retriever
        retriever = get_retriever()
        result = retriever.retrieve("How do I reset my password?", "it")
        assert result is not None
        assert "faq_id" in result
        assert "answer" in result
        assert "source" in result

    def test_retrieve_has_alternatives(self):
        from app.retrieval import get_retriever
        retriever = get_retriever()
        result = retriever.retrieve("WiFi connection problems", "it")
        assert result is not None
        assert "alternatives" in result

    def test_retrieve_invalid_domain(self):
        from app.retrieval import get_retriever
        retriever = get_retriever()
        result = retriever.retrieve("test", "nonexistent_domain")
        assert result is None

    def test_retrieve_source_format(self):
        from app.retrieval import get_retriever
        retriever = get_retriever()
        result = retriever.retrieve("tuition fee deadline", "finance")
        assert result is not None
        assert "Source:" in result["source"]
        assert "FIN-" in result["source"]


class TestSessionMemory:
    def test_session_creation(self):
        from app.main import _get_session
        session = _get_session("test-session-123")
        assert session is not None
        assert "history" in session
        assert "pending_query" in session

    def test_session_history_limit(self):
        from app.main import _update_session_history, _get_session
        session = _get_session("test-limit-session")
        for i in range(10):
            _update_session_history(session, "it", f"query {i}")
        assert len(session["history"]) <= 5


class TestAPIValidation:
    def test_empty_message_rejected(self):
        from app.main import ChatRequest
        with pytest.raises(Exception):
            ChatRequest(message="")

    def test_long_message_rejected(self):
        from app.main import ChatRequest
        with pytest.raises(Exception):
            ChatRequest(message="x" * 600)

    def test_valid_message_accepted(self):
        from app.main import ChatRequest
        req = ChatRequest(message="How do I reset my password?")
        assert req.message == "How do I reset my password?"

    def test_feedback_model(self):
        from app.main import FeedbackRequest
        fb = FeedbackRequest(session_id="test", faq_id="IT-001", rating=1)
        assert fb.rating == 1


class TestAzureFallback:
    def test_azure_disabled_by_default(self):
        from app.azure_layer import is_azure_enabled
        assert not is_azure_enabled()

    def test_rephrase_returns_original_when_offline(self):
        from app.azure_layer import rephrase_answer
        original = "This is the original answer."
        result, used_azure = rephrase_answer(original, "source")
        assert result == original
        assert not used_azure

    def test_second_opinion_returns_none_when_offline(self):
        from app.azure_layer import azure_second_opinion
        result = azure_second_opinion("test query", {"it": 0.5})
        assert result is None

    def test_validate_rephrase_preserves_urls(self):
        from app.azure_layer import _validate_rephrase
        original = "Visit portal.university.edu for more info."
        good = "Check out portal.university.edu for details."
        bad = "Check out different-site.com for details."
        assert _validate_rephrase(original, good)
        assert not _validate_rephrase(original, bad)

    def test_validate_rephrase_preserves_numbers(self):
        from app.azure_layer import _validate_rephrase
        original = "The fee is $500 per semester."
        good = "Each semester costs $500."
        bad = "Each semester costs $600."
        assert _validate_rephrase(original, good)
        assert not _validate_rephrase(original, bad)


class TestDatabase:
    def test_init_db(self):
        from app.database import init_db
        init_db()

    def test_log_and_retrieve_metrics(self):
        from app.database import init_db, log_query, get_metrics
        init_db()
        log_query("req-test", "sess-test", "test query", "it", 0.8, 0.9, "answer", "IT-001", 50.0)
        metrics = get_metrics()
        assert metrics["total"] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
