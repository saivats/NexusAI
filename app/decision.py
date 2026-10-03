from dataclasses import dataclass, field

ACTION_ANSWER = "answer"
ACTION_CLARIFY = "clarify"
ACTION_HANDOFF = "handoff"

REASON_OUT_OF_DOMAIN_CLASS = "out_of_domain_classifier"
REASON_OUT_OF_DOMAIN_SIMILARITY = "out_of_domain_similarity"
REASON_LOW_CONFIDENCE = "low_confidence"
REASON_CONFIDENT = "confident"
REASON_CLOSE_OVERLAP_PAIR = "close_overlapping_pair"
REASON_CLOSE_TOP_TWO = "close_top_two"
REASON_WEAK_RAW_SCORE = "weak_raw_score"
REASON_LOW_CALIBRATED = "below_answer_confidence"

OUT_OF_DOMAIN_REASONS = {REASON_OUT_OF_DOMAIN_CLASS, REASON_OUT_OF_DOMAIN_SIMILARITY}

DEFAULT_OVERLAP_PAIRS = (
    ("finance", "facilities"),
    ("finance", "academics"),
    ("hr", "it"),
    ("admissions", "academics"),
)


@dataclass(frozen=True)
class RoutingFeatures:
    top_domain: str
    second_domain: str
    top_score: float
    margin: float
    calibrated: float
    max_similarity: float
    other_probability: float


@dataclass
class DecisionThresholds:
    cal_answer: float = 0.55
    cal_low: float = 0.20
    margin: float = 0.04
    pair_margin: float = 0.08
    sim_floor: float = 0.05
    other_threshold: float = 0.5
    answer_floor: float = 0.12
    overlap_pairs: frozenset = field(default_factory=lambda: normalize_pairs(DEFAULT_OVERLAP_PAIRS))


def normalize_pairs(pairs):
    return frozenset(frozenset((first, second)) for first, second in pairs)


def required_margin(features, thresholds):
    pair = frozenset((features.top_domain, features.second_domain))
    if pair in thresholds.overlap_pairs:
        return max(thresholds.margin, thresholds.pair_margin)
    return thresholds.margin


def decide(features, thresholds):
    if features.other_probability >= thresholds.other_threshold:
        return ACTION_HANDOFF, REASON_OUT_OF_DOMAIN_CLASS
    if features.max_similarity < thresholds.sim_floor:
        return ACTION_HANDOFF, REASON_OUT_OF_DOMAIN_SIMILARITY
    if features.calibrated < thresholds.cal_low:
        return ACTION_HANDOFF, REASON_LOW_CONFIDENCE

    needed_margin = required_margin(features, thresholds)
    if features.margin < needed_margin:
        is_overlap = needed_margin > thresholds.margin
        return ACTION_CLARIFY, REASON_CLOSE_OVERLAP_PAIR if is_overlap else REASON_CLOSE_TOP_TWO
    if features.top_score < thresholds.answer_floor:
        return ACTION_CLARIFY, REASON_WEAK_RAW_SCORE
    if features.calibrated < thresholds.cal_answer:
        return ACTION_CLARIFY, REASON_LOW_CALIBRATED
    return ACTION_ANSWER, REASON_CONFIDENT


def thresholds_from_config(routing_config):
    pairs = routing_config.get("overlap_pairs")
    overlap = normalize_pairs(tuple(p.split("|")) for p in pairs) if pairs else normalize_pairs(DEFAULT_OVERLAP_PAIRS)
    defaults = DecisionThresholds()
    return DecisionThresholds(
        cal_answer=routing_config.get("cal_answer", defaults.cal_answer),
        cal_low=routing_config.get("cal_low", defaults.cal_low),
        margin=routing_config.get("margin", defaults.margin),
        pair_margin=routing_config.get("pair_margin", defaults.pair_margin),
        sim_floor=routing_config.get("sim_floor", defaults.sim_floor),
        other_threshold=routing_config.get("other_threshold", defaults.other_threshold),
        answer_floor=routing_config.get("answer_floor", defaults.answer_floor),
        overlap_pairs=overlap,
    )
