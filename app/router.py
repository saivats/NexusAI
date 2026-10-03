import json
import math
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

from app.config import load_config
from app.normalizer import normalize_query, normalize_light

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

_SENSITIVE_PATTERN = None


def _build_sensitive_pattern(config):
    global _SENSITIVE_PATTERN
    keywords = config.get("sensitive_keywords", [])
    if keywords:
        escaped = [re.escape(k) for k in keywords]
        _SENSITIVE_PATTERN = re.compile(r"\b(" + "|".join(escaped) + r")\b", re.IGNORECASE)
    return _SENSITIVE_PATTERN


def _load_domain_data(domains):
    domain_data = {}
    for domain in domains:
        filepath = DATA_DIR / f"{domain}.json"
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                domain_data[domain] = json.load(f)
    return domain_data


def _build_domain_corpus(domain_data):
    corpus = {}
    for domain, entries in domain_data.items():
        texts = []
        for entry in entries:
            combined = f"{entry['question']} {' '.join(entry.get('keywords', []))}"
            texts.append(combined)
        corpus[domain] = " ".join(texts)
    return corpus


def _extract_domain_keywords(domain_data):
    all_keywords = {}
    for domain, entries in domain_data.items():
        kw_set = set()
        for entry in entries:
            for kw in entry.get("keywords", []):
                kw_set.add(kw.lower())
        all_keywords[domain] = kw_set
    return all_keywords


def _build_training_data(domain_data):
    texts = []
    labels = []
    for domain, entries in domain_data.items():
        for entry in entries:
            question = entry["question"]
            keywords = " ".join(entry.get("keywords", []))
            texts.append(normalize_query(f"{question} {keywords}"))
            labels.append(domain)
            texts.append(normalize_query(question))
            labels.append(domain)
            for kw in entry.get("keywords", []):
                if len(kw) > 2:
                    texts.append(normalize_query(kw))
                    labels.append(domain)
    return texts, labels


class DomainRouter:
    def __init__(self):
        self._config = load_config()
        routing_config = self._config.get("routing", {})
        self._t_high = routing_config.get("t_high", 0.42)
        self._t_low = routing_config.get("t_low", 0.15)
        self._margin = routing_config.get("margin", 0.08)
        self._cal_answer = routing_config.get("cal_answer", 0.55)
        self._cal_low = routing_config.get("cal_low", 0.20)
        calibration_config = routing_config.get("calibration", {})
        self._cal_coef_score = calibration_config.get("coef_score", 9.0)
        self._cal_coef_margin = calibration_config.get("coef_margin", 6.0)
        self._cal_intercept = calibration_config.get("intercept", -2.2)
        self._tfidf_weight = routing_config.get("tfidf_weight", 0.35)
        self._keyword_weight = routing_config.get("keyword_weight", 0.25)
        self._classifier_weight = routing_config.get("classifier_weight", 0.40)

        domains = self._config.get("domains", ["it", "hr", "finance", "facilities"])
        self._domain_data = _load_domain_data(domains)
        self._domain_corpus = _build_domain_corpus(self._domain_data)
        self._domain_keywords = _extract_domain_keywords(self._domain_data)
        self._sensitive_config = self._config.get("sensitive_contact", {})

        _build_sensitive_pattern(self._config)

        self._domain_names = list(self._domain_corpus.keys())
        corpus_texts = [self._domain_corpus[d] for d in self._domain_names]

        self._vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            analyzer="word",
            max_features=8000,
            sublinear_tf=True,
        )
        self._tfidf_matrix = self._vectorizer.fit_transform(corpus_texts)

        self._char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=5000,
            sublinear_tf=True,
        )
        self._char_tfidf_matrix = self._char_vectorizer.fit_transform(corpus_texts)

        train_texts, train_labels = _build_training_data(self._domain_data)
        self._clf_vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 3),
            max_features=10000,
            sublinear_tf=True,
        )
        train_matrix = self._clf_vectorizer.fit_transform(train_texts)

        base_clf = LinearSVC(max_iter=5000, C=1.0)
        self._classifier = CalibratedClassifierCV(base_clf, cv=3, method="sigmoid")
        self._classifier.fit(train_matrix, train_labels)

    @property
    def domain_data(self):
        return self._domain_data

    @property
    def domain_names(self):
        return self._domain_names

    def is_sensitive(self, query):
        if _SENSITIVE_PATTERN is None:
            return False
        return bool(_SENSITIVE_PATTERN.search(query))

    def get_sensitive_response(self):
        contact = self._sensitive_config
        return {
            "email": contact.get("email", "welfare@university.edu"),
            "phone": contact.get("phone", "ext. 1100"),
            "message": contact.get("message", "Your wellbeing matters. Please reach out to our Welfare & Support team."),
        }

    def _keyword_score(self, query_lower, domain):
        keywords = self._domain_keywords.get(domain, set())
        if not keywords:
            return 0.0
        hits = sum(1 for kw in keywords if kw in query_lower)
        return min(hits / max(len(keywords) * 0.08, 1), 1.0)

    def _classifier_scores(self, query_normalized):
        query_vec = self._clf_vectorizer.transform([query_normalized])
        proba = self._classifier.predict_proba(query_vec)[0]
        classes = self._classifier.classes_
        return {cls: float(prob) for cls, prob in zip(classes, proba)}

    def score_domains(self, query):
        query_normalized = normalize_query(query)
        query_light = normalize_light(query)

        query_vec = self._vectorizer.transform([query_normalized])
        tfidf_scores = cosine_similarity(query_vec, self._tfidf_matrix)[0]

        char_vec = self._char_vectorizer.transform([query_light])
        char_scores = cosine_similarity(char_vec, self._char_tfidf_matrix)[0]

        combined_tfidf = 0.7 * tfidf_scores + 0.3 * char_scores

        clf_scores = self._classifier_scores(query_normalized)

        scores = {}
        for idx, domain in enumerate(self._domain_names):
            tfidf_s = float(combined_tfidf[idx])
            kw_s = self._keyword_score(query_light, domain)
            clf_s = clf_scores.get(domain, 0.0)

            blended = (
                self._tfidf_weight * tfidf_s
                + self._keyword_weight * kw_s
                + self._classifier_weight * clf_s
            )
            scores[domain] = round(blended, 4)

        return scores

    def calibrate_confidence(self, raw_score, margin=0.0):
        logit = self._cal_intercept + self._cal_coef_score * raw_score + self._cal_coef_margin * margin
        logit = max(min(logit, 30.0), -30.0)
        return round(1.0 / (1.0 + math.exp(-logit)), 4)

    def set_calibration(self, coef_score, coef_margin, intercept):
        self._cal_coef_score = coef_score
        self._cal_coef_margin = coef_margin
        self._cal_intercept = intercept

    def confidence_label(self, calibrated_score):
        if calibrated_score >= 0.70:
            return "High"
        elif calibrated_score >= 0.40:
            return "Medium"
        return "Low"

    def route(self, query):
        if self.is_sensitive(query):
            sensitive = self.get_sensitive_response()
            return {
                "top_domain": "sensitive",
                "confidence": 1.0,
                "calibrated_confidence": 1.0,
                "confidence_label": "High",
                "second_domain": "sensitive",
                "second_confidence": 0.0,
                "action": "sensitive",
                "all_scores": {},
                "sensitive_response": sensitive,
            }

        scores = self.score_domains(query)
        sorted_domains = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        top_domain, top_score = sorted_domains[0]
        second_domain, second_score = sorted_domains[1] if len(sorted_domains) > 1 else ("unknown", 0.0)

        margin = top_score - second_score
        calibrated = self.calibrate_confidence(top_score, margin)
        label = self.confidence_label(calibrated)

        if calibrated < self._cal_low:
            action = "handoff"
        elif calibrated >= self._cal_answer and margin >= self._margin:
            action = "answer"
        else:
            action = "clarify"

        return {
            "top_domain": top_domain,
            "confidence": top_score,
            "calibrated_confidence": calibrated,
            "confidence_label": label,
            "second_domain": second_domain,
            "second_confidence": second_score,
            "action": action,
            "all_scores": scores,
        }

    def update_thresholds(self, t_high=None, t_low=None, margin=None,
                          cal_answer=None, cal_low=None):
        if t_high is not None:
            self._t_high = t_high
        if t_low is not None:
            self._t_low = t_low
        if margin is not None:
            self._margin = margin
        if cal_answer is not None:
            self._cal_answer = cal_answer
        if cal_low is not None:
            self._cal_low = cal_low


_router_instance = None


def get_router(force_reload=False):
    global _router_instance
    if _router_instance is None or force_reload:
        _router_instance = DomainRouter()
    return _router_instance
