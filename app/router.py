import json
import math
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


DATA_DIR = Path(__file__).resolve().parent.parent / "data"

DOMAINS = ["it", "hr", "finance", "facilities"]

KEYWORD_BOOST = 0.15
TFIDF_WEIGHT = 0.75
KEYWORD_WEIGHT = 0.25


def _load_domain_data():
    domain_data = {}
    for domain in DOMAINS:
        filepath = DATA_DIR / f"{domain}.json"
        with open(filepath, "r", encoding="utf-8") as f:
            domain_data[domain] = json.load(f)
    return domain_data


def _build_domain_corpus(domain_data):
    corpus = {}
    for domain, entries in domain_data.items():
        texts = []
        for entry in entries:
            combined = f"{entry['question']} {' '.join(entry['keywords'])}"
            texts.append(combined)
        corpus[domain] = " ".join(texts)
    return corpus


def _extract_domain_keywords(domain_data):
    all_keywords = {}
    for domain, entries in domain_data.items():
        kw_set = set()
        for entry in entries:
            for kw in entry["keywords"]:
                kw_set.add(kw.lower())
        all_keywords[domain] = kw_set
    return all_keywords


class DomainRouter:
    def __init__(self):
        self._domain_data = _load_domain_data()
        self._domain_corpus = _build_domain_corpus(self._domain_data)
        self._domain_keywords = _extract_domain_keywords(self._domain_data)

        self._domain_names = list(self._domain_corpus.keys())
        corpus_texts = [self._domain_corpus[d] for d in self._domain_names]

        self._vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=5000,
        )
        self._tfidf_matrix = self._vectorizer.fit_transform(corpus_texts)

    @property
    def domain_data(self):
        return self._domain_data

    def _keyword_score(self, query_lower, domain):
        keywords = self._domain_keywords[domain]
        hits = sum(1 for kw in keywords if kw in query_lower)
        if not keywords:
            return 0.0
        return min(hits / max(len(keywords) * 0.1, 1), 1.0)

    def score_domains(self, query):
        query_lower = query.lower().strip()
        query_vec = self._vectorizer.transform([query_lower])
        tfidf_scores = cosine_similarity(query_vec, self._tfidf_matrix)[0]

        scores = {}
        for idx, domain in enumerate(self._domain_names):
            tfidf_s = float(tfidf_scores[idx])
            kw_s = self._keyword_score(query_lower, domain)
            blended = TFIDF_WEIGHT * tfidf_s + KEYWORD_WEIGHT * kw_s
            scores[domain] = round(blended, 4)

        return scores

    def route(self, query):
        scores = self.score_domains(query)
        sorted_domains = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        top_domain, top_score = sorted_domains[0]
        second_domain, second_score = sorted_domains[1]

        ambiguous = (
            top_score >= 0.08
            and top_score < 0.15
        ) or (
            top_score >= 0.15
            and abs(top_score - second_score) < 0.05
        )

        if top_score < 0.08:
            action = "handoff"
        elif ambiguous:
            action = "clarify"
        else:
            action = "answer"

        return {
            "top_domain": top_domain,
            "confidence": top_score,
            "second_domain": second_domain,
            "second_confidence": second_score,
            "action": action,
            "all_scores": scores,
        }


_router_instance = None


def get_router():
    global _router_instance
    if _router_instance is None:
        _router_instance = DomainRouter()
    return _router_instance
