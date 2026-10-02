import math
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.router import get_router
from app.normalizer import normalize_light
from app.config import load_config


def _bm25_score(query_terms, doc_terms, avg_dl, k1=1.5, b=0.75):
    dl = len(doc_terms)
    doc_counter = Counter(doc_terms)
    score = 0.0
    for term in query_terms:
        tf = doc_counter.get(term, 0)
        if tf == 0:
            continue
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * dl / max(avg_dl, 1))
        score += numerator / denominator
    return score


class DomainSkill:
    def __init__(self, domain, entries, retrieval_config):
        self._domain = domain
        self._entries = entries
        self._top_k = retrieval_config.get("top_k", 3)
        self._bm25_weight = retrieval_config.get("bm25_weight", 0.45)
        self._tfidf_weight = retrieval_config.get("tfidf_weight", 0.55)

        self._texts = []
        self._tokenized = []
        for e in entries:
            combined = f"{e['question']} {' '.join(e.get('keywords', []))}"
            self._texts.append(combined)
            self._tokenized.append(combined.lower().split())

        self._avg_dl = np.mean([len(t) for t in self._tokenized]) if self._tokenized else 1

        self._vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True,
        )
        self._tfidf_matrix = self._vectorizer.fit_transform(self._texts)

    @property
    def domain(self):
        return self._domain

    def retrieve(self, query):
        query_clean = normalize_light(query)
        query_terms = query_clean.split()

        query_vec = self._vectorizer.transform([query_clean])
        tfidf_scores = cosine_similarity(query_vec, self._tfidf_matrix)[0]

        bm25_scores = np.array([
            _bm25_score(query_terms, doc_terms, self._avg_dl)
            for doc_terms in self._tokenized
        ])

        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / bm25_scores.max()
        if tfidf_scores.max() > 0:
            tfidf_norm = tfidf_scores / tfidf_scores.max()
        else:
            tfidf_norm = tfidf_scores

        combined = self._tfidf_weight * tfidf_norm + self._bm25_weight * bm25_scores

        top_indices = combined.argsort()[::-1][: self._top_k]
        results = []
        for idx in top_indices:
            entry = self._entries[idx]
            source_title = entry.get("source_title", f"{self._domain.upper()} FAQ")
            last_updated = entry.get("last_updated", "")
            source_label = f"{self._domain.upper()} FAQ #{entry['id']}"
            if last_updated:
                source_label += f", updated {last_updated}"

            results.append({
                "faq_id": entry["id"],
                "question": entry["question"],
                "answer": entry["answer"],
                "score": float(combined[idx]),
                "source": f"Source: {source_label}",
                "source_title": source_title,
                "follow_up": self._suggest_followup(idx),
            })

        return results

    def _suggest_followup(self, current_idx):
        if len(self._entries) <= 1:
            return None

        current_vec = self._tfidf_matrix[current_idx]
        all_scores = cosine_similarity(current_vec, self._tfidf_matrix)[0]
        all_scores[current_idx] = -1

        next_idx = int(all_scores.argmax())
        if all_scores[next_idx] > 0.1:
            return self._entries[next_idx]["question"]
        return None


class DomainRetriever:
    def __init__(self):
        router = get_router()
        self._domain_data = router.domain_data
        config = load_config()
        retrieval_config = config.get("retrieval", {})
        self._skills = {}

        for domain, entries in self._domain_data.items():
            self._skills[domain] = DomainSkill(domain, entries, retrieval_config)

    def retrieve(self, query, domain):
        if domain not in self._skills:
            return None

        results = self._skills[domain].retrieve(query)
        if not results:
            return None

        best = results[0]
        alternatives = [
            {"faq_id": r["faq_id"], "question": r["question"]}
            for r in results[1:]
            if r["score"] > 0.1
        ]

        return {
            "faq_id": best["faq_id"],
            "question": best["question"],
            "answer": best["answer"],
            "score": best["score"],
            "source": best["source"],
            "source_title": best["source_title"],
            "follow_up": best["follow_up"],
            "alternatives": alternatives,
        }

    def retrieve_top_k(self, query, domain, k=3):
        if domain not in self._skills:
            return []
        return self._skills[domain].retrieve(query)[:k]


_retriever_instance = None


def get_retriever(force_reload=False):
    global _retriever_instance
    if _retriever_instance is None or force_reload:
        _retriever_instance = DomainRetriever()
    return _retriever_instance
