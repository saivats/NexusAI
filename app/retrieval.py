from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.router import get_router


class DomainRetriever:
    def __init__(self):
        router = get_router()
        self._domain_data = router.domain_data
        self._vectorizers = {}
        self._tfidf_matrices = {}
        self._entries = {}

        for domain, entries in self._domain_data.items():
            texts = [
                f"{e['question']} {' '.join(e['keywords'])}" for e in entries
            ]
            vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                max_features=3000,
            )
            matrix = vectorizer.fit_transform(texts)
            self._vectorizers[domain] = vectorizer
            self._tfidf_matrices[domain] = matrix
            self._entries[domain] = entries

    def retrieve(self, query, domain):
        if domain not in self._vectorizers:
            return None

        vectorizer = self._vectorizers[domain]
        matrix = self._tfidf_matrices[domain]
        entries = self._entries[domain]

        query_vec = vectorizer.transform([query.lower().strip()])
        scores = cosine_similarity(query_vec, matrix)[0]
        best_idx = int(scores.argmax())
        best_entry = entries[best_idx]

        return {
            "faq_id": best_entry["id"],
            "question": best_entry["question"],
            "answer": best_entry["answer"],
            "score": float(scores[best_idx]),
            "source": f"Source: {domain.upper()} FAQ #{best_entry['id']}",
        }


_retriever_instance = None


def get_retriever():
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = DomainRetriever()
    return _retriever_instance
