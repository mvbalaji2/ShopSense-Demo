"""Small dependency-free TF-IDF vector store for the ShopSense knowledge base.

The index is persisted to data/knowledge_index.json so retrieval is a real,
inspectable local RAG step rather than a prompt-only keyword lookup.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

TOKEN_ALIASES = {
    "tracking": "shipment_status", "track": "shipment_status", "tracking_link": "shipment_status",
    "scan": "shipment_status", "scans": "shipment_status", "moving": "shipment_status",
    "stopped": "shipment_status", "carrier": "shipment_status", "delivery": "shipment_status",
    "chipped": "damaged", "broken": "damaged", "damage": "damaged", "damaged": "damaged",
    "charged": "charge", "charging": "charge", "duplicate": "duplicate_charge",
    "twice": "duplicate_charge", "address": "address_change", "change": "address_change",
    "refund": "refund", "refunded": "refund", "return": "return",
}
STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "was", "are", "has", "have", "from",
    "your", "you", "can", "our", "but", "not", "how", "what", "when", "where", "please",
    "into", "they", "will", "just", "been", "would", "could", "about", "need", "like",
}


def tokens(text: str) -> list[str]:
    raw = re.findall(r"[a-z][a-z0-9_]{2,}", text.lower())
    output: list[str] = []
    for token in raw:
        if token in STOPWORDS:
            continue
        output.append(TOKEN_ALIASES.get(token, token))
    return output


class VectorStore:
    def __init__(self, documents: list[dict], index_path: Path):
        self.documents = documents
        self.index_path = index_path
        self.document_frequency: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.vectors: list[dict[str, float]] = []
        self.norms: list[float] = []
        self._build_or_load()

    def _build_or_load(self) -> None:
        fingerprint = [(doc["id"], doc["title"], doc["text"]) for doc in self.documents]
        if self.index_path.exists():
            try:
                payload = json.loads(self.index_path.read_text(encoding="utf-8"))
                if payload.get("fingerprint") == fingerprint:
                    self.document_frequency = payload["document_frequency"]
                    self.idf = payload["idf"]
                    self.vectors = payload["vectors"]
                    self.norms = payload["norms"]
                    return
            except (OSError, ValueError, KeyError, TypeError):
                pass
        self._build()
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(json.dumps({
            "version": 1, "fingerprint": fingerprint, "document_frequency": self.document_frequency,
            "idf": self.idf, "vectors": self.vectors, "norms": self.norms,
        }, indent=2), encoding="utf-8")

    def _build(self) -> None:
        token_sets = [set(tokens(doc["title"] + " " + doc["text"])) for doc in self.documents]
        for token_set in token_sets:
            for token in token_set:
                self.document_frequency[token] = self.document_frequency.get(token, 0) + 1
        count = len(self.documents)
        self.idf = {token: math.log((1 + count) / (1 + frequency)) + 1 for token, frequency in self.document_frequency.items()}
        self.vectors, self.norms = [], []
        for doc in self.documents:
            term_counts: dict[str, int] = {}
            for token in tokens(doc["title"] + " " + doc["text"]):
                term_counts[token] = term_counts.get(token, 0) + 1
            total = max(1, sum(term_counts.values()))
            vector = {token: (frequency / total) * self.idf.get(token, 1) for token, frequency in term_counts.items()}
            self.vectors.append(vector)
            self.norms.append(math.sqrt(sum(value * value for value in vector.values())) or 1.0)

    def search(self, query: str, limit: int = 3) -> list[dict]:
        query_counts: dict[str, int] = {}
        for token in tokens(query):
            query_counts[token] = query_counts.get(token, 0) + 1
        total = max(1, sum(query_counts.values()))
        query_vector = {token: (frequency / total) * self.idf.get(token, 1) for token, frequency in query_counts.items()}
        query_norm = math.sqrt(sum(value * value for value in query_vector.values())) or 1.0
        ranked = []
        for index, vector in enumerate(self.vectors):
            score = sum(query_vector.get(token, 0) * value for token, value in vector.items()) / (query_norm * self.norms[index])
            if score > 0:
                ranked.append((score, index))
        ranked.sort(reverse=True)
        results = []
        for score, index in ranked[:limit]:
            results.append({**self.documents[index], "score": round(score, 4)})
        return results
