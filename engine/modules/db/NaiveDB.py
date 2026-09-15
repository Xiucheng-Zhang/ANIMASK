"""Dependency-free lexical retrieval DB, drop-in replacement for a vector store.

Scoring: token-overlap (case-folded, alnum tokens) between query and each
stored document, IDF-weighted. Good enough for short-story scale retrieval
(references / knowledges, top_k 1-3); swap back to a vector DB for novels.
"""
import math
import re
from collections import Counter, defaultdict


def _toks(s: str) -> list:
    """ASCII runs as words; CJK runs as character bigrams — the old rule
    kept a whole Chinese passage as ONE token, so zh queries and documents
    almost never overlapped and retrieval degraded to insertion order."""
    toks = []
    for run in re.findall(r"[a-z0-9]+|[一-鿿]+", s.lower()):
        if run[0].isascii() or len(run) == 1:
            toks.append(run)
        else:
            toks.extend(run[i:i + 2] for i in range(len(run) - 1))
    return toks


class NaiveDB:
    def __init__(self, embedding=None, save_type="temporary"):
        self.stores = defaultdict(dict)   # db_name -> {id: text}
        self._df = defaultdict(Counter)   # db_name -> token document frequency

    def init_from_data(self, data, db_name):
        for i, text in enumerate(data):
            self.add(text, str(i), db_name)

    def add(self, text, idx, db_name):
        self.stores[db_name][idx] = text
        self._df[db_name].update(set(_toks(text)))

    def delete(self, idx, db_name=None):
        for name in ([db_name] if db_name else list(self.stores)):
            self.stores[name].pop(idx, None)

    def search(self, query, n_results, db_name):
        docs = self.stores.get(db_name, {})
        if not docs:
            return []
        q = set(_toks(query))
        n_docs = max(1, len(docs))
        df = self._df[db_name]

        def score(text: str) -> float:
            t = set(_toks(text))
            common = q & t
            return sum(math.log(1 + n_docs / (1 + df[w])) for w in common)

        ranked = sorted(docs.values(), key=score, reverse=True)
        return ranked[:n_results]

    @property
    def len(self):
        return sum(len(v) for v in self.stores.values())
