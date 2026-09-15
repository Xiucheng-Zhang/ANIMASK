# Modified file; the original work is under Apache-2.0, see README (License).
"""Dependency-free memory module.

The "ga" option (LangChain GenerativeAgentMemory + FAISS) is
disabled; the naive vector-store memory is kept, backed by whatever DB
build_db provides (NaiveDB lexical retrieval).
"""
import sys

sys.path.append("../")

from utils import build_db


def build_role_agent_memory(type="naive", **kwargs):
    if type == "ga":
        raise ImportError(
            "GA memory is not available (requires langchain+faiss); "
            "use type='naive'.")
    db_name = kwargs["db_name"]
    embedding = kwargs.get("embedding")
    db_type = kwargs.get("db_type", "naive")
    capacity = kwargs.get("capacity", 5)
    return RoleMemory(db_name=db_name, embedding=embedding,
                      db_type=db_type, capacity=capacity)


class RoleMemory:
    def __init__(self, db_name, embedding, db_type="naive", capacity=5) -> None:
        self.idx = 0
        self.capacity = capacity
        self.db_name = db_name
        self.db = build_db(["_init_"], db_name, db_type, embedding,
                           save_type="temporary")

    def init_from_data(self, data):
        for text in data:
            self.add_record(text)

    def add_record(self, text):
        self.idx += 1
        self.db.add(text, str(self.idx), db_name=self.db_name)

    def search(self, query, top_k):
        return self.db.search(query, top_k, self.db_name)

    def delete_record(self, idx):
        self.db.delete(idx)

    @property
    def len(self):
        return self.db.len
