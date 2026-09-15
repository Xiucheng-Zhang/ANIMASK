# Modified file; the original work is under Apache-2.0, see README (License).
"""Embedding loader for the engine's retrieval layer.

The pipeline uses lexical retrieval (NaiveDB), which needs no embedding
model, so "naive" returns None. Any other name is rejected explicitly: the
vector-store path is not part of this release.
"""


def get_embedding_model(model_name="naive", language="en"):
    if model_name in (None, "", "naive", "none"):
        return None
    raise ValueError(f"embedding model {model_name!r} is not supported; "
                     "this release uses lexical retrieval (embedding_name=\"naive\")")
