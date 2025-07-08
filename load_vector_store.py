# load_vector_store.py – Reload FAISS index + metadata for instant search
"""
Why this file exists (layman explanation)
========================================
Building the vector index can take minutes (database hit, OpenAI calls, FAISS build).
But once the index is saved to disk we want to **skip all that heavy lifting** on every
app restart.  This module gives you a one‑liner to *load* the pre‑built index so the
system is ready to answer queries instantly.

What the functions do
---------------------
1. **load_index(filepath)**
   • Opens the pickle file created by `vector_store.save_index()`.
   • Returns three objects:
       - FAISS `index`  → performs similarity search
       - `docs` (list[str]) → original metadata text chunks
       - `paths` (list[list[str]]) → [[TABLE_NAME, COLUMN_NAME], ...] for display

2. **semantic_search(query, k, index, docs, paths)**
   • Embeds the user query using the same `embed()` helper.
   • Runs `index.search()` to get the *k* closest vectors.
   • Formats the results into a friendly list of dicts you can pass to the LLM.
"""

import pickle
import numpy as np
from typing import Tuple, List, Dict
import pandas as pd

import faiss
from embedding_utils import embed  # re‑use the same embedding model so vectors live in same space

# -----------------------------------------------------------------------------
# 1. Load the stored index + metadata
# -----------------------------------------------------------------------------

def load_index(path="vector_index.faiss"):
    """
    Load FAISS index and full metadata from pickle.

    Returns:
        df_metadata (pd.DataFrame), index (faiss.Index)
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
        index = data["index"]
        df_metadata = pd.DataFrame(data["metadata"])
    
    #print(f"📦 Loaded index from {path}")
    return df_metadata, index

# -----------------------------------------------------------------------------
# 2. Semantic search given a user query
# -----------------------------------------------------------------------------

def semantic_search(query: str, k: int, index: faiss.Index, docs: List[str], paths: List[List[str]]) -> List[Dict]:
    """Return top‑k matches: distance, doc text, table, column."""
    q_vec = embed([query])[0]
    D, I = index.search(np.array([q_vec]), k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        table, column = paths[idx]
        results.append({
            "distance": float(dist),
            "doc": docs[idx],
            "table": table,
            "column": column,
        })
    return results

# Example usage (remove or wrap under __name__ check in production)
if __name__ == "__main__":
    idx, docs_list, path_list = load_index()
    hits = semantic_search("duplicate applications reason for decline", 3, idx, docs_list, path_list)
    for h in hits:
        print(h)
