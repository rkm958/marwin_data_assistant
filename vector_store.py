# vector_store.py – Build and persist vector index for semantic search
"""
This file takes care of two core responsibilities:

1. **Indexing**: It uses FAISS to index the embeddings of all metadata entries. This makes future similarity searches fast and efficient.
2. **Persistence**: It saves both the FAISS index and associated metadata (like the original doc string and table/column path) using pickle, so that the index can be reloaded quickly without recomputing everything.

Functions:
- build_faiss_index(embeddings): Builds a FAISS index from a 2D numpy array of embeddings.
- save_index(index, df_metadata, filepath): Saves the FAISS index and associated metadata to disk.
"""

import faiss
import pickle
import numpy as np

def build_faiss_index(embeddings):
    """
    Create a FAISS index for fast vector similarity search.

    Args:
        embeddings (np.ndarray): 2D array where each row is an embedding vector.

    Returns:
        faiss.Index: Indexed FAISS object.
    """
    dim = embeddings.shape[1]  # e.g., 1536 for OpenAI embeddings
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print("✅ vectors indexed:", index.ntotal)
    return index


def save_index(index, df_metadata, filepath="vector_index.faiss"):
    """
    Save the FAISS index and full metadata using pickle.

    Args:
        index (faiss.Index): Trained FAISS index object.
        df_metadata (pd.DataFrame): Must include all relevant metadata fields used in prompting.
        filepath (str): File path to save the index and metadata.
        
    """
    with open(filepath, "wb") as f:
        pickle.dump({
            "index": index,
            "metadata": df_metadata.to_dict(orient="list")  # Save all columns
        }, f)
    print(f"💾 Saved full index and metadata to {filepath}")



    
