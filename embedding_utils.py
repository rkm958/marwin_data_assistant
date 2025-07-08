# embedding_utils.py – Vectorize metadata and enable search using semantic similarity
"""
This file handles two key tasks:
1. embed(texts): Converts a list of descriptive metadata strings (from the 'doc' column) into vector embeddings using OpenAI's embedding model. These embeddings represent the meaning of text in a way machines can understand.
2. search(query, k=5): Takes a user query, embeds it the same way, and compares it to the stored metadata vectors using vector similarity. Returns the most semantically similar metadata entries.

These utilities power the semantic search experience in our metadata assistant.
"""

import os
import numpy as np
import openai
from dotenv import load_dotenv

# Load your OpenAI API key from environment variable
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

model = "text-embedding-3-small"  # cheap & fast embedding model

def embed(texts, batch_size=100):
    """
    Convert a list of text strings into vector embeddings using OpenAI API.

    Args:
        texts (list): List of strings to embed.
        batch_size (int): How many texts to process in one API call.

    Returns:
        np.ndarray: 2D array of embeddings.
    """
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        response = openai.embeddings.create(
            model=model,
            input=batch
        )
        batch_vecs = [item.embedding for item in response.data]
        vectors.extend(batch_vecs)
    return np.array(vectors, dtype="float32")

#----------------------------

"""
    Comment out the below if using load_vector_store.py to search already indexed faiss
"""
def search(query, df_metadata, index, k=5):
    """
    Embed a user query and find the most similar metadata entries based on vector distance.

    Args:
        query (str): User's natural language question.
        df_metadata (DataFrame): Metadata containing 'doc', 'TABLE_NAME', 'COLUMN_NAME'.
        index (faiss.Index): FAISS index of document embeddings.
        k (int): Number of top results to return.

    Returns:
        list[dict]: Top matching entries with metadata and distance scores.
    """
    q_vec = embed([query])[0]          # get embedding for the query
    D, I = index.search(np.array([q_vec]), k)   # find closest vectors and store the found results
    results = []
    for dist, idx in zip(D[0], I[0]):
        results.append({
            "distance": float(dist),
            "doc": df_metadata["doc"].iloc[idx],
            "table": df_metadata["TABLE_NAME"].iloc[idx],
            "column": df_metadata["COLUMN_NAME"].iloc[idx],
        })
    return results



