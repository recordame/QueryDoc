# src/search/vector_search.py

import hnswlib
import numpy as np
from typing import List, Dict

def hnsw_vector_search(query_emb: List[float], index_data: List[Dict], top_k: int = 5) -> List[Dict]:
    """
    Performs vector search using hnswlib.
    index_data: a list of items, each item is a dict with keys "embedding" and "metadata".
    query_emb: query embedding as a list of floats.
    Returns the top_k nearest items.
    """
    vectors = np.array([item["embedding"] for item in index_data], dtype=np.float32)
    num_elements, dim = vectors.shape

    # Create hnswlib index (cosine similarity space)
    p = hnswlib.Index(space='cosine', dim=dim)
    p.init_index(max_elements=num_elements, ef_construction=200, M=16)
    p.add_items(vectors)
    p.set_ef(50)

    query_np = np.array(query_emb, dtype=np.float32).reshape(1, dim)
    labels, distances = p.knn_query(query_np, k=top_k)
    
    top_results = [index_data[idx] for idx in labels[0]]
    return top_results