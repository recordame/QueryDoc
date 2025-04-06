# src/search/fine_search.py

import hnswlib
import numpy as np
from typing import List, Dict

def hnsw_fine_search(query_emb: List[float], chunk_index: List[Dict], target_sections: List[Dict], top_k: int = 5) -> List[Dict]:
    """
    Filters chunk_index by target_sections and performs hnswlib search on the filtered set.
    
    Parameters:
      - query_emb: query embedding as list of floats.
      - chunk_index: list of chunk dicts; each should have "embedding" and "metadata" with key "section_title".
      - target_sections: list of section dicts; each has a "title" key.
      - top_k: number of top chunks to return.
      
    Returns:
      A list of top_k chunk dicts.
    """
    target_titles = {sec["title"] for sec in target_sections}
    candidates = [item for item in chunk_index if item["metadata"].get("section_title") in target_titles]

    if not candidates:
        return []

    vectors = np.array([item["embedding"] for item in candidates], dtype=np.float32)
    num_candidates, dim = vectors.shape

    p = hnswlib.Index(space='cosine', dim=dim)
    p.init_index(max_elements=num_candidates, ef_construction=200, M=16)
    p.add_items(vectors)
    p.set_ef(50)

    query_np = np.array(query_emb, dtype=np.float32).reshape(1, dim)
    labels, distances = p.knn_query(query_np, k=top_k)
    
    top_chunks = [candidates[idx] for idx in labels[0]]
    return top_chunks