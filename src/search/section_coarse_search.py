# src/search/section_coarse_search.py

import hnswlib
import numpy as np
from typing import List, Dict
from src.inference.embedding_model import embedding_model

def build_section_reps(sections: List[Dict], beta: float = 0.3) -> np.ndarray:
    """
    For each section in sections, compute a combined representative vector as:
      rep = beta * title_emb + (1 - beta) * avg_chunk_emb
    Returns a numpy array of shape (num_sections, dim).
    """
    reps = []
    for sec in sections:
        title_emb = np.array(sec.get("title_emb"), dtype=np.float32)
        avg_emb = np.array(sec.get("avg_chunk_emb"), dtype=np.float32)
        rep = beta * title_emb + (1 - beta) * avg_emb
        reps.append(rep)
    return np.array(reps, dtype=np.float32)

def hnsw_section_search(query: str, sections: List[Dict], beta: float = 0.3, top_k: int = 3) -> List[Dict]:
    """
    Performs coarse search among sections using hnswlib.
    sections: list of section dicts; each should have "title_emb" and "avg_chunk_emb" fields.
    query: query string.
    beta: weight for title embedding.
    top_k: number of sections to return.
    """
    query_emb = np.array(embedding_model.get_embedding(query), dtype=np.float32)
    
    section_reps = build_section_reps(sections, beta)
    num_sections, dim = section_reps.shape

    p = hnswlib.Index(space='cosine', dim=dim)
    p.init_index(max_elements=num_sections, ef_construction=200, M=16)
    p.add_items(section_reps)
    p.set_ef(50)

    query_np = query_emb.reshape(1, dim)
    labels, distances = p.knn_query(query_np, k=top_k)
    
    top_sections = [sections[idx] for idx in labels[0]]
    return top_sections