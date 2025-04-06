# src/chatbot.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
from src.search.section_coarse_search import coarse_search_sections
from src.search.fine_search import fine_search_chunks
from src.inference.embedding_model import embedding_model
from src.inference.llm_model import local_llm  # 로컬 LLM 구현 예시

class PDFChatBot:
    def __init__(self, sections, chunk_index):
        """
        sections: 각 섹션은 {"title", "title_emb", "avg_chunk_emb", ...}를 포함하는 리스트
        chunk_index: 각 청크는 {"embedding": [...], "metadata": {...}} 형태의 리스트
        """
        self.sections = sections
        self.chunk_index = chunk_index

    def build_prompt(self, user_query, retrieved_chunks):
        """
        LLM에 전달할 프롬프트를 구성합니다.
        retrieved_chunks: [{"embedding": [...], "metadata": {...}}, ...]
        """
        context_parts = []
        for item in retrieved_chunks:
            meta = item.get("metadata", {})
            section_title = meta.get("section_title", "")
            content = meta.get("content", "")
            context_parts.append(f"[{section_title}] {content}")
        context_text = "\n\n".join(context_parts)
        prompt = f"""
Please answer the question below using the following document context.

=== Document Context ===
{context_text}

=== User Question ===
{user_query}
"""
        return prompt.strip()

    def answer(self, query: str, beta: float = 0.3, top_sections: int = 3, top_chunks: int = 5):
        """
        1) Coarse Search: 섹션 레벨에서 상위 top_sections개 섹션을 찾습니다.
        2) Fine Search: 해당 섹션 내 청크들 중 상위 top_chunks개를 검색합니다.
        3) LLM에 프롬프트를 전달하여 최종 답변을 생성합니다.
        """
        # Coarse Search (섹션 레벨)
        relevant_secs = coarse_search_sections(query, self.sections, beta=beta, top_k=top_sections)
        
        # Fine Search (청크 레벨)
        query_emb = embedding_model.get_embedding(query)
        best_chunks = fine_search_chunks(query_emb, self.chunk_index, relevant_secs, top_k=top_chunks)
        
        # LLM 답변 생성
        prompt = self.build_prompt(query, best_chunks)
        answer_text = local_llm.generate(prompt, max_length=4096)
        return answer_text

if __name__ == "__main__":
    sections_path = "data/extracted/sections_with_emb.json"
    chunk_index_path = "data/index/sample_chunks_vectors.json"

    if not os.path.exists(sections_path):
        print(f"[ERROR] Sections file not found: {sections_path}")
        exit(1)
    else:
        with open(sections_path, 'r', encoding='utf-8') as f:
            sections = json.load(f)

    if not os.path.exists(chunk_index_path):
        print(f"[ERROR] Chunk index file not found: {chunk_index_path}")
        exit(1)
    else:
        with open(chunk_index_path, 'r', encoding='utf-8') as f:
            chunk_index = json.load(f)

    chatbot = PDFChatBot(sections, chunk_index)
    print("Chatbot is ready. Enter your question below:")

    while True:
        query = input("Question (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        answer = chatbot.answer(query)
        print("Answer:", answer)