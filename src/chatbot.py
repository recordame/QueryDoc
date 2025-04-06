# src/chatbot.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.search.section_coarse_search import hnsw_section_search
from src.search.fine_search import hnsw_fine_search
from src.inference.embedding_model import embedding_model
from src.inference.llm_model import local_llm  # LLM 구현 예시

class PDFChatBot:
    def __init__(self, sections, chunk_index):
        """
        sections: 섹션 정보 리스트 (각 섹션은 "title", "title_emb", "avg_chunk_emb" 등을 포함)
        chunk_index: 청크 임베딩 리스트 (각 항목은 {"embedding": [...], "metadata": {...}} 형태)
        """
        self.sections = sections
        self.chunk_index = chunk_index

    def build_prompt(self, user_query, retrieved_chunks):
        """
        LLM에게 전달할 프롬프트를 구성합니다.
        retrieved_chunks: [{"embedding": [...], "metadata": {...}}, ...]
        """
        context_parts = []
        for item in retrieved_chunks:
            meta = item["metadata"]
            section_t = meta.get("section_title", "")
            content = meta.get("content", "")
            context_parts.append(f"[{section_t}] {content}")

        context_text = "\n\n".join(context_parts)
        prompt = f"""
아래 문서를 참조하여 질문에 답변해 주세요.

=== 문서 내용 ===
{context_text}

=== 사용자 질문 ===
{user_query}
"""
        return prompt.strip()

    def answer(self, query: str, beta=0.3, top_sections=3, top_chunks=5):
        """
        1) Coarse Search: 섹션 레벨에서 상위 top_sections개 섹션을 찾습니다.
        2) Fine Search: 해당 섹션 내 청크들 중 상위 top_chunks개를 검색합니다.
        3) LLM을 통해 최종 답변을 생성합니다.
        """
        # Coarse Search (섹션 레벨)
        relevant_secs = hnsw_section_search(query, self.sections, beta=beta, top_k=top_sections)

        # Fine Search (청크 레벨)
        query_emb = embedding_model.get_embedding(query)
        best_chunks = hnsw_fine_search(query_emb, self.chunk_index, relevant_secs, top_k=top_chunks)

        # LLM 답변 생성
        prompt = self.build_prompt(query, best_chunks)
        answer_text = local_llm.generate(prompt, max_length=200)
        return answer_text

if __name__ == "__main__":
    import json
    import os

    # 섹션 정보와 청크 인덱스 파일 경로 설정 (환경에 맞게 수정)
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

    # 챗봇 인스턴스 생성
    chatbot = PDFChatBot(sections, chunk_index)
    print("Chatbot is ready. Enter your question below:")

    while True:
        query = input("Question (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        answer = chatbot.answer(query)
        print("Answer:", answer)