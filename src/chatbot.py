# src/chatbot.py

from src.search.section_coarse_search import coarse_search_sections
from src.search.fine_search import fine_search_chunks
from src.inference.embedding_model import embedding_model
from src.inference.llm_model import local_llm

class PDFChatBot:
    def __init__(self, sections, chunk_index):
        """
        sections: 섹션 정보 리스트 (title, title_emb, avg_chunk_emb 포함)
        chunk_index: 청크 임베딩 리스트
        """
        self.sections = sections
        self.chunk_index = chunk_index

    def build_prompt(self, user_query, retrieved_chunks):
        """
        LLM에게 전달할 프롬프트를 구성.
        retrieved_chunks: [{"embedding": [...], "metadata": {...}}, ...]
        """
        context_parts = []
        for item in retrieved_chunks:
            meta = item["metadata"]
            section_t = meta["section_title"]
            content = meta["content"]
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
        1) Coarse: 섹션 레벨에서 top_sections개 선별
        2) Fine: 해당 섹션 내 청크 top_chunks개
        3) LLM에 프롬프트 전달해 응답 생성
        """
        # Coarse Search (섹션 레벨)
        relevant_secs = coarse_search_sections(query, self.sections, beta=beta, top_k=top_sections)

        # Fine Search (청크 레벨)
        query_emb = kure_embedding_model.get_embedding(query)
        best_chunks = fine_search_chunks(query_emb, self.chunk_index, relevant_secs, top_k=top_chunks)

        # LLM 답변 생성
        prompt = self.build_prompt(query, best_chunks)
        answer_text = local_llm.generate(prompt, max_length=200)
        return answer_text