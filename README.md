# QueryDoc

이 프로젝트는 **KURE-v1** 임베딩 모델을 사용해 PDF 문서를 분석하고, Coarse-to-Fine 검색(RAG) 방식으로 LLM 답변을 생성하는 챗봇 예시입니다.

## QueryDoc
```bash
my_kure_chatbot/
├─ scripts/
│   ├─ pdf_extractor.py
│   ├─ chunker.py
│   ├─ build_index.py
│   └─ section_rep_builder.py
├─ src/
│   ├─ inference/
│   │   ├─ kure_embedding_model.py
│   │   └─ llm_model.py
│   ├─ search/
│   │   ├─ section_coarse_search.py
│   │   ├─ fine_search.py
│   │   └─ vector_search.py
│   ├─ chatbot.py
│   └─ utils/
│       ├─ init.py
│       └─ text_cleaning.py
├─ data/
│   ├─ sample.pdf
│   ├─ chunks/
│   └─ index/
├─ app.py
├─ requirements.txt
└─ README.md
```

## 설치 및 실행

1. **가상 환경 생성 및 라이브러리 설치**  
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
(Windows 환경이라면 .\venv\Scripts\activate 등으로 활성화)

2.	PDF 추출 & 청크 분할
```bash
python scripts/pdf_extractor.py
python scripts/chunker.py
```
•	실행 후, data/extracted/*.json과 data/chunks/*.json이 생성됩니다.

3.	임베딩 빌드
```bash
python scripts/build_index.py
```
•	data/index/*_vectors.json이 생성됩니다.

4.	섹션 대표벡터 생성
```bash
python scripts/section_rep_builder.py
```
•	sections_with_emb.json 등이 생성됩니다.

5.	챗봇 서버 실행
```bash
python app.py
```
    
•	FastAPI 서버가 http://0.0.0.0:8000(기본 포트)에서 동작합니다.

•	POST /ask 엔드포인트에 JSON 형식으로 질문을 전송하면 답변을 받을 수 있습니다.

예시 API 요청
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "설치 방법이 뭔가요?"}'
```

주요 라이브러리

•	PyMuPDF (fitz): PDF 텍스트와 ToC 추출

•	SentenceTransformers: KURE-v1 임베딩 모델 로딩

•	Transformers: 로컬 LLM (예: EXAONE-3.5-2.4B-Instruct)

•	FastAPI: 간단한 REST API 서버

주의사항

•	KURE-v1, EXAONE-3.5-2.4B-Instruct 등은 처음 로드 시 모델 파일을 다운로드하므로 다소 시간이 걸릴 수 있습니다.

•	Summarization 모델 없이 섹션 청크 임베딩 평균 방식으로 섹션 내용을 보완하기 때문에, 섹션이 매우 긴 경우 검색 정확도가 떨어질 수 있습니다. (추후 Summarization 모델 활용 가능)

