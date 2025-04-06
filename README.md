# QueryDoc

This project is an example chatbot that analyzes PDF documents using an embedding model and generates LLM answers through a Coarse-to-Fine search (RAG) approach.

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

## Installation and Execution

1. Create a virtual environment and install libraries
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
(On Windows, activate with .\venv\Scripts\activate or a similar command.)

2.	Extract PDF & Split into Chunks
```bash
python scripts/pdf_extractor.py
python scripts/chunker.py
```
•	After execution, JSON files will be created in data/extracted/*.json and data/chunks/*.json.

3.	Build Embeddings
```bash
python scripts/build_index.py
```
•	This generates data/index/*_vectors.json.

4.	Generate Section Representative Vectors
```bash
python scripts/section_rep_builder.py
```
•	This creates files like sections_with_emb.json.


5. Directly Testing chatbot.py

To test the chatbot interactively, run:
```bash
python src/chatbot.py
```

Then type your question (e.g., “Explain chapter 1.”) and press Enter. Type “exit” to quit.

6.	Run the Chatbot Server
```bash
python app.py
```
    
• A FastAPI server will run at http://0.0.0.0:8000 (default port).

• You can send a JSON-formatted question to the POST /ask endpoint to receive an answer.

7. Example API Request
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Explain chapter 1."}'
```

## Key Libraries

• PyMuPDF (fitz): Extracts PDF text and table of contents (ToC).

• SentenceTransformers: Loads the intfloat/multilingual-e5-large embedding model.

• Transformers: Provides the local LLM (e.g., EXAONE-3.5-2.4B-Instruct).

• FastAPI: A simple REST API server.


## Notes

• Models such as intfloat/multilingual-e5-large and EXAONE-3.5-2.4B-Instruct may take some time to download the first time they are loaded.

• Since section content is complemented using the average of section chunk embeddings (without a summarization model), very long sections may result in reduced search accuracy. (Utilizing a summarization model may be considered in the future.)

