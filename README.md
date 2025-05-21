# QueryDoc

This project is an example chatbot that analyzes PDF documents using an embedding model and generates LLM answers through a Coarse-to-Fine search (RAG) approach.

## QueryDoc
```bash
QueryDoc/
├─ scripts/
│   ├─ pdf_extractor.py
│   ├─ chunker.py
│   ├─ build_index.py
│   └─ section_rep_builder.py
├─ src/
│   ├─ inference/
│   │   ├─ embedding_model.py
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
│   ├─ chunks/
│   ├─ index/
│   └─ original/
│       └─ sample.pdf
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

•	Then type your question (e.g., “Explain Chapter 1.”) and press Enter. 

•	Type “exit” to quit.

The system prompt used by the chatbot can be customized by editing `DEFAULT_SYSTEM_PROMPT` in `src/chatbot.py` or by passing a custom prompt when creating a `PDFChatBot` instance.

6.	Run the Chatbot Server
```bash
python app.py
```
    
• A FastAPI server will run at http://0.0.0.0:8000 (default port).

• You can send a JSON-formatted question to the POST /ask endpoint to receive an answer.

7. Launch the Web Demo
```bash
python web_demo.py
```

• A browser window will appear allowing you to upload a PDF and edit the system prompt before asking questions.

8. Example API Request
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What's egocentric AI agent?"}'
```

8. Run the Web Demo
```bash
python web_demo.py
```
• Use the login form (default credentials: admin / password) and upload a PDF. You can modify the "System Prompt" field to test different instructions. Uploaded files are stored in `data/user_uploads/<username>`.

The default prompt used by the chatbot is defined as `DEFAULT_SYSTEM_PROMPT` in `src/chatbot.py`. You may edit this constant or pass your own prompt when creating `PDFChatBot`.

## Key Libraries

• PyMuPDF (fitz): Extracts PDF text and table of contents (ToC).

• SentenceTransformers: Loads the e5-large-v2 embedding model.

• Transformers: Provides the local LLM (e.g., EXAONE-Deep-2.4B).

• FastAPI: A simple REST API server.


• Gradio: Provides the interactive web demo.



## Notes

• Models such as e5-large-v2 and EXAONE-Deep-2.4B may take some time to download the first time they are loaded.

• Since section content is complemented using the average of section chunk embeddings (without a summarization model), very long sections may result in reduced search accuracy. (Utilizing a summarization model may be considered in the future.)

