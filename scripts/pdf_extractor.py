# scripts/pdf_extractor.py

import os
import json
import fitz  # PyMuPDF
from typing import Dict, Any

def extract_pdf_content(pdf_path: str) -> Dict[str, Any]:
    """
    pdf_path의 PDF에서 다음을 추출:
    - 목차 (getToC)
    - 각 페이지 텍스트

    return {
        "file_path": pdf_path,
        "toc": [(level, title, start_page), ...],
        "pages_text": ["...", "...", ...],
        ...
    }
    """
    doc = fitz.open(pdf_path)
    toc = doc.get_toc(simple=True)  # [(level, title, page_number), ...]

    pages_text = []
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        text = page.get_text("text")
        pages_text.append(text)

    return {
        "file_path": pdf_path,
        "toc": toc,
        "pages_text": pages_text
    }

def save_extracted_content(content: Dict[str, Any], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(content, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # 예시 사용
    pdf_folder = "data"
    output_folder = "data/extracted"
    os.makedirs(output_folder, exist_ok=True)

    for fname in os.listdir(pdf_folder):
        if fname.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, fname)
            extracted_data = extract_pdf_content(pdf_path)

            base_name = os.path.splitext(fname)[0]
            output_json = os.path.join(output_folder, f"{base_name}.json")
            save_extracted_content(extracted_data, output_json)

    print("PDF Extraction Complete.")