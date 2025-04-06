# scripts/pdf_extractor.py

import os
import json
import fitz  # PyMuPDF
from typing import Dict, Any, List

def extract_pdf_content(pdf_path: str) -> Dict[str, Any]:
    """
    pdf_path의 PDF에서 다음을 추출:
    - 목차 (getToC)
    - 각 페이지 텍스트
    - 섹션 정보: 목차를 바탕으로 각 섹션의 시작/끝 페이지와 레벨을 산출

    반환 예시:
    {
        "file_path": pdf_path,
        "toc": [(level, title, start_page), ...],
        "pages_text": ["...", "...", ...],
        "sections": [
             {"title": "1장 개요", "start_page": 1, "end_page": 5, "level": 1},
             {"title": "2장 설치방법", "start_page": 6, "end_page": 15, "level": 1},
             ...
        ]
    }
    """
    doc = fitz.open(pdf_path)
    toc = doc.get_toc(simple=True)  # [(level, title, page_number), ...]
    
    pages_text = []
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        text = page.get_text("text")
        pages_text.append(text)
    
    sections = build_sections(toc, len(doc))
    
    return {
        "file_path": pdf_path,
        "toc": toc,
        "pages_text": pages_text,
        "sections": sections
    }

def build_sections(toc: List[List], total_pages: int) -> List[Dict[str, Any]]:
    """
    toc: [(level, title, start_page), ...]
    total_pages: 전체 페이지 수
    각 항목에 대해, 다음 항목의 start_page - 1을 end_page로 설정하고,
    마지막 항목은 total_pages로 지정한다.
    """
    sections = []
    for i, entry in enumerate(toc):
        level, title, start_page = entry
        if i < len(toc) - 1:
            next_start = toc[i + 1][2]
            end_page = next_start - 1
        else:
            end_page = total_pages
        sections.append({
            "title": title,
            "start_page": start_page,
            "end_page": end_page,
            "level": level
        })
    return sections

def save_extracted_content(content: Dict[str, Any], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(content, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    pdf_folder = "data"
    output_folder = "data/extracted"
    os.makedirs(output_folder, exist_ok=True)

    processed_sections = []
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    
    for fname in pdf_files:
        pdf_path = os.path.join(pdf_folder, fname)
        extracted_data = extract_pdf_content(pdf_path)

        base_name = os.path.splitext(fname)[0]
        output_json = os.path.join(output_folder, f"{base_name}.json")
        save_extracted_content(extracted_data, output_json)
        
        # 수동으로 섹션 정보 파일을 생성할 경우, 하나의 문서에 대해 sections.json으로 저장
        processed_sections.append(extracted_data["sections"])
    
    # 만약 PDF가 하나라면 sections.json 파일로 저장 (여러 개일 경우, 개별 파일로 관리하는 것이 좋음)
    if len(processed_sections) == 1:
        sections_output = os.path.join(output_folder, "sections.json")
        with open(sections_output, 'w', encoding='utf-8') as f:
            json.dump(processed_sections[0], f, ensure_ascii=False, indent=2)
        print(f"Sections saved to {sections_output}")
    else:
        print("Multiple PDF files processed. Please check individual section files.")
    
    print("PDF Extraction Complete.")