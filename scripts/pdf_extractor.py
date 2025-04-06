# scripts/pdf_extractor.py

import os
import json
import fitz  # PyMuPDF
from typing import Dict, Any, List
import pdfplumber

def build_sections_from_toc(toc: List[List], total_pages: int) -> List[Dict[str, Any]]:
    """
    Built-in TOC를 이용해 섹션 정보를 생성합니다.
    toc: [(level, title, start_page), ...]
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
            "method": "TOC"
        })
    return sections

def build_sections_from_layout(pdf_path: str, font_size_threshold: float = 14.0) -> List[Dict[str, Any]]:
    """
    pdfplumber를 이용해 페이지별 단어 정보를 추출한 후, 
    폰트 크기가 font_size_threshold 이상이고 키워드("Chapter", "Section", "Part", "장", "절")가 포함된 단어들을 제목 후보로 사용하여 섹션 정보를 생성합니다.
    """

    candidate_headings = []
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        for page in pdf.pages:
            words = page.extract_words(extra_attrs=["size", "fontname"])
            for word in words:
                size = word.get("size", 0)
                text = word.get("text", "")
                if size >= font_size_threshold and any(kw.lower() in text.lower() for kw in ["chapter", "section", "part", "장", "절"]):
                    candidate_headings.append({
                        "page": page.page_number,  # pdfplumber는 1-based page_number 제공
                        "text": text,
                        "font_size": size
                    })
        candidate_headings.sort(key=lambda x: x["page"])
    
    sections = []
    if candidate_headings:
        for i, heading in enumerate(candidate_headings):
            start_page = heading["page"]
            if i < len(candidate_headings) - 1:
                end_page = candidate_headings[i+1]["page"] - 1
            else:
                end_page = total_pages
            sections.append({
                "title": heading["text"],
                "start_page": start_page,
                "end_page": end_page,
                "method": "Layout"
            })
    return sections

def extract_pdf_content(pdf_path: str) -> Dict[str, Any]:
    """
    PDF에서 다음을 추출합니다:
      - 내장 TOC (PyMuPDF)
      - 각 페이지 텍스트
      - 섹션 정보: 내장 TOC가 있으면 TOC 기반, 없으면 레이아웃 분석 기반,
        최종적으로 모두 없으면 페이지 기반 섹션(각 페이지를 섹션으로)으로 대체.
    """
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    pages_text = []
    for page_idx in range(total_pages):
        page = doc[page_idx]
        text = page.get_text("text")
        pages_text.append(text)
    
    toc = doc.get_toc(simple=True)  # 수정된 부분: get_toc 사용
    
    if toc:
        sections = build_sections_from_toc(toc, total_pages)
    else:
        sections = build_sections_from_layout(pdf_path)
        if not sections:
            sections = [{"title": f"Page {i+1}", "start_page": i+1, "end_page": i+1, "method": "Page-based"} for i in range(total_pages)]
    
    return {
        "file_path": pdf_path,
        "toc": toc,
        "pages_text": pages_text,
        "sections": sections
    }

def save_extracted_content(content: Dict[str, Any], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(content, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    pdf_folder = "data"
    output_folder = "data/extracted"
    os.makedirs(output_folder, exist_ok=True)

    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    processed_sections = []  # 각 PDF의 섹션 정보를 저장할 리스트

    for fname in pdf_files:
        pdf_path = os.path.join(pdf_folder, fname)
        extracted_data = extract_pdf_content(pdf_path)

        base_name = os.path.splitext(fname)[0]
        output_json = os.path.join(output_folder, f"{base_name}.json")
        save_extracted_content(extracted_data, output_json)
        print(f"Processed {fname}: Found {len(extracted_data['sections'])} sections.")
        
        processed_sections.append(extracted_data["sections"])

    # 만약 PDF가 하나라면 sections.json 파일로 저장
    if len(processed_sections) == 1:
        sections_output = os.path.join(output_folder, "sections.json")
        with open(sections_output, 'w', encoding='utf-8') as f:
            json.dump(processed_sections[0], f, ensure_ascii=False, indent=2)
        print(f"Sections saved to {sections_output}")
    else:
        print("Multiple PDF files processed. Please check individual section files.")
    
    print("PDF Extraction Complete.")