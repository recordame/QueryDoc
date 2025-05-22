# scripts/pdf_extractor.py

import os
import json
import sys
import fitz  # PyMuPDF
from typing import Dict, Any, List
import pdfplumber

def build_sections_from_toc(toc: List[List], total_pages: int) -> List[Dict[str, Any]]:
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
                        "page": page.page_number,
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
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    pages_text = [doc[i].get_text("text") for i in range(total_pages)]
    
    toc = doc.get_toc(simple=True)  # get_toc 사용
    
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
    # 원본 PDF 파일을 저장할 별도 디렉토리 설정 (예: data/original)
    pdf_folder = os.path.join("data", "original")
    output_folder = os.path.join("data", "extracted")
    os.makedirs(output_folder, exist_ok=True)

    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"[ERROR] No PDF files found in '{pdf_folder}'.")
        sys.exit(1)
    
    # 여러 파일이 있으면 사용자에게 목록을 보여주고 하나만 선택하도록 함
    if len(pdf_files) > 1:
        print("Multiple PDF files found:")
        for idx, fname in enumerate(pdf_files):
            print(f"{idx+1}. {fname}")
        selection = input("Select a file by number: ")
        try:
            selection_idx = int(selection) - 1
            if selection_idx < 0 or selection_idx >= len(pdf_files):
                print("Invalid selection.")
                sys.exit(1)
            selected_file = pdf_files[selection_idx]
        except ValueError:
            print("Invalid input.")
            sys.exit(1)
    else:
        selected_file = pdf_files[0]

    pdf_path = os.path.join(pdf_folder, selected_file)
    print(f"Processing file: {selected_file}")
    extracted_data = extract_pdf_content(pdf_path)

    base_name = os.path.splitext(selected_file)[0]
    output_json = os.path.join(output_folder, f"{base_name}.json")
    save_extracted_content(extracted_data, output_json)
    print(f"Processed {selected_file}: Found {len(extracted_data['sections'])} sections.")

    # For a single PDF, also save merged sections as sections.json.
    sections_output = os.path.join(output_folder, "sections.json")
    with open(sections_output, 'w', encoding='utf-8') as f:
        json.dump(extracted_data["sections"], f, ensure_ascii=False, indent=2)
    print(f"Sections saved to {sections_output}")
    
    print("PDF Extraction Complete.")