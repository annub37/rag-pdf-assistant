from pathlib import Path

import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_path: str | Path) -> list[dict]:
    """
    Open a PDF and return extracted text for each page.

    Returns a list like:
    [
        {"page_number": 1, "text": "Welcome to ABC Corp..."},
        {"page_number": 2, "text": "Leave Policy: All employees..."},
    ]
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text().strip()
        pages.append({
            "page_number": page_num + 1,  # 1-based for humans
            "text": text,
        })

    doc.close()
    return pages
