# pdf extraction using PyMuPDF and saving as json files
import fitz  # PyMuPDF
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#BLOCK 1: extract pages from PDF and save as JSONL
def extract_pdf_pages(pdf_path: str):
    """Extract pages from PDF using PyMuPDF"""
    pages = []
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_data = {
            "page_number": page_num + 1,
            "text": page.get_text(),
            "blocks": page.get_text("blocks")
        }
        pages.append(page_data)
    doc.close()
    logger.info(f"Extracted {len(pages)} pages from {pdf_path}")
    return pages

#BLOCK 2: write pages to JSONL file
def write_pages_jsonl(pages, out_path: str):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for p in pages:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    logger.info("Wrote pages JSONL to %s", out_path)

#BLOCK 3: main function to run the extraction
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--pdf", default="data/qatar.pdf", help="Path to PDF")
    p.add_argument("--out", default="data/pages.jsonl", help="Output JSONL")
    args = p.parse_args()

    pages = extract_pdf_pages(args.pdf)
    write_pages_jsonl(pages, args.out)
