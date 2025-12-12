#chunking pages into smaller pieces and writing chunks to chunks.jsonl

import json
from pathlib import Path
from typing import List

# BLOCK 1: chunking text into smaller pieces by paragraph with overlap
def chunk_text(text: str, max_chars: int = 2000, overlap: int = 200) -> List[str]:
    if not text:
        return []
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    cur = ""
    for p in paragraphs:
        if len(cur) + len(p) <= max_chars:
            cur = (cur + "\n\n" + p).strip() if cur else p
        else:
            chunks.append(cur)
            cur = p
    if cur:
        chunks.append(cur)
    # Add overlap between successive chunks
    out = []
    for i, c in enumerate(chunks):
        if i == 0:
            out.append(c)
        else:
            prev_tail = out[-1][-overlap:] if len(out[-1]) > overlap else out[-1]
            out.append(prev_tail + "\n\n" + c)
    return out

# BLOCK 2: reading pages.jsonl and writing chunks.jsonl
def pages_to_chunks(pages_jsonl: str, out_jsonl: str, doc_id: str = "qatar"):
    out_path = Path(out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(pages_jsonl, "r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            obj = json.loads(line)
            page_no = obj.get("page")
            text = obj.get("text", "")
            c_list = chunk_text(text)
            for i, c in enumerate(c_list):
                chunk_obj = {
                    "doc": doc_id,
                    "page": page_no,
                    "chunk_id": f"{page_no}_{i}",
                    "text": c
                }
                fout.write(json.dumps(chunk_obj, ensure_ascii=False) + "\n")
                count += 1
    print(f"Wrote {count} chunks to {out_jsonl}")

#BLOCK 3: main function to run the chunking
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--pages", default="data/pages.jsonl")
    p.add_argument("--out", default="data/chunks.jsonl")
    args = p.parse_args()
    pages_to_chunks(args.pages, args.out)
