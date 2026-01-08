# scripts/build_corpus.py
import argparse
from pathlib import Path

from oran_qa.data.parse_pdf import parse_pdf_to_pages
from oran_qa.data.chunking import chunk_pages_to_sections
from oran_qa.utils.io import write_jsonl


def iter_pdfs(raw_dir: Path):
    for p in sorted(raw_dir.glob("*.pdf")):
        yield p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=str, required=True, help="Directory containing PDF specs")
    ap.add_argument("--out", type=str, required=True, help="Output JSONL path")
    ap.add_argument("--max_chars", type=int, default=2200)
    ap.add_argument("--overlap_chars", type=int, default=300)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_chunks = []
    pdfs = list(iter_pdfs(raw_dir))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {raw_dir}")

    for pdf_path in pdfs:
        doc_id = pdf_path.stem
        pages = parse_pdf_to_pages(pdf_path)
        chunks = chunk_pages_to_sections(
            pages,
            doc_id=doc_id,
            max_chars=args.max_chars,
            overlap_chars=args.overlap_chars,
        )
        all_chunks.extend(chunks)
        print(f"✓ {pdf_path.name}: pages={len(pages)} chunks={len(chunks)}")

    write_jsonl(all_chunks, out_path)
    print(f"\nDone. Wrote {len(all_chunks)} chunks to {out_path}")


if __name__ == "__main__":
    main()
