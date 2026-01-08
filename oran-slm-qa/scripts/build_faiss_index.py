import argparse
from pathlib import Path
from oran_qa.utils.io import read_jsonl
from oran_qa.retrieval.faiss_index import build_and_save_faiss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=str, required=True)
    ap.add_argument("--index_dir", type=str, required=True)
    ap.add_argument("--embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    chunks = list(read_jsonl(Path(args.chunks)))
    out_dir = Path(args.index_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    build_and_save_faiss(chunks, out_dir, embed_model=args.embed_model)
    print(f"Saved index to {out_dir}")


if __name__ == "__main__":
    main()
