from pathlib import Path
import json
import faiss
from oran_qa.retrieval.embed import Embedder


def build_and_save_faiss(chunks, out_dir: Path, embed_model: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    embedder = Embedder(embed_model)

    texts = [c["text"] for c in chunks]
    vecs = embedder.encode(texts)

    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)

    faiss.write_index(index, str(out_dir / "index.faiss"))
    (out_dir / "chunks.jsonl").write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in chunks),
        encoding="utf-8",
    )
    (out_dir / "meta.json").write_text(
        json.dumps({"embed_model": embed_model, "dim": dim, "count": len(chunks)}, indent=2),
        encoding="utf-8",
    )
