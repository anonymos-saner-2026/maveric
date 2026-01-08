from pathlib import Path
import json
import faiss
from oran_qa.retrieval.embed import Embedder


class FaissRetriever:
    def __init__(self, index_dir: str, embed_model: str, top_k: int = 5):
        self.index_dir = Path(index_dir)
        self.top_k = top_k
        self.embedder = Embedder(embed_model)

        self.index = faiss.read_index(str(self.index_dir / "index.faiss"))

        self.chunks = []
        with (self.index_dir / "chunks.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                self.chunks.append(json.loads(line))

    def search(self, query: str):
        qv = self.embedder.encode([query])
        scores, idxs = self.index.search(qv, self.top_k)
        out = []
        for s, i in zip(scores[0].tolist(), idxs[0].tolist()):
            if i < 0:
                continue
            ch = dict(self.chunks[i])
            ch["score"] = float(s)
            out.append(ch)
        return out
