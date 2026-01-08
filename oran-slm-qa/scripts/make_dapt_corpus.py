import argparse
from pathlib import Path
import random
import re
from oran_qa.utils.io import read_jsonl

_WS = re.compile(r"\s+")

def clean_text(s: str) -> str:
    s = s.replace("\x00", " ")
    s = _WS.sub(" ", s).strip()
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=str, required=True, help="data/corpus/chunks.jsonl")
    ap.add_argument("--out_dir", type=str, required=True, help="e.g. data/dapt")
    ap.add_argument("--train_ratio", type=float, default=0.98)
    ap.add_argument("--min_chars", type=int, default=300)
    ap.add_argument("--max_chars", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dedup", action="store_true", help="Deduplicate identical lines")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks = list(read_jsonl(Path(args.chunks)))

    texts = []
    for ch in chunks:
        t = clean_text(ch.get("text", ""))
        if len(t) < args.min_chars:
            continue
        if len(t) > args.max_chars:
            t = t[:args.max_chars].rsplit(" ", 1)[0].strip()
        if t:
            texts.append(t)

    if args.dedup:
        # order-preserving dedup
        seen = set()
        uniq = []
        for t in texts:
            if t not in seen:
                seen.add(t)
                uniq.append(t)
        texts = uniq

    rng.shuffle(texts)
    n_train = int(len(texts) * args.train_ratio)
    train = texts[:n_train]
    valid = texts[n_train:] if n_train < len(texts) else texts[-max(1, len(texts)//100):]

    (out_dir / "train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (out_dir / "valid.txt").write_text("\n".join(valid) + "\n", encoding="utf-8")

    print(f"Total lines: {len(texts)}")
    print(f"Train lines: {len(train)} -> {out_dir/'train.txt'}")
    print(f"Valid lines: {len(valid)} -> {out_dir/'valid.txt'}")

if __name__ == "__main__":
    main()
