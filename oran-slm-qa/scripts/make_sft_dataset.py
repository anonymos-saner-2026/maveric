import argparse
from pathlib import Path
import random
from oran_qa.utils.io import read_jsonl, write_jsonl
from oran_qa.data.qa_synthesis import synthesize_qa_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=str, required=True, help="data/corpus/chunks.jsonl")
    ap.add_argument("--out", type=str, required=True, help="output dir e.g. data/sft")
    ap.add_argument("--train_ratio", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_per_chunk", type=int, default=2)
    args = ap.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks = list(read_jsonl(Path(args.chunks)))
    examples = []
    for ch in chunks:
        examples.extend(synthesize_qa_pairs(ch, max_pairs=args.max_per_chunk))

    random.shuffle(examples)
    n_train = int(len(examples) * args.train_ratio)

    train = examples[:n_train]
    valid = examples[n_train:]

    write_jsonl(train, out_dir / "train.jsonl")
    write_jsonl(valid, out_dir / "valid.jsonl")

    print(f"Total examples: {len(examples)}")
    print(f"Train: {len(train)} | Valid: {len(valid)}")


if __name__ == "__main__":
    main()
