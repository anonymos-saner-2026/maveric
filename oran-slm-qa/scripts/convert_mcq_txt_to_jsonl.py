import argparse
from pathlib import Path
import ast
import random
import json


def parse_line(line: str):
    """
    Expect each non-empty line to be a Python literal list/tuple like:

    ["Question text?", ["1. opt", "2. opt", "3. opt", "4. opt"], "3"]

    Returns dict:
      { "question": str, "options": [str...], "label": "1"-"4" }
    """
    obj = ast.literal_eval(line.strip())
    if not (isinstance(obj, (list, tuple)) and len(obj) == 3):
        raise ValueError(f"Bad sample format: {obj}")

    question = str(obj[0]).strip()
    options = list(obj[1])
    label = str(obj[2]).strip()

    if len(options) < 2:
        raise ValueError("Need at least 2 options")
    if label not in {"1", "2", "3", "4"}:
        # allow ints too
        if label.isdigit() and int(label) in [1, 2, 3, 4]:
            label = str(int(label))
        else:
            raise ValueError(f"Label must be 1-4, got: {label}")

    # Normalize option strings: remove leading "1. " etc if present
    norm_options = []
    for opt in options:
        s = str(opt).strip()
        # remove leading "{k}. " only if it matches current index style
        # e.g. "1. O-RAN.WG3" -> "O-RAN.WG3"
        if len(s) >= 3 and s[0].isdigit() and s[1] == "." and s[2] == " ":
            s = s[3:].strip()
        norm_options.append(s)

    return {"question": question, "options": norm_options, "label": label}


def maybe_shuffle_options(sample: dict, rng: random.Random):
    """
    Shuffle options to reduce position bias.
    Adjust label accordingly.
    """
    opts = list(sample["options"])
    y = int(sample["label"]) - 1  # 0-based

    idxs = list(range(len(opts)))
    rng.shuffle(idxs)

    new_opts = [opts[i] for i in idxs]
    new_y = idxs.index(y)  # where did the correct option go?

    sample["options"] = new_opts
    sample["label"] = str(new_y + 1)
    return sample


def read_samples(txt_path: Path):
    samples = []
    for ln in txt_path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        samples.append(parse_line(ln))
    return samples


def write_jsonl(items, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_txt", type=str, required=True, help="Input txt: one sample per line (python literal)")
    ap.add_argument("--out_dir", type=str, required=True, help="Output dir, e.g. data/mcq")
    ap.add_argument("--train_ratio", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shuffle_options", action="store_true", help="Shuffle options per sample (recommended)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    in_path = Path(args.in_txt)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = read_samples(in_path)
    if args.shuffle_options:
        samples = [maybe_shuffle_options(s, rng) for s in samples]

    rng.shuffle(samples)
    n_train = int(len(samples) * args.train_ratio)
    train = samples[:n_train]
    valid = samples[n_train:]

    write_jsonl(train, out_dir / "train.jsonl")
    write_jsonl(valid, out_dir / "valid.jsonl")

    print(f"Total: {len(samples)} | Train: {len(train)} | Valid: {len(valid)}")
    print(f"Wrote: {out_dir/'train.jsonl'}")
    print(f"Wrote: {out_dir/'valid.jsonl'}")


if __name__ == "__main__":
    main()
