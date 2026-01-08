#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


def build_text(example, eos: str):
    # chunks.jsonl has fields like: id, doc_id, section, title, page_start, page_end, text
    t = (example.get("text") or "").strip()
    title = (example.get("title") or "").strip()
    # Keep it simple: optional title prefix helps structure
    if title and title.lower() not in t.lower():
        t = f"{title}\n{t}"
    # Add EOS to separate chunks during LM training
    return {"text": t + "\n" + eos + "\n"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=str, default="data/corpus/chunks.jsonl")
    ap.add_argument("--out", type=str, default="data/dapt/tokenized")
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    ap.add_argument("--block_size", type=int, default=2048)
    ap.add_argument("--min_chars", type=int, default=300)
    ap.add_argument("--num_proc", type=int, default=8)
    args = ap.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        raise FileNotFoundError(corpus_path)

    out_dir = Path(args.out)
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    eos = tokenizer.eos_token or "</s>"

    ds = load_dataset("json", data_files=str(corpus_path), split="train")

    # basic filter: too short chunks are usually TOC/footer snippets
    def ok_len(ex):
        t = (ex.get("text") or "")
        return len(t) >= args.min_chars

    ds = ds.filter(ok_len, num_proc=args.num_proc)

    # build text with EOS separators
    ds = ds.map(lambda ex: build_text(ex, eos), remove_columns=[c for c in ds.column_names if c != "text"],
                num_proc=args.num_proc)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], add_special_tokens=False)

    tok = ds.map(tokenize_fn, batched=True, num_proc=args.num_proc, remove_columns=["text"])

    # pack to fixed-length blocks
    block_size = args.block_size

    def group_texts(examples):
        # Concatenate
        concatenated = {}
        for k in examples.keys():
            concatenated[k] = sum(examples[k], [])
        total_len = len(concatenated["input_ids"])
        # Drop remainder
        total_len = (total_len // block_size) * block_size
        result = {}
        for k, v in concatenated.items():
            v = v[:total_len]
            result[k] = [v[i : i + block_size] for i in range(0, total_len, block_size)]
        # labels = input_ids for causal LM
        result["labels"] = result["input_ids"].copy()
        return result

    packed = tok.map(group_texts, batched=True, num_proc=args.num_proc)

    packed.save_to_disk(str(out_dir))
    meta = {
        "corpus": str(corpus_path),
        "model": args.model,
        "block_size": block_size,
        "min_chars": args.min_chars,
        "num_examples": len(packed),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved tokenized+packed DAPT dataset to: {out_dir}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
