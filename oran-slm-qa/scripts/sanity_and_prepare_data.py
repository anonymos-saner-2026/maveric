#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sanity-check ORAN corpus chunks + prepare closed-book MCQ SFT data.

1) Corpus sanity check (chunks.jsonl):
   - length stats, doc distribution
   - heuristic noise detection (TOC lines, ETSI boilerplate, etc.)
   - random samples
   - write report JSON

2) Benchmark MCQ -> JSONL for SFT:
   - input formats supported:
        a) JSON list: [[q, [opt...], label], ...]
        b) JSONL: each line is JSON array [q, [opt...], label]
   - output JSONL: {"id","messages":[...]} (chat-style) or {"prompt","response"} (plain)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median


# ---------------------------
# Utilities
# ---------------------------

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}: {e}")

def write_jsonl(items, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def percentile(sorted_vals, p: float):
    """p in [0,100]"""
    if not sorted_vals:
        return None
    if p <= 0:
        return sorted_vals[0]
    if p >= 100:
        return sorted_vals[-1]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1


# ---------------------------
# Corpus sanity check
# ---------------------------

TOC_LINE_RE = re.compile(r"\.{10,}\s*\d+\s*$")
ETSI_BOILER_RE = re.compile(
    r"(Route des Lucioles|Sophia Antipolis|ETSI Search & Browse|All rights reserved|shall not be modified|"
    r"Notice of disclaimer|limitation of liability|Copyright Notification|Milestones listing|deliver repository)",
    re.IGNORECASE
)
PAGE_FOOTER_RE = re.compile(r"^\s*ETSI\s+(TS|TR)\s+\d+|^\s*ETSI\s*$", re.IGNORECASE)

ASCII_DIAGRAM_RE = re.compile(r"(\|__|\+--|---\+|==>|<==|\[={3,}|\]{3,})")


@dataclass
class NoiseFlags:
    toc_lines: int = 0
    boiler_lines: int = 0
    footer_hits: int = 0
    ascii_diagram_hits: int = 0

def analyze_text_noise(text: str) -> NoiseFlags:
    flags = NoiseFlags()
    if not text:
        return flags
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in lines:
        if TOC_LINE_RE.search(ln):
            flags.toc_lines += 1
        if ETSI_BOILER_RE.search(ln):
            flags.boiler_lines += 1
        if PAGE_FOOTER_RE.search(ln):
            flags.footer_hits += 1
        if ASCII_DIAGRAM_RE.search(ln):
            flags.ascii_diagram_hits += 1
    return flags


def corpus_sanity_check(
    corpus_path: Path,
    sample_k: int = 8,
    min_chars: int = 280,
    max_report_docs: int = 15,
    seed: int = 7
) -> dict:
    random.seed(seed)

    lens = []
    doc_counts = Counter()
    short_chunks = 0
    empty_chunks = 0

    # noise accumulators
    total_flags = NoiseFlags()
    noisy_examples = []  # store a few IDs

    chunks_cache = []  # for sampling
    for obj in read_jsonl(corpus_path):
        text = obj.get("text", "") or ""
        doc_id = obj.get("doc_id", "UNKNOWN")
        doc_counts[doc_id] += 1

        L = len(text)
        lens.append(L)
        if L == 0:
            empty_chunks += 1
        if L < min_chars:
            short_chunks += 1

        flags = analyze_text_noise(text)
        total_flags.toc_lines += flags.toc_lines
        total_flags.boiler_lines += flags.boiler_lines
        total_flags.footer_hits += flags.footer_hits
        total_flags.ascii_diagram_hits += flags.ascii_diagram_hits

        # mark as "noisy" if it contains a lot of toc/boiler/footer/diagram signals
        noise_score = flags.toc_lines + flags.boiler_lines + flags.footer_hits + flags.ascii_diagram_hits
        if noise_score >= 3 and len(noisy_examples) < 12:
            noisy_examples.append({
                "id": obj.get("id"),
                "doc_id": doc_id,
                "title": obj.get("title", ""),
                "noise_score": noise_score,
                "flags": flags.__dict__,
                "preview": (text[:260].replace("\n", " ") + " ...") if text else ""
            })

        chunks_cache.append(obj)

    lens_sorted = sorted(lens)
    stats = {
        "num_chunks": len(lens),
        "num_docs": len(doc_counts),
        "min_len": lens_sorted[0] if lens_sorted else 0,
        "p10_len": int(percentile(lens_sorted, 10) or 0),
        "p25_len": int(percentile(lens_sorted, 25) or 0),
        "median_len": int(percentile(lens_sorted, 50) or 0),
        "p75_len": int(percentile(lens_sorted, 75) or 0),
        "p90_len": int(percentile(lens_sorted, 90) or 0),
        "max_len": lens_sorted[-1] if lens_sorted else 0,
        "mean_len": int(mean(lens_sorted)) if lens_sorted else 0,
        "short_chunks_lt_min_chars": short_chunks,
        "empty_chunks": empty_chunks,
        "short_ratio": (short_chunks / len(lens)) if lens else 0.0,
        "empty_ratio": (empty_chunks / len(lens)) if lens else 0.0,
    }

    top_docs = doc_counts.most_common(max_report_docs)
    samples = []
    if chunks_cache:
        for obj in random.sample(chunks_cache, min(sample_k, len(chunks_cache))):
            text = obj.get("text", "") or ""
            samples.append({
                "id": obj.get("id"),
                "doc_id": obj.get("doc_id"),
                "title": obj.get("title", ""),
                "len": len(text),
                "preview": text[:600].replace("\n", " ") + (" ..." if len(text) > 600 else "")
            })

    report = {
        "corpus_path": str(corpus_path),
        "stats": stats,
        "top_docs_by_chunk_count": [{"doc_id": d, "chunks": c} for d, c in top_docs],
        "noise_totals": total_flags.__dict__,
        "noisy_examples": noisy_examples,
        "random_samples": samples,
    }
    return report


# ---------------------------
# Benchmark MCQ -> SFT JSONL
# ---------------------------

def load_mcq_any(path: Path):
    """
    Supports:
      - .json : a JSON list of records (each record can be [q, opts, label] or dict)
      - .jsonl: each line is JSON record (array or dict)
    """
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("JSON must be a list.")
        for rec in data:
            yield rec
    else:
        for rec in read_jsonl(path):
            yield rec


def normalize_mcq_record(rec, idx: int):
    """
    Accepts:
      - array: [question, [options...], label]
      - dict: {"question":..., "options":[...], "label":...} or {"q":...,"choices":...,"answer":...}
    Returns: (qid, question, options, label_str)
    """
    qid = f"mcq_{idx:06d}"

    if isinstance(rec, list) or isinstance(rec, tuple):
        if len(rec) < 3:
            raise ValueError(f"MCQ record (array) must have 3 items: [q, options, label]. Got: {rec}")
        question = str(rec[0]).strip()
        options = rec[1]
        label = rec[2]
    elif isinstance(rec, dict):
        question = rec.get("question") or rec.get("q") or rec.get("prompt")
        options = rec.get("options") or rec.get("choices") or rec.get("candidates")
        label = rec.get("label") or rec.get("answer") or rec.get("gold")
        qid = rec.get("id") or qid
    else:
        raise ValueError(f"Unsupported MCQ record type: {type(rec)}")

    if question is None or options is None or label is None:
        raise ValueError(f"Missing fields in record idx={idx}: {rec}")

    question = str(question).strip()
    if not isinstance(options, list) or len(options) < 2:
        raise ValueError(f"Options must be a list of >=2 strings. idx={idx}: {options}")
    options = [str(o).strip() for o in options]

    # labels in your sample are "1"/"2"/"3"/"4" (1-indexed)
    label_str = str(label).strip().strip('"').strip()
    # normalize common variants like "A"/"B"
    if label_str.upper() in ["A", "B", "C", "D", "E"]:
        label_str = str(["A", "B", "C", "D", "E"].index(label_str.upper()) + 1)

    # final validation
    try:
        li = int(label_str)
    except:
        raise ValueError(f"Label must be int-like (e.g., '3'). idx={idx}: {label_str}")
    if not (1 <= li <= len(options)):
        raise ValueError(f"Label out of range. idx={idx}: label={li}, options={len(options)}")

    return qid, question, options, str(li)


def build_mcq_prompt(question: str, options: list[str]) -> str:
    """
    Closed-book instruction: answer must be a single option number.
    """
    opt_lines = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])
    prompt = (
        "You are an expert on O-RAN specifications.\n"
        "Answer the following multiple-choice question using your internal knowledge only (closed-book).\n"
        "Respond with ONLY the option number (e.g., 1).\n\n"
        f"Question: {question}\n"
        f"Options:\n{opt_lines}\n\n"
        "Answer:"
    )
    return prompt


def make_sft_example_chat(qid: str, question: str, options: list[str], label: str):
    prompt = build_mcq_prompt(question, options)
    return {
        "id": qid,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": label}
        ]
    }


def make_sft_example_plain(qid: str, question: str, options: list[str], label: str):
    return {
        "id": qid,
        "prompt": build_mcq_prompt(question, options),
        "response": label
    }


def split_indices(n: int, valid_ratio: float, test_ratio: float, seed: int):
    idxs = list(range(n))
    random.Random(seed).shuffle(idxs)
    n_test = int(round(n * test_ratio))
    n_valid = int(round(n * valid_ratio))
    test = set(idxs[:n_test])
    valid = set(idxs[n_test:n_test + n_valid])
    train = set(idxs[n_test + n_valid:])
    return train, valid, test


def prepare_mcq_sft(
    benchmark_path: Path,
    out_dir: Path,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 7,
    format: str = "chat"  # "chat" or "plain"
) -> dict:
    records = list(load_mcq_any(benchmark_path))
    n = len(records)
    if n == 0:
        raise ValueError(f"No records found in {benchmark_path}")

    train_idx, valid_idx, test_idx = split_indices(n, valid_ratio, test_ratio, seed)

    maker = make_sft_example_chat if format == "chat" else make_sft_example_plain

    train, valid, test = [], [], []
    for i, rec in enumerate(records):
        qid, q, opts, label = normalize_mcq_record(rec, i)
        ex = maker(qid, q, opts, label)
        if i in test_idx:
            test.append(ex)
        elif i in valid_idx:
            valid.append(ex)
        else:
            train.append(ex)

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train_mcq.jsonl"
    valid_path = out_dir / "valid_mcq.jsonl"
    test_path = out_dir / "test_mcq.jsonl"
    write_jsonl(train, train_path)
    write_jsonl(valid, valid_path)
    write_jsonl(test, test_path)

    return {
        "benchmark_path": str(benchmark_path),
        "format": format,
        "num_total": n,
        "num_train": len(train),
        "num_valid": len(valid),
        "num_test": len(test),
        "train_path": str(train_path),
        "valid_path": str(valid_path),
        "test_path": str(test_path),
        "seed": seed,
        "valid_ratio": valid_ratio,
        "test_ratio": test_ratio
    }


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=str, default="data/corpus/chunks.jsonl", help="Path to chunks.jsonl")
    ap.add_argument("--report_out", type=str, default="outputs/corpus_report.json", help="Where to write report JSON")
    ap.add_argument("--sample_k", type=int, default=8)
    ap.add_argument("--min_chars", type=int, default=280)

    ap.add_argument("--benchmark", type=str, default=None,
                    help="MCQ benchmark file (.json or .jsonl). If provided, will also prepare SFT data.")
    ap.add_argument("--bench_out_dir", type=str, default="data/benchmarks",
                    help="Output dir for train_mcq.jsonl/valid_mcq.jsonl/test_mcq.jsonl")
    ap.add_argument("--valid_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--format", type=str, choices=["chat", "plain"], default="chat",
                    help="Output format for SFT: chat messages or plain prompt/response")

    args = ap.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        raise FileNotFoundError(corpus_path)

    report = corpus_sanity_check(
        corpus_path=corpus_path,
        sample_k=args.sample_k,
        min_chars=args.min_chars,
        seed=args.seed
    )
    report_path = Path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Print a concise human-readable summary
    s = report["stats"]
    print("\n=== Corpus sanity check ===")
    print("Corpus:", report["corpus_path"])
    print(f"Chunks: {s['num_chunks']}  Docs: {s['num_docs']}")
    print(f"Len chars: min={s['min_len']} p10={s['p10_len']} p50={s['median_len']} p90={s['p90_len']} max={s['max_len']}")
    print(f"Short(<{args.min_chars}): {s['short_chunks_lt_min_chars']} ({s['short_ratio']:.1%})  Empty: {s['empty_chunks']} ({s['empty_ratio']:.1%})")
    print("Noise totals:", report["noise_totals"])
    print("Top docs by chunks:")
    for d in report["top_docs_by_chunk_count"][:10]:
        print(f"  {d['doc_id']}: {d['chunks']}")

    print("\nRandom samples:")
    for ex in report["random_samples"]:
        print("-" * 80)
        print(ex["id"])
        print(ex["title"])
        print(ex["preview"])

    if report["noisy_examples"]:
        print("\nNoisy examples (first few):")
        for ex in report["noisy_examples"][:5]:
            print("-" * 80)
            print(ex["id"], "score=", ex["noise_score"], "flags=", ex["flags"])
            print(ex["preview"])

    print(f"\nReport written to: {report_path}")

    # Prepare benchmark if provided
    if args.benchmark:
        bench_path = Path(args.benchmark)
        if not bench_path.exists():
            raise FileNotFoundError(bench_path)

        info = prepare_mcq_sft(
            benchmark_path=bench_path,
            out_dir=Path(args.bench_out_dir),
            valid_ratio=args.valid_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            format=args.format
        )
        print("\n=== Prepared MCQ SFT data ===")
        print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
