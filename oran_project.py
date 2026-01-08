#!/usr/bin/env python3
"""
bootstrap_oran_slm_qa_repo.py

Run:
  python bootstrap_oran_slm_qa_repo.py --path oran-slm-qa
  python bootstrap_oran_slm_qa_repo.py --path oran-slm-qa --force

This creates the full folder structure + writes all files.
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import textwrap


def write_file(path: Path, content: str, force: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        print(f"skip (exists): {path}")
        return
    path.write_text(content, encoding="utf-8")
    print(f"write: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=str, default="oran-slm-qa", help="Repo root folder to create")
    ap.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = ap.parse_args()

    root = Path(args.path).resolve()
    root.mkdir(parents=True, exist_ok=True)

    files: dict[str, str] = {}

    # -------------------------
    # Top-level files
    # -------------------------
    files["README.md"] = textwrap.dedent(
        """\
        # oran-slm-qa

        End-to-end template to fine-tune a Small Language Model (SLM) for Question Answering over O-RAN specifications.

        ## What you get
        - Download scripts for publicly available O-RAN PAS PDFs hosted by ETSI (plus optional custom URL list).
        - PDF parsing + structure-ish chunking.
        - Heuristic QA + evidence generation for SFT.
        - (Q)LoRA fine-tuning with HuggingFace + PEFT.
        - FAISS index building + simple RAG chat CLI.

        ## Notes on licensing
        This repo only downloads public PDFs by default. If you have access to other specs (e.g., behind login),
        provide your own URL list via --url_file and ensure you comply with the relevant license terms.

        ## Quickstart
        ```bash
        pip install -r requirements.txt

        python scripts/download_oran_specs.py --out data/raw_specs
        python scripts/build_corpus.py --raw_dir data/raw_specs --out data/corpus/chunks.jsonl
        python scripts/make_sft_dataset.py --chunks data/corpus/chunks.jsonl --out data/sft --train_ratio 0.95

        python scripts/train_lora.py --config configs/train_lora.yaml

        python scripts/build_faiss_index.py --chunks data/corpus/chunks.jsonl --index_dir data/index
        python scripts/chat_rag.py --config configs/rag.yaml
        ```
        """
    )

    files["requirements.txt"] = textwrap.dedent(
        """\
        # Core
        numpy>=1.26
        tqdm>=4.66
        pyyaml>=6.0
        regex>=2024.5.15

        # Data
        pandas>=2.2
        pypdf>=5.0.0

        # HF stack
        torch>=2.2
        transformers>=4.45
        datasets>=2.20
        accelerate>=0.33
        peft>=0.12
        bitsandbytes>=0.43

        # Optional trainer helpers
        trl>=0.11.0

        # Retrieval
        sentence-transformers>=3.0.1
        faiss-cpu>=1.8.0

        # CLI niceties
        rich>=13.7
        """
    )

    files["pyproject.toml"] = textwrap.dedent(
        """\
        [project]
        name = "oran-slm-qa"
        version = "0.1.0"
        description = "Fine-tune small language models for QA on O-RAN specifications (with evidence-grounded RAG)."
        requires-python = ">=3.10"

        [tool.setuptools.packages.find]
        where = ["src"]
        """
    )

    files[".gitignore"] = textwrap.dedent(
        """\
        __pycache__/
        *.pyc
        .venv/
        .env
        data/
        outputs/
        wandb/
        runs/
        *.log
        .DS_Store
        """
    )

    # -------------------------
    # Configs
    # -------------------------
    files["configs/data.yaml"] = textwrap.dedent(
        """\
        raw_dir: "data/raw_specs"
        corpus_out: "data/corpus/chunks.jsonl"

        chunking:
          max_chars: 2200
          overlap_chars: 300

        pdf_parse:
          keep_page_numbers: true
        """
    )

    files["configs/train_lora.yaml"] = textwrap.dedent(
        """\
        # Base model: choose your SLM (examples: Qwen2.5-1.5B-Instruct, Llama-3.2-1B-Instruct, Phi-3-mini, etc.)
        base_model: "Qwen/Qwen2.5-1.5B-Instruct"

        # Data
        train_file: "data/sft/train.jsonl"
        eval_file: "data/sft/valid.jsonl"

        # Output
        output_dir: "outputs/oran-slm-lora"
        save_steps: 200
        eval_steps: 200

        # Training
        max_seq_len: 2048
        per_device_train_batch_size: 2
        per_device_eval_batch_size: 2
        gradient_accumulation_steps: 8
        num_train_epochs: 1
        learning_rate: 2.0e-4
        warmup_ratio: 0.03
        weight_decay: 0.0
        logging_steps: 20

        # QLoRA
        use_4bit: true
        lora_r: 16
        lora_alpha: 32
        lora_dropout: 0.05
        target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]

        # Mixed precision
        bf16: true
        fp16: false

        # Prompting
        prompt_style: "evidence_qa_v1"
        """
    )

    files["configs/rag.yaml"] = textwrap.dedent(
        """\
        model:
          base_model: "Qwen/Qwen2.5-1.5B-Instruct"
          lora_dir: "outputs/oran-slm-lora"

        retrieval:
          index_dir: "data/index"
          embed_model: "sentence-transformers/all-MiniLM-L6-v2"
          top_k: 5

        generation:
          max_new_tokens: 256
          temperature: 0.2
        """
    )

    # -------------------------
    # Scripts
    # -------------------------
    files["scripts/download_oran_specs.py"] = textwrap.dedent(
        """\
        import argparse
        import time
        import json
        from pathlib import Path
        import requests

        # Default: publicly available ETSI PAS PDFs that adopt O-RAN Alliance specs.
        # You can add/remove URLs freely.
        DEFAULT_URLS = [
            ("ETSI_TS_103_982_O-RAN_Architecture_Description_v8.0.0_2024-01",
             "https://www.etsi.org/deliver/etsi_ts/103900_103999/103982/08.00.00_60/ts_103982v080000p.pdf"),
            ("ETSI_TS_103_983_A1_General_Aspects_v3.1.0_2024-01",
             "https://www.etsi.org/deliver/etsi_ts/103900_103999/103983/03.01.00_60/ts_103983v030100p.pdf"),
            ("ETSI_TS_103_985_A1_Use_Cases_Requirements_v1.1.0_2024-01",
             "https://www.etsi.org/deliver/etsi_ts/103900_103999/103985/01.01.00_60/ts_103985v010100p.pdf"),
            ("ETSI_TS_103_986_A1_Transport_Protocol_v2.1.0_2024-01",
             "https://www.etsi.org/deliver/etsi_ts/103900_103999/103986/02.01.00_60/ts_103986v020100p.pdf"),
            ("ETSI_TS_103_987_A1_Application_Protocol_v4.0.0_2024-01",
             "https://www.etsi.org/deliver/etsi_ts/103900_103999/103987/04.00.00_60/ts_103987v040000p.pdf"),
            ("ETSI_TS_104_023_O-RAN_Fronthaul_Management_Plane_v12.00.01_2024-05",
             "https://www.etsi.org/deliver/etsi_ts/104000_104099/104023/12.00.01_60/ts_104023v120001p.pdf"),
            ("ETSI_TR_104_037_O-RAN_Use_Cases_Analysis_Report_v12.0.0_2025-04",
             "https://www.etsi.org/deliver/etsi_tr/104000_104099/104037/12.00.00_60/tr_104037v120000p.pdf"),
        ]


        def read_url_file(path: str):
            \"\"\"
            url_file supports either:
              - json list: [{"name": "...", "url": "..."}, ...]
              - plain text: one url per line (name will be derived)
            \"\"\"
            p = Path(path)
            if not p.exists():
                raise FileNotFoundError(path)

            if p.suffix.lower() == ".json":
                items = json.loads(p.read_text(encoding="utf-8"))
                out = []
                for it in items:
                    out.append((it.get("name") or Path(it["url"]).name, it["url"]))
                return out

            out = []
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                name = Path(line.split("?")[0]).name or f"doc_{len(out):03d}"
                out.append((name, line))
            return out


        def download_one(name: str, url: str, out_dir: Path, timeout=60, retries=3):
            out_dir.mkdir(parents=True, exist_ok=True)
            safe = "".join(c if c.isalnum() or c in "._-+" else "_" for c in name)
            if not safe.lower().endswith(".pdf"):
                safe = safe + ".pdf"
            out_path = out_dir / safe

            if out_path.exists() and out_path.stat().st_size > 1000:
                print(f"✓ exists: {out_path.name}")
                return

            headers = {"User-Agent": "Mozilla/5.0 (oran-slm-qa; research)"}
            for attempt in range(1, retries + 1):
                try:
                    print(f"↓ downloading ({attempt}/{retries}): {safe}")
                    r = requests.get(url, headers=headers, timeout=timeout)
                    r.raise_for_status()
                    out_path.write_bytes(r.content)
                    print(f"✓ saved: {out_path} ({out_path.stat().st_size/1024:.1f} KB)")
                    return
                except Exception as e:
                    print(f"✗ failed: {url}\\n  {e}")
                    if attempt < retries:
                        time.sleep(2.0 * attempt)
                    else:
                        print("  giving up.\\n")


        def main():
            ap = argparse.ArgumentParser()
            ap.add_argument("--out", type=str, default="data/raw_specs", help="Output directory")
            ap.add_argument("--url_file", type=str, default=None, help="Optional URL list file (txt or json)")
            ap.add_argument("--timeout", type=int, default=60)
            args = ap.parse_args()

            out_dir = Path(args.out)
            urls = list(DEFAULT_URLS)
            if args.url_file:
                urls.extend(read_url_file(args.url_file))

            print(f"Will download {len(urls)} documents into: {out_dir}")
            for name, url in urls:
                download_one(name, url, out_dir, timeout=args.timeout)

            print("Done.")


        if __name__ == "__main__":
            main()
        """
    )

    files["scripts/build_corpus.py"] = textwrap.dedent(
        """\
        import argparse
        from pathlib import Path
        from oran_qa.data.parse_pdf import parse_pdf_to_pages
        from oran_qa.data.chunking import chunk_pages_to_sections, section_chunks_to_jsonl
        from oran_qa.utils.logging import setup_logger


        def main():
            ap = argparse.ArgumentParser()
            ap.add_argument("--raw_dir", type=str, required=True)
            ap.add_argument("--out", type=str, required=True)
            ap.add_argument("--max_chars", type=int, default=2200)
            ap.add_argument("--overlap_chars", type=int, default=300)
            args = ap.parse_args()

            log = setup_logger("build_corpus")

            raw_dir = Path(args.raw_dir)
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            pdfs = sorted(raw_dir.glob("*.pdf"))
            if not pdfs:
                raise RuntimeError(f"No PDFs found in {raw_dir}")

            all_chunks = []
            for pdf in pdfs:
                log.info(f"Parsing: {pdf.name}")
                pages = parse_pdf_to_pages(pdf)
                chunks = chunk_pages_to_sections(
                    pages,
                    doc_id=pdf.stem,
                    max_chars=args.max_chars,
                    overlap_chars=args.overlap_chars,
                )
                all_chunks.extend(chunks)

            log.info(f"Total chunks: {len(all_chunks)}")
            section_chunks_to_jsonl(all_chunks, out_path)
            log.info(f"Wrote: {out_path}")


        if __name__ == "__main__":
            main()
        """
    )

    files["scripts/make_sft_dataset.py"] = textwrap.dedent(
        """\
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
        """
    )

    files["scripts/train_lora.py"] = textwrap.dedent(
        """\
        import argparse
        from pathlib import Path
        import yaml
        from oran_qa.training.sft import run_sft_lora


        def main():
            ap = argparse.ArgumentParser()
            ap.add_argument("--config", type=str, required=True)
            args = ap.parse_args()

            cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
            run_sft_lora(cfg)


        if __name__ == "__main__":
            main()
        """
    )

    files["scripts/evaluate.py"] = textwrap.dedent(
        """\
        import argparse
        from pathlib import Path
        import yaml
        from oran_qa.utils.io import read_jsonl
        from oran_qa.modeling.generation import load_model_and_tokenizer, generate_answer
        from oran_qa.modeling.prompt import build_prompt
        from oran_qa.utils.text import normalize_text, f1_score


        def main():
            ap = argparse.ArgumentParser()
            ap.add_argument("--config", type=str, required=True)
            ap.add_argument("--eval_file", type=str, default=None)
            ap.add_argument("--max_n", type=int, default=200)
            args = ap.parse_args()

            cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
            eval_file = args.eval_file or cfg["eval_file"]

            model, tok = load_model_and_tokenizer(
                cfg["base_model"],
                lora_dir=cfg["output_dir"],
                use_4bit=cfg.get("use_4bit", True),
                bf16=cfg.get("bf16", True),
            )

            examples = list(read_jsonl(Path(eval_file)))[: args.max_n]
            ems, f1s = [], []

            for ex in examples:
                prompt = build_prompt(ex["question"], ex["context"], style=cfg.get("prompt_style", "evidence_qa_v1"))
                pred = generate_answer(model, tok, prompt, max_new_tokens=192, temperature=0.0)
                gold = ex["answer"]

                em = 1.0 if normalize_text(pred) == normalize_text(gold) else 0.0
                f1 = f1_score(pred, gold)
                ems.append(em)
                f1s.append(f1)

            print(f"N={len(examples)} | EM={sum(ems)/len(ems):.3f} | F1={sum(f1s)/len(f1s):.3f}")


        if __name__ == "__main__":
            main()
        """
    )

    files["scripts/build_faiss_index.py"] = textwrap.dedent(
        """\
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
        """
    )

    files["scripts/chat_rag.py"] = textwrap.dedent(
        """\
        import argparse
        from pathlib import Path
        import yaml
        from rich.console import Console
        from oran_qa.retrieval.retriever import FaissRetriever
        from oran_qa.modeling.generation import load_model_and_tokenizer, generate_answer
        from oran_qa.modeling.prompt import build_prompt


        def main():
            ap = argparse.ArgumentParser()
            ap.add_argument("--config", type=str, required=True)
            args = ap.parse_args()

            cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
            console = Console()

            retr = FaissRetriever(
                index_dir=cfg["retrieval"]["index_dir"],
                embed_model=cfg["retrieval"]["embed_model"],
                top_k=int(cfg["retrieval"]["top_k"]),
            )

            model, tok = load_model_and_tokenizer(
                cfg["model"]["base_model"],
                lora_dir=cfg["model"].get("lora_dir"),
                use_4bit=True,
                bf16=True,
            )

            console.print("[bold]RAG chat ready. Type 'exit' to quit.[/bold]")
            while True:
                q = console.input("\\n[bold cyan]Q> [/bold cyan]").strip()
                if not q or q.lower() in {"exit", "quit"}:
                    break

                hits = retr.search(q)
                context = "\\n\\n".join(
                    [f"[{h['doc_id']} | {h.get('section','?')}] {h['text']}" for h in hits]
                )

                prompt = build_prompt(q, context, style="evidence_qa_v1")
                ans = generate_answer(
                    model,
                    tok,
                    prompt,
                    max_new_tokens=int(cfg["generation"]["max_new_tokens"]),
                    temperature=float(cfg["generation"]["temperature"]),
                )

                console.print("\\n[bold green]Answer[/bold green]")
                console.print(ans)


        if __name__ == "__main__":
            main()
        """
    )

    # -------------------------
    # Package files
    # -------------------------
    files["src/oran_qa/__init__.py"] = " __all__ = [\"data\", \"retrieval\", \"modeling\", \"training\", \"utils\"]\n"

    files["src/oran_qa/utils/__init__.py"] = ""
    files["src/oran_qa/utils/logging.py"] = textwrap.dedent(
        """\
        import logging
        import sys


        def setup_logger(name: str) -> logging.Logger:
            logger = logging.getLogger(name)
            logger.setLevel(logging.INFO)
            if not logger.handlers:
                h = logging.StreamHandler(sys.stdout)
                fmt = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
                h.setFormatter(fmt)
                logger.addHandler(h)
            return logger
        """
    )

    files["src/oran_qa/utils/io.py"] = textwrap.dedent(
        """\
        import json
        from pathlib import Path
        from typing import Iterable, Dict, Any


        def read_jsonl(path: Path):
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)


        def write_jsonl(items: Iterable[Dict[str, Any]], path: Path):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as f:
                for it in items:
                    f.write(json.dumps(it, ensure_ascii=False) + "\\n")
        """
    )

    files["src/oran_qa/utils/text.py"] = textwrap.dedent(
        """\
        import re
        from collections import Counter


        _WS = re.compile(r"\\s+")
        _PUNCT = re.compile(r"[^\\w\\s]")


        def normalize_text(s: str) -> str:
            s = s.lower().strip()
            s = _PUNCT.sub(" ", s)
            s = _WS.sub(" ", s)
            return s.strip()


        def _tokens(s: str):
            return normalize_text(s).split()


        def f1_score(pred: str, gold: str) -> float:
            pt = _tokens(pred)
            gt = _tokens(gold)
            if not pt and not gt:
                return 1.0
            if not pt or not gt:
                return 0.0
            pc = Counter(pt)
            gc = Counter(gt)
            common = pc & gc
            num_same = sum(common.values())
            if num_same == 0:
                return 0.0
            precision = num_same / len(pt)
            recall = num_same / len(gt)
            return 2 * precision * recall / (precision + recall)
        """
    )

    files["src/oran_qa/data/__init__.py"] = ""
    files["src/oran_qa/data/parse_pdf.py"] = textwrap.dedent(
        """\
        from dataclasses import dataclass
        from pathlib import Path
        from typing import List
        from pypdf import PdfReader


        @dataclass
        class Page:
            page_num: int
            text: str


        def parse_pdf_to_pages(pdf_path: Path) -> List[Page]:
            reader = PdfReader(str(pdf_path))
            pages: List[Page] = []
            for i, p in enumerate(reader.pages):
                txt = p.extract_text() or ""
                txt = txt.replace("\\x00", " ").strip()
                pages.append(Page(page_num=i + 1, text=txt))
            return pages
        """
    )

    files["src/oran_qa/data/chunking.py"] = textwrap.dedent(
        """\
        from typing import List, Dict, Any
        import re
        from pathlib import Path
        from oran_qa.utils.io import write_jsonl

        # Very simple heading heuristic: lines starting with "1", "1.2", "7.3.2" etc.
        HEADING_RE = re.compile(r"^\\s*(\\d+(?:\\.\\d+){0,6})\\s+(.{3,120})\\s*$")


        def _split_lines(text: str) -> List[str]:
            lines = [ln.strip() for ln in text.splitlines()]
            return [ln for ln in lines if ln]


        def chunk_pages_to_sections(
            pages,
            doc_id: str,
            max_chars: int = 2200,
            overlap_chars: int = 300,
        ) -> List[Dict[str, Any]]:
            \"\"\"
            Pipeline:
              - scan headings across pages
              - build coarse sections
              - chunk section text into overlapping windows
            \"\"\"
            stream = []
            for pg in pages:
                for ln in _split_lines(pg.text):
                    stream.append((pg.page_num, ln))

            headings = []
            for idx, (pg, ln) in enumerate(stream):
                m = HEADING_RE.match(ln)
                if m:
                    sec = m.group(1)
                    title = m.group(2)
                    headings.append((idx, pg, sec, title))

            if not headings:
                full = "\\n".join([ln for _, ln in stream])
                return _chunk_text_windows(
                    doc_id=doc_id,
                    section="0",
                    title="Document",
                    page_start=1,
                    page_end=pages[-1].page_num if pages else 1,
                    text=full,
                    max_chars=max_chars,
                    overlap_chars=overlap_chars,
                )

            chunks: List[Dict[str, Any]] = []
            for i, (idx, pg, sec, title) in enumerate(headings):
                start = idx
                end = headings[i + 1][0] if i + 1 < len(headings) else len(stream)
                sec_lines = [ln for _, ln in stream[start:end]]
                sec_text = "\\n".join(sec_lines).strip()
                if len(sec_text) < 50:
                    continue

                page_end = stream[end - 1][0] if end - 1 >= 0 else pg
                chunks.extend(
                    _chunk_text_windows(
                        doc_id=doc_id,
                        section=sec,
                        title=title,
                        page_start=pg,
                        page_end=page_end,
                        text=sec_text,
                        max_chars=max_chars,
                        overlap_chars=overlap_chars,
                    )
                )

            return chunks


        def _chunk_text_windows(
            doc_id: str,
            section: str,
            title: str,
            page_start: int,
            page_end: int,
            text: str,
            max_chars: int,
            overlap_chars: int,
        ) -> List[Dict[str, Any]]:
            out = []
            step = max_chars - overlap_chars
            if step <= 0:
                step = max_chars

            for i, start in enumerate(range(0, len(text), step)):
                window = text[start : start + max_chars].strip()
                if len(window) < 80:
                    continue
                cid = f"{doc_id}::{section}::{i:04d}"
                out.append(
                    {
                        "id": cid,
                        "doc_id": doc_id,
                        "section": section,
                        "title": title,
                        "page_start": page_start,
                        "page_end": page_end,
                        "text": window,
                    }
                )
            return out


        def section_chunks_to_jsonl(chunks: List[Dict[str, Any]], out_path: Path):
            write_jsonl(chunks, out_path)
        """
    )

    files["src/oran_qa/data/qa_synthesis.py"] = textwrap.dedent(
        """\
        import re
        import random
        from typing import Dict, Any, List

        MODAL_RE = re.compile(r"\\b(shall|shall not|should|should not|may|need not|must)\\b", re.IGNORECASE)
        DEF_LINE_RE = re.compile(r"^\\s*([A-Z][A-Za-z0-9/\\-\\s]{1,40})\\s*:\\s*(.{10,300})\\s*$")


        def _pick_definition_candidates(text: str) -> List[Dict[str, str]]:
            out = []
            for ln in text.splitlines():
                ln = ln.strip()
                m = DEF_LINE_RE.match(ln)
                if m:
                    term = m.group(1).strip()
                    defi = m.group(2).strip()
                    if 2 <= len(term.split()) <= 8 and 20 <= len(defi) <= 260:
                        out.append({"term": term, "def": defi, "evidence": ln})
            return out


        def _pick_requirement_sentences(text: str) -> List[str]:
            sents = re.split(r"(?<=[\\.\\?])\\s+", text.replace("\\n", " "))
            out = []
            for s in sents:
                s = s.strip()
                if 40 <= len(s) <= 280 and MODAL_RE.search(s):
                    out.append(s)
            return out


        def synthesize_qa_pairs(chunk: Dict[str, Any], max_pairs: int = 2) -> List[Dict[str, Any]]:
            doc_id = chunk["doc_id"]
            section = chunk.get("section", "?")
            title = chunk.get("title", "")
            text = chunk["text"]

            examples = []

            defs = _pick_definition_candidates(text)
            random.shuffle(defs)
            for d in defs[: max_pairs]:
                q = f"What is {d['term']}?"
                a = d["def"]
                examples.append(
                    {
                        "id": f"{chunk['id']}::def::{d['term'][:24]}",
                        "question": q,
                        "context": f"[{doc_id} | {section} | {title}]\\n{text}",
                        "answer": a,
                        "citations": [{"doc_id": doc_id, "section": section, "chunk_id": chunk["id"]}],
                        "meta": {"type": "definition", "doc_id": doc_id, "section": section, "title": title},
                    }
                )

            if len(examples) >= max_pairs:
                return examples[:max_pairs]

            reqs = _pick_requirement_sentences(text)
            random.shuffle(reqs)
            for s in reqs[: max_pairs - len(examples)]:
                m = MODAL_RE.search(s)
                topic = "this requirement"
                if m:
                    left = s[: m.start()].strip()
                    toks = left.split()
                    topic = " ".join(toks[-6:]) if toks else "this requirement"

                q = f"According to {doc_id} section {section}, what does it require about {topic}?"
                a = s
                examples.append(
                    {
                        "id": f"{chunk['id']}::req::{abs(hash(s)) % 10**8}",
                        "question": q,
                        "context": f"[{doc_id} | {section} | {title}]\\n{text}",
                        "answer": a,
                        "citations": [{"doc_id": doc_id, "section": section, "chunk_id": chunk["id"]}],
                        "meta": {"type": "requirement", "doc_id": doc_id, "section": section, "title": title},
                    }
                )

            return examples[:max_pairs]
        """
    )

    files["src/oran_qa/retrieval/__init__.py"] = ""
    files["src/oran_qa/retrieval/embed.py"] = textwrap.dedent(
        """\
        from sentence_transformers import SentenceTransformer
        import numpy as np


        class Embedder:
            def __init__(self, model_name: str):
                self.model = SentenceTransformer(model_name)

            def encode(self, texts, batch_size: int = 64):
                emb = self.model.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
                return np.asarray(emb, dtype="float32")
        """
    )

    files["src/oran_qa/retrieval/faiss_index.py"] = textwrap.dedent(
        """\
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
                "\\n".join(json.dumps(c, ensure_ascii=False) for c in chunks),
                encoding="utf-8",
            )
            (out_dir / "meta.json").write_text(
                json.dumps({"embed_model": embed_model, "dim": dim, "count": len(chunks)}, indent=2),
                encoding="utf-8",
            )
        """
    )

    files["src/oran_qa/retrieval/retriever.py"] = textwrap.dedent(
        """\
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
        """
    )

    files["src/oran_qa/modeling/__init__.py"] = ""
    files["src/oran_qa/modeling/prompt.py"] = textwrap.dedent(
        """\
        def build_prompt(question: str, context: str, style: str = "evidence_qa_v1") -> str:
            if style == "evidence_qa_v1":
                return (
                    "You are a careful assistant for telecom standards QA.\\n"
                    "Answer ONLY using the provided context. If the answer is not in the context, say: "
                    "\\"I don't know based on the provided context.\\"\\n\\n"
                    "Output format:\\n"
                    "Answer: <1-4 sentences>\\n"
                    "Citations: <bullet list of [doc_id | section] you used>\\n\\n"
                    f"Question:\\n{question}\\n\\n"
                    f"Context:\\n{context}\\n\\n"
                    "Answer:"
                )

            return f"Question: {question}\\n\\nContext:\\n{context}\\n\\nAnswer:"
        """
    )

    files["src/oran_qa/modeling/generation.py"] = textwrap.dedent(
        """\
        from typing import Optional
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel


        def load_model_and_tokenizer(
            base_model: str,
            lora_dir: Optional[str] = None,
            use_4bit: bool = True,
            bf16: bool = True,
        ):
            tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)

            kwargs = {}
            if use_4bit:
                kwargs.update(dict(load_in_4bit=True, device_map="auto"))
            else:
                kwargs.update(dict(device_map="auto"))
            if bf16:
                kwargs.update(dict(torch_dtype=torch.bfloat16))

            model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
            if lora_dir:
                model = PeftModel.from_pretrained(model, lora_dir)

            model.eval()
            return model, tok


        @torch.no_grad()
        def generate_answer(
            model,
            tok,
            prompt: str,
            max_new_tokens: int = 256,
            temperature: float = 0.2,
        ) -> str:
            inputs = tok(prompt, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else 1.0,
                pad_token_id=tok.eos_token_id,
            )
            text = tok.decode(out[0], skip_special_tokens=True)

            # Return only the part after the last occurrence of "Answer:"
            if "Answer:" in text:
                return text.split("Answer:")[-1].strip()
            return text.strip()
        """
    )

    # Fix: ensure no leading whitespace in __init__.py
    files["src/oran_qa/__init__.py"] = '__all__ = ["data", "retrieval", "modeling", "training", "utils"]\n'

    files["src/oran_qa/training/__init__.py"] = ""
    files["src/oran_qa/training/sft.py"] = textwrap.dedent(
        """\
        from datasets import load_dataset
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            TrainingArguments,
            DataCollatorForLanguageModeling,
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from oran_qa.modeling.prompt import build_prompt


        def _format_example(ex, prompt_style: str):
            prompt = build_prompt(ex["question"], ex["context"], style=prompt_style)

            citations = ex.get("citations", [])
            cite_lines = "\\n".join([f"- [{c['doc_id']} | {c.get('section','?')}]" for c in citations]) or "- [unknown]"
            target = f"{ex['answer']}\\n\\nCitations:\\n{cite_lines}\\n"
            return {"text": prompt + " " + target}


        def run_sft_lora(cfg: dict):
            base_model = cfg["base_model"]
            train_file = cfg["train_file"]
            eval_file = cfg["eval_file"]
            output_dir = cfg["output_dir"]

            prompt_style = cfg.get("prompt_style", "evidence_qa_v1")
            max_seq_len = int(cfg.get("max_seq_len", 2048))

            use_4bit = bool(cfg.get("use_4bit", True))
            bf16 = bool(cfg.get("bf16", True))
            fp16 = bool(cfg.get("fp16", False))

            tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token

            model_kwargs = dict(device_map="auto")
            if use_4bit:
                model_kwargs["load_in_4bit"] = True
            if bf16:
                model_kwargs["torch_dtype"] = torch.bfloat16

            model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
            if use_4bit:
                model = prepare_model_for_kbit_training(model)

            lora = LoraConfig(
                r=int(cfg.get("lora_r", 16)),
                lora_alpha=int(cfg.get("lora_alpha", 32)),
                lora_dropout=float(cfg.get("lora_dropout", 0.05)),
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
            )
            model = get_peft_model(model, lora)

            ds = load_dataset("json", data_files={"train": train_file, "eval": eval_file})
            ds = ds.map(lambda ex: _format_example(ex, prompt_style), remove_columns=ds["train"].column_names)

            def tokenize(batch):
                return tok(
                    batch["text"],
                    truncation=True,
                    max_length=max_seq_len,
                    padding=False,
                )

            ds_tok = ds.map(tokenize, batched=True, remove_columns=["text"])
            collator = DataCollatorForLanguageModeling(tok, mlm=False)

            args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=float(cfg.get("num_train_epochs", 1)),
                per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 2)),
                per_device_eval_batch_size=int(cfg.get("per_device_eval_batch_size", 2)),
                gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 8)),
                learning_rate=float(cfg.get("learning_rate", 2e-4)),
                warmup_ratio=float(cfg.get("warmup_ratio", 0.03)),
                weight_decay=float(cfg.get("weight_decay", 0.0)),
                logging_steps=int(cfg.get("logging_steps", 20)),
                save_steps=int(cfg.get("save_steps", 200)),
                eval_steps=int(cfg.get("eval_steps", 200)),
                evaluation_strategy="steps",
                save_strategy="steps",
                bf16=bf16,
                fp16=fp16,
                report_to=[],
            )

            # Prefer TRL if available; fallback to HF Trainer.
            try:
                from trl import SFTTrainer

                trainer = SFTTrainer(
                    model=model,
                    args=args,
                    train_dataset=ds_tok["train"],
                    eval_dataset=ds_tok["eval"],
                    data_collator=collator,
                    tokenizer=tok,
                    max_seq_length=max_seq_len,
                )
            except Exception:
                from transformers import Trainer

                trainer = Trainer(
                    model=model,
                    args=args,
                    train_dataset=ds_tok["train"],
                    eval_dataset=ds_tok["eval"],
                    data_collator=collator,
                    tokenizer=tok,
                )

            trainer.train()
            trainer.save_model(output_dir)
            tok.save_pretrained(output_dir)

            print(f"Saved LoRA adapter to: {output_dir}")
        """
    )

    # -------------------------
    # Tests
    # -------------------------
    files["tests/test_chunking.py"] = textwrap.dedent(
        """\
        from oran_qa.data.chunking import HEADING_RE


        def test_heading_regex():
            assert HEADING_RE.match("1 Introduction")
            assert HEADING_RE.match("7.3.2 Near-RT RIC functions")
            assert not HEADING_RE.match("Introduction without numbering")
        """
    )

    # -------------------------
    # Write everything
    # -------------------------
    for rel, content in files.items():
        write_file(root / rel, content, force=args.force)

    print("\\n✅ Repo created at:", root)
    print("\\nNext steps:")
    print(f"  cd {root.name}")
    print("  pip install -r requirements.txt")
    print("  python scripts/download_oran_specs.py --out data/raw_specs")
    print("  python scripts/build_corpus.py --raw_dir data/raw_specs --out data/corpus/chunks.jsonl")
    print("  python scripts/make_sft_dataset.py --chunks data/corpus/chunks.jsonl --out data/sft --train_ratio 0.95")
    print("  python scripts/train_lora.py --config configs/train_lora.yaml")
    print("  python scripts/build_faiss_index.py --chunks data/corpus/chunks.jsonl --index_dir data/index")
    print("  python scripts/chat_rag.py --config configs/rag.yaml")


if __name__ == "__main__":
    main()