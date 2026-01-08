# oran-slm-qa

Finetune a Small Language Model (SLM) on O-RAN specification PDFs (closed-book) and benchmark it on multiple-choice QA (MCQ).

This repo implements an end-to-end pipeline:

1) Download publicly available O-RAN/ETSI PDFs  
2) Parse PDFs -> build a clean chunked corpus  
3) Sanity-check corpus quality + prepare MCQ train/valid/test  
4) Domain-Adaptive Pretraining (DAPT) on the PDF corpus (QLoRA)  
5) Supervised Fine-tuning (SFT) on MCQ to teach the model to answer with an option number  
6) Evaluate accuracy on the MCQ benchmark

The default base model is:
- Qwen/Qwen3-4B


## What "closed-book" means here

This project is closed-book QA:
- At inference time, the model does not retrieve or read PDFs.
- All knowledge must be stored in the model weights (via DAPT + SFT).

If you want RAG (open-book with citations), that is a different project setup.


## Repository layout

Typical important paths:

- scripts/
  - download_specs.py : download O-RAN/ETSI spec PDFs
  - build_corpus.py : parse PDFs into chunks (chunks.jsonl)
  - sanity_and_prepare_data.py : corpus report + prepare MCQ splits
  - prepare_dapt_data.py : pack tokenized corpus for DAPT
  - train_dapt.py : train DAPT adapter (LoRA/QLoRA)
  - train_mcq_sft.py : train MCQ SFT adapter (init from DAPT adapter)
  - eval_mcq.py : evaluate MCQ accuracy

- data/
  - raw_specs/ : downloaded PDFs
  - corpus/chunks.jsonl : extracted corpus chunks
  - benchmarks/
    - raw_mcq.jsonl : raw benchmark input (chat jsonl format)
    - train_mcq.jsonl, valid_mcq.jsonl, test_mcq.jsonl : prepared splits
  - dapt/tokenized/ : packed dataset for DAPT

- outputs/
  - dapt_adapter/ : LoRA adapter after domain-adaptive pretraining
  - mcq_adapter/ : LoRA adapter after MCQ supervised fine-tuning
  - corpus_report.json : sanity-check report


## Data formats

### 1) Corpus chunks (data/corpus/chunks.jsonl)

Each line is a JSON object like:

{
  "id": "DOC_ID::SECTION::CHUNK_ID",
  "doc_id": "DOC_ID",
  "section": "4.1.2",
  "title": "Motivation",
  "page_start": 13,
  "page_end": 13,
  "text": "..."
}

Notes:
- Chunking uses max_chars and overlap_chars.
- Some chunks may still contain boilerplate/headers/footers; sanity check helps detect them.


### 2) MCQ benchmark (data/benchmarks/*.jsonl)

Chat-style JSONL. Each example is:

{
  "id": "mcq_000001",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {
      "role": "user",
      "content": "You are an expert on O-RAN specifications.\nAnswer the following multiple-choice question using your internal knowledge only (closed-book).\nRespond with ONLY the option number (e.g., 1).\n\nQuestion: ...\nOptions:\n1. ...\n2. ...\n3. ...\n4. ...\n\nAnswer:"
    },
    {"role": "assistant", "content": "3"}
  ]
}

The SFT stage learns to output only the option number.


## Setup

### 1) Create a virtual environment

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip

### 2) Install dependencies

pip install -U \
  "transformers>=4.44" \
  "datasets>=2.20" \
  "accelerate>=0.33" \
  "peft>=0.12" \
  "trl>=0.10" \
  bitsandbytes \
  sentencepiece

Make sure PYTHONPATH allows importing local modules:

export PYTHONPATH="$(pwd)"


## End-to-end run (recommended)

If you have a run_oran_dapt_sft.sh script:

chmod +x run_oran_dapt_sft.sh
./run_oran_dapt_sft.sh

This script should:
- download PDFs
- build corpus
- run sanity check + prepare MCQ splits
- run DAPT
- run MCQ SFT
- evaluate test accuracy


## Manual run (step-by-step)

### Step 1: Download PDFs

python scripts/download_specs.py --out data/raw_specs

### Step 2: Build corpus chunks

python scripts/build_corpus.py \
  --raw_dir data/raw_specs \
  --out data/corpus/chunks.jsonl \
  --max_chars 2200 \
  --overlap_chars 300

### Step 3: Sanity-check corpus and prepare MCQ splits

Make sure you have:
- data/benchmarks/raw_mcq.jsonl

Then:

python scripts/sanity_and_prepare_data.py \
  --corpus data/corpus/chunks.jsonl \
  --report_out outputs/corpus_report.json \
  --benchmark data/benchmarks/raw_mcq.jsonl \
  --bench_out_dir data/benchmarks \
  --format chat \
  --valid_ratio 0.1 \
  --test_ratio 0.1 \
  --answer_style number

Expected outputs:
- data/benchmarks/train_mcq.jsonl
- data/benchmarks/valid_mcq.jsonl
- data/benchmarks/test_mcq.jsonl


### Step 4: Prepare DAPT packed dataset

python scripts/prepare_dapt_data.py \
  --corpus data/corpus/chunks.jsonl \
  --out data/dapt/tokenized \
  --model Qwen/Qwen3-4B \
  --block_size 2048 \
  --min_chars 300

### Step 5: Train DAPT adapter

python scripts/train_dapt.py \
  --model Qwen/Qwen3-4B \
  --data data/dapt/tokenized \
  --out outputs/dapt_adapter \
  --epochs 1 \
  --bsz 1 \
  --grad_accum 16 \
  --lr 2e-4 \
  --save_steps 200

### Step 6: Train MCQ SFT adapter (initialized from DAPT)

python scripts/train_mcq_sft.py \
  --model Qwen/Qwen3-4B \
  --init_adapter outputs/dapt_adapter \
  --train data/benchmarks/train_mcq.jsonl \
  --valid data/benchmarks/valid_mcq.jsonl \
  --out outputs/mcq_adapter \
  --use_4bit \
  --epochs 3 \
  --lr 1e-4 \
  --bsz 1 \
  --grad_accum 16 \
  --max_seq_len 2048 \
  --save_steps 200

### Step 7: Evaluate

python scripts/eval_mcq.py \
  --model Qwen/Qwen3-4B \
  --adapter outputs/mcq_adapter \
  --test data/benchmarks/test_mcq.jsonl


## Tips & troubleshooting

### 1) ModuleNotFoundError: No module named 'oran_qa'
Run from repo root and set:

export PYTHONPATH="$(pwd)"

Or install your package in editable mode (if applicable):

pip install -e .

### 2) Corpus looks full of headers/footers/TOC
This is common for ETSI PDFs. Use outputs/corpus_report.json to find bad chunks.
If too many chunks have heavy footer/header repetition, improve the cleaning rules in your PDF parsing/corpus build step.

### 3) MCQ dataset is too small
If your benchmark is small, accuracy will be noisy and SFT may overfit.
Consider:
- adding more MCQs
- adding "format enforcement" examples (model always outputs 1..4 only)
- keep SFT small (few epochs) and rely on DAPT for knowledge


## Outputs

After a successful run:

- outputs/dapt_adapter/ : domain knowledge adapter trained on O-RAN PDFs
- outputs/mcq_adapter/ : MCQ answering adapter (initialized from DAPT)
- outputs/corpus_report.json : quality report for corpus chunks


## License & data notes

- PDFs are downloaded from public sources (e.g., ETSI deliverables). Respect the terms of use of the source repositories.
- This repo stores extracted text chunks for research and model training; ensure you comply with any licensing requirements of the documents you use.
