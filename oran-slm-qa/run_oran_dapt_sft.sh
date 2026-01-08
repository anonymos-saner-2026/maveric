#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# End-to-end pipeline: O-RAN PDFs -> corpus -> DAPT (continued pretrain) -> MCQ SFT -> eval
# Base model: Qwen/Qwen3-4B
# Timezone/date not relevant for script. Assumes you run from repo root.
# ============================================================================

# -----------------------
# Config (edit if needed)
# -----------------------
BASE_MODEL="Qwen/Qwen3-4B"

RAW_SPECS_DIR="data/raw_specs"
CORPUS_JSONL="data/corpus/chunks.jsonl"

BENCH_DIR="data/benchmarks"
RAW_MCQ="${BENCH_DIR}/raw_mcq.jsonl"
TRAIN_MCQ="${BENCH_DIR}/train_mcq.jsonl"
VALID_MCQ="${BENCH_DIR}/valid_mcq.jsonl"
TEST_MCQ="${BENCH_DIR}/test_mcq.jsonl"

DAPT_DATA_DIR="data/dapt/tokenized"
DAPT_ADAPTER_DIR="outputs/dapt_adapter"
MCQ_ADAPTER_DIR="outputs/mcq_adapter"

CORPUS_REPORT="outputs/corpus_report.json"

# DAPT params
DAPT_BLOCK_SIZE=2048
DAPT_MIN_CHARS=300
DAPT_EPOCHS=1
DAPT_LR=2e-4
DAPT_BSZ=1
DAPT_GRAD_ACCUM=16
DAPT_SAVE_STEPS=200

# SFT params
SFT_EPOCHS=3
SFT_LR=1e-4
SFT_BSZ=1
SFT_GRAD_ACCUM=16
SFT_MAX_SEQ_LEN=2048
SFT_SAVE_STEPS=200

# -----------------------
# Helpers
# -----------------------
log() { echo -e "\n\033[1;34m[RUN]\033[0m $*"; }
warn() { echo -e "\n\033[1;33m[WARN]\033[0m $*"; }
die() { echo -e "\n\033[1;31m[ERR]\033[0m $*"; exit 1; }

# -----------------------
# Sanity: run from repo root
# -----------------------
if [[ ! -d "scripts" ]]; then
  die "Please run this script from the repo root (folder containing ./scripts)."
fi

# -----------------------
# (Optional) venv setup
# -----------------------
if [[ ! -d ".venv" ]]; then
  log "Creating venv: .venv"
  python3 -m venv .venv
fi

log "Activating venv"
# shellcheck disable=SC1091
source .venv/bin/activate

log "Upgrading pip"
python -m pip install -U pip

# -----------------------
# Install deps
# -----------------------
log "Installing Python dependencies"
pip install -U \
  "transformers>=4.44" \
  "datasets>=2.20" \
  "accelerate>=0.33" \
  "peft>=0.12" \
  "trl>=0.10" \
  bitsandbytes \
  sentencepiece

# -----------------------
# Ensure imports work
# -----------------------
log "Setting PYTHONPATH"
export PYTHONPATH="$(pwd)"

# If your repo is installable, you can uncomment:
# log "Editable install"
# pip install -e .

# -----------------------
# 1) Download O-RAN/ETSI PDFs
# -----------------------
log "Step 1/7: Download specs PDFs -> ${RAW_SPECS_DIR}"
python scripts/download_specs.py --out "${RAW_SPECS_DIR}"

# -----------------------
# 2) Build corpus chunks.jsonl
# -----------------------
log "Step 2/7: Build corpus -> ${CORPUS_JSONL}"
python scripts/build_corpus.py \
  --raw_dir "${RAW_SPECS_DIR}" \
  --out "${CORPUS_JSONL}" \
  --max_chars 2200 \
  --overlap_chars 300

# -----------------------
# 3) Sanity-check corpus + prepare MCQ splits
# -----------------------
log "Step 3/7: Sanity-check corpus + prepare MCQ splits"
if [[ ! -f "${RAW_MCQ}" ]]; then
  warn "Missing benchmark file: ${RAW_MCQ}"
  warn "Create it first, then rerun. Corpus build is done already."
  warn "Expected outputs: ${TRAIN_MCQ}, ${VALID_MCQ}, ${TEST_MCQ}"
  exit 0
fi

python scripts/sanity_and_prepare_data.py \
  --corpus "${CORPUS_JSONL}" \
  --report_out "${CORPUS_REPORT}" \
  --benchmark "${RAW_MCQ}" \
  --bench_out_dir "${BENCH_DIR}" \
  --format chat \
  --valid_ratio 0.1 \
  --test_ratio 0.1 \
  --answer_style number

# Check splits exist
[[ -f "${TRAIN_MCQ}" ]] || die "Missing ${TRAIN_MCQ} after sanity_and_prepare_data.py"
[[ -f "${VALID_MCQ}" ]] || warn "Missing ${VALID_MCQ} (ok if you set valid_ratio=0)"
[[ -f "${TEST_MCQ}"  ]] || warn "Missing ${TEST_MCQ} (ok if you set test_ratio=0)"

# -----------------------
# 4) Prepare DAPT tokenized+packed dataset
# -----------------------
log "Step 4/7: Prepare DAPT data -> ${DAPT_DATA_DIR}"
python scripts/prepare_dapt_data.py \
  --corpus "${CORPUS_JSONL}" \
  --out "${DAPT_DATA_DIR}" \
  --model "${BASE_MODEL}" \
  --block_size "${DAPT_BLOCK_SIZE}" \
  --min_chars "${DAPT_MIN_CHARS}"

# -----------------------
# 5) Train DAPT adapter (QLoRA)
# -----------------------
log "Step 5/7: Train DAPT adapter -> ${DAPT_ADAPTER_DIR}"
python scripts/train_dapt.py \
  --model "${BASE_MODEL}" \
  --data "${DAPT_DATA_DIR}" \
  --out "${DAPT_ADAPTER_DIR}" \
  --epochs "${DAPT_EPOCHS}" \
  --bsz "${DAPT_BSZ}" \
  --grad_accum "${DAPT_GRAD_ACCUM}" \
  --lr "${DAPT_LR}" \
  --save_steps "${DAPT_SAVE_STEPS}"

# -----------------------
# 6) Train MCQ SFT adapter initialized from DAPT adapter
# -----------------------
log "Step 6/7: Train MCQ SFT adapter -> ${MCQ_ADAPTER_DIR}"
# Note: --use_4bit enables QLoRA for the SFT stage too (recommended).
python scripts/train_mcq_sft.py \
  --model "${BASE_MODEL}" \
  --init_adapter "${DAPT_ADAPTER_DIR}" \
  --train "${TRAIN_MCQ}" \
  --valid "${VALID_MCQ}" \
  --out "${MCQ_ADAPTER_DIR}" \
  --use_4bit \
  --epochs "${SFT_EPOCHS}" \
  --lr "${SFT_LR}" \
  --bsz "${SFT_BSZ}" \
  --grad_accum "${SFT_GRAD_ACCUM}" \
  --max_seq_len "${SFT_MAX_SEQ_LEN}" \
  --save_steps "${SFT_SAVE_STEPS}"

# -----------------------
# 7) Eval on test set
# -----------------------
log "Step 7/7: Evaluate MCQ accuracy"
if [[ -f "${TEST_MCQ}" ]]; then
  python scripts/eval_mcq.py \
    --model "${BASE_MODEL}" \
    --adapter "${MCQ_ADAPTER_DIR}" \
    --test "${TEST_MCQ}"
else
  warn "No test split found at ${TEST_MCQ}. Skipping eval."
fi

log "ALL DONE ✅"
