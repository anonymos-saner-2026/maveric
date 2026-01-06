# build_multitask_data.py
"""
Sinh file multitask_data.json để dùng cho E3 multi-task fine-tune.

Nguồn:
  - qa_dataset.json        (từ prepare_qa_dataset.py)
  - claim_dataset.json     (từ main.py / DataPipeline)

Đầu ra:
  - multitask_data.json    gồm 2 loại sample:
      {
        "task": "qa",
        "question": "...",
        "answer": "..."
      }
      {
        "task": "lattice",
        "claim": "...",
        "paraphrase": "...",
        "weakening": "...",
        "strengthening": "...",
        "negation": "...",
        "label": 0/1
      }
"""

import json
import random
from typing import List, Dict, Any

from tqdm import tqdm

from coclac import (
    LatticeGenerator,
    OpenAILLMClient,
)


QA_DATA_PATH = "qa_dataset.json"
CLAIM_DATA_PATH = "claim_dataset.json"
OUTPUT_PATH = "multitask_data.json"

# Số lượng mẫu tối đa (có thể chỉnh tuỳ GPU / thời gian)
MAX_QA_SAMPLES = 500          # số QA sample (task="qa")
MAX_LATTICE_SAMPLES = 500     # số lattice sample (task="lattice")


def load_qa_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Mỗi item dạng: {"question": ..., "gold_answers": [...]}
    return data


def load_claim_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Mỗi item dạng: {"question", "answer", "claim_text", "label"}
    return data


def build_qa_samples(qa_data: List[Dict[str, Any]], max_samples: int) -> List[Dict[str, Any]]:
    qa_samples = []
    # shuffle để random subset
    random.shuffle(qa_data)

    for item in qa_data[:max_samples]:
        q = item["question"]
        gold_answers = item.get("gold_answers", [])
        if not gold_answers:
            continue
        # Lấy gold_answers[0] làm answer target
        ans = gold_answers[0]
        qa_samples.append(
            {
                "task": "qa",
                "question": q,
                "answer": ans,
            }
        )
    return qa_samples


def build_lattice_samples(
    claim_data: List[Dict[str, Any]],
    lattice_gen: LatticeGenerator,
    max_samples: int,
) -> List[Dict[str, Any]]:
    lattice_samples = []
    random.shuffle(claim_data)

    for item in tqdm(claim_data[:max_samples], desc="Lattice samples", unit="claim"):
        claim_text = item["claim_text"]
        label = int(item["label"])

        # Sinh paraphrase/weakening/strengthening/negation bằng LatticeGenerator
        lattice = lattice_gen.build_lattice(claim_text)

        lattice_samples.append(
            {
                "task": "lattice",
                "claim": lattice.original,
                "paraphrase": lattice.paraphrase,
                "weakening": lattice.weakening,
                "strengthening": lattice.strengthening,
                "negation": lattice.negation,
                "label": label,
            }
        )

    return lattice_samples


def main():
    print(f"Loading QA data from {QA_DATA_PATH} ...")
    qa_data = load_qa_dataset(QA_DATA_PATH)
    print(f"Loaded {len(qa_data)} QA items.")

    print(f"Loading claim data from {CLAIM_DATA_PATH} ...")
    claim_data = load_claim_dataset(CLAIM_DATA_PATH)
    print(f"Loaded {len(claim_data)} claim items.")

    # Init LLM client + lattice generator
    llm = OpenAILLMClient()  # dùng OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL từ env
    lattice_gen = LatticeGenerator(llm)

    # Build QA samples
    qa_samples = build_qa_samples(qa_data, MAX_QA_SAMPLES)
    print(f"Built {len(qa_samples)} QA samples.")

    # Build lattice samples
    lattice_samples = build_lattice_samples(claim_data, lattice_gen, MAX_LATTICE_SAMPLES)
    print(f"Built {len(lattice_samples)} lattice samples.")

    # Gộp & shuffle
    all_samples = qa_samples + lattice_samples
    random.shuffle(all_samples)

    print(f"Total multitask samples: {len(all_samples)}")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)

    print(f"Saved multitask data to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
