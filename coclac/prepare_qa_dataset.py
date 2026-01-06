# prepare_qa_dataset.py
import json
from io import StringIO

import requests  # pip install requests
import pandas as pd  # pip install pandas


TRUTHFULQA_CSV_URL = (
    "https://raw.githubusercontent.com/sylinrl/TruthfulQA/refs/heads/main/TruthfulQA.csv"
)


def main():
    print(f"Downloading TruthfulQA.csv from:\n  {TRUTHFULQA_CSV_URL}")
    resp = requests.get(TRUTHFULQA_CSV_URL)
    resp.raise_for_status()

    # Read CSV into pandas
    df = pd.read_csv(StringIO(resp.text))

    # Sanity check các cột
    expected_cols = {"Question", "Best Answer"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise RuntimeError(
            f"Missing expected columns in TruthfulQA.csv: {missing}\n"
            f"Columns found: {list(df.columns)}"
        )

    qa_list = []
    for _, row in df.iterrows():
        question = str(row["Question"]).strip()
        best_answer = str(row["Best Answer"]).strip()

        if not question or not best_answer:
            continue

        qa_item = {
            "question": question,
            "gold_answers": [best_answer],  # đúng format coclac.DataPipeline cần
        }
        qa_list.append(qa_item)

    print(f"Total QA examples: {len(qa_list)}")

    # Save ra qa_dataset.json
    out_path = "qa_dataset.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(qa_list, f, ensure_ascii=False, indent=2)

    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
