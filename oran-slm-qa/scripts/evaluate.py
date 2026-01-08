import argparse
from pathlib import Path
import re
import yaml
from oran_qa.utils.io import read_jsonl
from oran_qa.modeling.generation import load_model_and_tokenizer, generate_answer
from oran_qa.modeling.prompt import build_prompt


LABEL_RE = re.compile(r"\b([1-4])\b")


def extract_label(s: str) -> str | None:
    s = s.strip()
    # if model outputs "3" or "Answer: 3" etc.
    m = LABEL_RE.search(s)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--eval_file", type=str, default=None)
    ap.add_argument("--max_n", type=int, default=500)
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
    correct = 0
    total = 0
    invalid = 0

    for ex in examples:
        prompt = build_prompt(
            question=ex["question"],
            style="mcq_v1",
            options=ex["options"],
        )
        pred = generate_answer(model, tok, prompt, max_new_tokens=4, temperature=0.0)
        yhat = extract_label(pred)
        y = str(ex["label"]).strip()

        total += 1
        if yhat is None:
            invalid += 1
            continue
        if yhat == y:
            correct += 1

    acc = correct / total if total else 0.0
    print(f"N={total} | Acc={acc:.4f} | Invalid={invalid} ({invalid/total:.2%})")


if __name__ == "__main__":
    main()
