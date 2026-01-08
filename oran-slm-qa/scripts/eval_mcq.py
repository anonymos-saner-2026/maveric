#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


ANS_RE = re.compile(r"\b([1-9]\d*)\b")


def load_jsonl(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return load_dataset("json", data_files=str(p), split="train")


def parse_answer(text: str):
    # We expect just "1".."4", but be robust.
    if not text:
        return None
    m = ANS_RE.search(text.strip())
    if not m:
        return None
    return m.group(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    ap.add_argument("--adapter", type=str, required=True, help="MCQ SFT adapter dir")
    ap.add_argument("--test", type=str, required=True, help="test_mcq.jsonl")
    ap.add_argument("--max_new_tokens", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=-1)
    args = ap.parse_args()

    ds = load_jsonl(args.test)
    if args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        load_in_4bit=True,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    correct = 0
    total = 0
    bad = 0

    for ex in ds:
        msgs = ex["messages"]
        # last message is assistant with gold label in your data
        gold = None
        for m in reversed(msgs):
            if m.get("role") == "assistant":
                gold = parse_answer(m.get("content", ""))
                break

        # Build prompt: same messages but WITHOUT the assistant answer
        prompt_msgs = [m for m in msgs if m.get("role") != "assistant"]
        if hasattr(tok, "apply_chat_template"):
            prompt = tok.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        else:
            parts = []
            for m in prompt_msgs:
                parts.append(f"{m.get('role','user').upper()}:\n{m.get('content','')}\n")
            prompt = "\n".join(parts) + "\nASSISTANT:\n"

        inputs = tok(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=(args.temperature > 0),
                temperature=args.temperature if args.temperature > 0 else None,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )

        gen = tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        pred = parse_answer(gen)

        total += 1
        if gold is None or pred is None:
            bad += 1
            continue
        if pred == gold:
            correct += 1

    acc = correct / max(1, (total - bad))
    print(json.dumps({
        "total": total,
        "scored": total - bad,
        "bad_unparsed": bad,
        "correct": correct,
        "accuracy": acc,
    }, indent=2))


if __name__ == "__main__":
    main()
