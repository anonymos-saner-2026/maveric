#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel


def load_chat_jsonl(path: str):
    # datasets can read jsonl directly, but we validate schema lightly
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return load_dataset("json", data_files=str(p), split="train")


def extract_text_from_messages(example, tokenizer):
    """
    Convert OpenAI-style chat messages to a single training string.
    We train causal LM to reproduce assistant content given the prior context.
    """
    msgs = example.get("messages", [])
    # Tokenizer chat template is best if available
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=False,  # include assistant content if present in msgs
        )
        return {"text": text}

    # Fallback: simple concat (works but less optimal)
    parts = []
    for m in msgs:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"{role.upper()}:\n{content}\n")
    return {"text": "\n".join(parts)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    ap.add_argument("--init_adapter", type=str, default=None, help="Optional DAPT adapter to merge into base before SFT")
    ap.add_argument("--train", type=str, required=True)
    ap.add_argument("--valid", type=str, default=None)
    ap.add_argument("--out", type=str, default="outputs/mcq_adapter")

    # training
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bsz", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--logging_steps", type=int, default=20)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--max_steps", type=int, default=-1)

    # seq
    ap.add_argument("--max_seq_len", type=int, default=2048)

    # LoRA
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--target_modules", type=str, default="all-linear")

    # compute
    ap.add_argument("--use_4bit", action="store_true", help="Use QLoRA (4-bit). Recommended for 4B on single GPU.")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    train_ds = load_chat_jsonl(args.train)
    train_ds = train_ds.map(lambda ex: extract_text_from_messages(ex, tokenizer), remove_columns=train_ds.column_names)

    eval_ds = None
    if args.valid:
        eval_ds = load_chat_jsonl(args.valid)
        eval_ds = eval_ds.map(lambda ex: extract_text_from_messages(ex, tokenizer), remove_columns=eval_ds.column_names)

    # Load base model
    kwargs = dict(
        trust_remote_code=True,
        device_map="auto",
    )
    if args.use_4bit:
        kwargs.update(dict(load_in_4bit=True, torch_dtype=torch.bfloat16))
    else:
        kwargs.update(dict(torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32))

    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)

    # If init_adapter provided (DAPT), merge into base before applying new LoRA for SFT
    if args.init_adapter:
        init_path = Path(args.init_adapter)
        if not init_path.exists():
            raise FileNotFoundError(init_path)
        model = PeftModel.from_pretrained(model, str(init_path))
        model = model.merge_and_unload()  # bake DAPT knowledge into weights

    # Prepare for LoRA fine-tune
    if args.use_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # SFT trainer
    train_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs if args.max_steps <= 0 else 1.0,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=torch.cuda.is_available(),
        fp16=False,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit" if args.use_4bit else "adamw_torch",
        report_to="none",
        evaluation_strategy="steps" if eval_ds is not None else "no",
        eval_steps=args.save_steps if eval_ds is not None else None,
    )

    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_text_field="text",
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_len,
        packing=True,  # pack multiple samples into one sequence
    )

    trainer.train()
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    # Save a small run metadata
    meta = {
        "base_model": args.model,
        "init_adapter": args.init_adapter,
        "train": args.train,
        "valid": args.valid,
        "use_4bit": bool(args.use_4bit),
        "max_seq_len": args.max_seq_len,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved MCQ SFT adapter to: {out_dir}")


if __name__ == "__main__":
    main()
