from datasets import load_dataset
import re
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
    if prompt_style == "mcq_v1":
        prompt = build_prompt(
            question=ex["question"],
            style="mcq_v1",
            options=ex["options"],
        )
        # label should be "1"/"2"/"3"/"4"
        target = str(ex["label"]).strip()
        return {"text": prompt + " " + target + "\n"}

    # fallback evidence QA (giữ nguyên nếu bạn vẫn muốn dùng)
    prompt = build_prompt(ex["question"], ex.get("context", ""), style=prompt_style)
    citations = ex.get("citations", [])
    cite_lines = "\n".join([f"- [{c['doc_id']} | {c.get('section','?')}]" for c in citations]) or "- [unknown]"
    target = f"{ex['answer']}\n\nCitations:\n{cite_lines}\n"
    return {"text": prompt + " " + target}


def run_sft_lora(cfg: dict):
    base_model = cfg["base_model"]
    train_file = cfg["train_file"]
    eval_file = cfg["eval_file"]
    output_dir = cfg["output_dir"]

    prompt_style = cfg.get("prompt_style", "mcq_v1")
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
        return tok(batch["text"], truncation=True, max_length=max_seq_len, padding=False)

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
