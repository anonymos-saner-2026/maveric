#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
E3: Multi-task fine-tuning for CoClaC-style LLM

Tasks:
  1) QA (causal LM):
     - Input: "Q: {question}\nA: {answer}"
     - Loss: standard LM (cross-entropy) on the whole sequence.

  2) Lattice self-eval + logic:
     - Input text:
         "Evaluate factual correctness of the following claims.
          [CLAIM_C] {claim}
          [CLAIM_PARA] {paraphrase}
          [CLAIM_WEAK] {weakening}
          [CLAIM_STR] {strengthening}
          [CLAIM_NEG] {negation}
          End."
     - Model:
         - Encode with base LLM.
         - For each of the 5 special tokens, take hidden state.
         - Self-eval head -> 5 probs in (0,1):
             p_c, p_para, p_weak, p_str, p_neg
     - Loss:
         - BCE(p_c, label) where label ∈ {0,1} (pseudo-label correctness of original claim).
         - Logic loss:
              L_neg  = |p_c + p_neg - 1|
              L_weak = max(0, p_c - p_weak)
              L_str  = max(0, p_str - p_c)
           L_logic = α_neg * L_neg + α_weak * L_weak + α_str * L_str

Total loss:
  L = λ_qa * L_LM + λ_self * L_self + λ_logic * L_logic

You need a file `multitask_data.json` with items of form:

  {"task": "qa", "question": "...", "answer": "..."}
  {"task": "lattice",
   "claim": "...", "paraphrase": "...", "weakening": "...",
   "strengthening": "...", "negation": "...", "label": 0 or 1}

The script will:
  - Load this JSON
  - Split into train/eval
  - Train with HuggingFace Transformers Trainer
  - Save final checkpoint under `mt_coclac_checkpoints/final/`

After that, you can:
  - Load this model with Transformers for further experiments, OR
  - Load it into vLLM for high-throughput inference.
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizerBase,
    DataCollator,
)


# ======================= Data definitions =======================


@dataclass
class QASample:
    question: str
    answer: str


@dataclass
class LatticeSample:
    claim: str
    paraphrase: str
    weakening: str
    strengthening: str
    negation: str
    label: int  # 1 = correct, 0 = incorrect


@dataclass
class MultiTaskItem:
    task: str  # "qa" or "lattice"
    qa: Optional[QASample] = None
    lattice: Optional[LatticeSample] = None


def load_multitask_data(path: str) -> List[MultiTaskItem]:
    """
    Load multitask_data.json and convert to MultiTaskItem list.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    items: List[MultiTaskItem] = []
    for obj in raw:
        task = obj["task"]
        if task == "qa":
            qa = QASample(
                question=obj["question"],
                answer=obj["answer"],
            )
            items.append(MultiTaskItem(task="qa", qa=qa))
        elif task == "lattice":
            lat = LatticeSample(
                claim=obj["claim"],
                paraphrase=obj["paraphrase"],
                weakening=obj["weakening"],
                strengthening=obj["strengthening"],
                negation=obj["negation"],
                label=int(obj["label"]),
            )
            items.append(MultiTaskItem(task="lattice", lattice=lat))
        else:
            raise ValueError(f"Unknown task type in multitask_data.json: {task}")
    return items


class MultiTaskCoclacDataset(Dataset):
    def __init__(self, items: List[MultiTaskItem]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx) -> MultiTaskItem:
        return self.items[idx]


# ======================= Collator =======================


class CoclacDataCollator(DataCollator):
    """
    Collator:
      - For QA items: build "Q: ...\nA: ..." sequence and standard LM labels.
      - For lattice items: build sequence containing [CLAIM_*] tokens, mask LM labels,
        provide positions of claim tokens + claim label for self-eval & logic loss.

    We add 5 special tokens (must be in tokenizer):
      [CLAIM_C], [CLAIM_PARA], [CLAIM_WEAK], [CLAIM_STR], [CLAIM_NEG]
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length_qa: int = 512,
        max_length_lattice: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_length_qa = max_length_qa
        self.max_length_lattice = max_length_lattice

        self.claim_tokens = [
            "[CLAIM_C]",
            "[CLAIM_PARA]",
            "[CLAIM_WEAK]",
            "[CLAIM_STR]",
            "[CLAIM_NEG]",
        ]
        self.claim_token_ids = [
            self.tokenizer.convert_tokens_to_ids(tok) for tok in self.claim_tokens
        ]

    def __call__(self, features: List[MultiTaskItem]) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        task_type_list = []      # 0 = QA, 1 = lattice
        claim_positions_list = []
        claim_labels_list = []

        for item in features:
            if item.task == "qa":
                # Build QA text
                q = item.qa.question
                a = item.qa.answer
                text = f"Q: {q}\nA: {a}"

                tok = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=self.max_length_qa,
                    return_tensors="pt",
                )
                input_ids = tok["input_ids"].squeeze(0)
                attention_mask = tok["attention_mask"].squeeze(0)

                # Standard LM labels = input_ids
                labels = input_ids.clone()

                task_type = 0
                claim_positions = [-1] * 5
                claim_label = 0

            elif item.task == "lattice":
                lat = item.lattice
                text_parts = [
                    "Evaluate factual correctness of the following claims.\n",
                    "[CLAIM_C] " + lat.claim + "\n",
                    "[CLAIM_PARA] " + lat.paraphrase + "\n",
                    "[CLAIM_WEAK] " + lat.weakening + "\n",
                    "[CLAIM_STR] " + lat.strengthening + "\n",
                    "[CLAIM_NEG] " + lat.negation + "\n",
                    "End.",
                ]
                text = "".join(text_parts)

                tok = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=self.max_length_lattice,
                    return_tensors="pt",
                )
                input_ids = tok["input_ids"].squeeze(0)
                attention_mask = tok["attention_mask"].squeeze(0)

                # Mask out LM labels: we don't train language modeling on this task
                labels = torch.full_like(input_ids, -100)

                task_type = 1
                claim_positions = []
                for claim_tok_id in self.claim_token_ids:
                    pos = (input_ids == claim_tok_id).nonzero(as_tuple=True)[0]
                    if len(pos) == 0:
                        # If tokenizer splits special token (shouldn't happen), fallback 0
                        claim_positions.append(0)
                    else:
                        claim_positions.append(int(pos[0].item()))

                claim_label = int(lat.label)

            else:
                raise ValueError(f"Unknown task type: {item.task}")

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)
            task_type_list.append(task_type)
            claim_positions_list.append(claim_positions)
            claim_labels_list.append(claim_label)

        # Pad sequences
        batch = self.tokenizer.pad(
            {
                "input_ids": input_ids_list,
                "attention_mask": attention_mask_list,
                "labels": labels_list,
            },
            padding=True,
            return_tensors="pt",
        )

        batch["task_type"] = torch.tensor(task_type_list, dtype=torch.long)
        batch["claim_positions"] = torch.tensor(claim_positions_list, dtype=torch.long)
        batch["claim_labels"] = torch.tensor(claim_labels_list, dtype=torch.float32)

        return batch


# ======================= Model wrapper =======================


class CoclacMultiTaskModel(torch.nn.Module):
    """
    Wrap AutoModelForCausalLM + a self-eval head.

    forward() expects:
      - input_ids, attention_mask, labels   (standard LM)
      - task_type (0/1), claim_positions, claim_labels
      - loss weights + logic hyperparams
    """

    def __init__(self, base_model_name: str, tokenizer: PreTrainedTokenizerBase):
        super().__init__()
        self.lm = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            output_hidden_states=True,
        )
        hidden_size = self.lm.config.hidden_size
        self.self_eval_head = torch.nn.Linear(hidden_size, 1)

        # Resize token embeddings because we added special tokens
        self.lm.resize_token_embeddings(len(tokenizer))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        task_type: torch.Tensor,
        claim_positions: torch.Tensor,
        claim_labels: torch.Tensor,
        qa_loss_weight: float = 1.0,
        self_loss_weight: float = 1.0,
        logic_loss_weight: float = 1.0,
        alpha_neg: float = 1.0,
        alpha_weak: float = 0.5,
        alpha_str: float = 0.5,
    ) -> Dict[str, Any]:

        # Standard LM pass
        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        lm_loss = outputs.loss
        hidden_states = outputs.hidden_states[-1]  # (batch, seq_len, hidden)

        # Self-eval + logic only for lattice items
        lattice_mask = (task_type == 1)
        if lattice_mask.any():
            idx = lattice_mask.nonzero(as_tuple=True)[0]      # indices of lattice examples
            pos = claim_positions[idx]                        # (B_lat, 5)
            lat_hidden = hidden_states[idx]                   # (B_lat, seq_len, H)

            B_lat, num_claims = pos.shape
            H = lat_hidden.size(-1)

            # Gather hidden states at claim token positions
            pos_expanded = pos.unsqueeze(-1).expand(-1, -1, H)      # (B_lat, 5, H)
            lat_hidden_expanded = lat_hidden.gather(
                dim=1,
                index=pos_expanded,
            )                                                        # (B_lat, 5, H)

            logits = self.self_eval_head(lat_hidden_expanded).squeeze(-1)  # (B_lat, 5)
            probs = torch.sigmoid(logits)                                   # (B_lat, 5)

            p_c = probs[:, 0]
            p_para = probs[:, 1]
            p_weak = probs[:, 2]
            p_str = probs[:, 3]
            p_neg = probs[:, 4]

            labels_lat = claim_labels[idx]  # (B_lat,)

            # Self-eval loss: only p_c supervised by label
            self_eval_loss = torch.nn.functional.binary_cross_entropy(
                p_c, labels_lat, reduction="mean"
            )

            # Logic losses
            viol_neg = torch.abs(p_c + p_neg - 1.0)
            viol_weak = torch.clamp(p_c - p_weak, min=0.0)
            viol_str = torch.clamp(p_str - p_c, min=0.0)

            loss_neg = viol_neg.mean()
            loss_weak = viol_weak.mean()
            loss_str = viol_str.mean()

            logic_loss = alpha_neg * loss_neg + alpha_weak * loss_weak + alpha_str * loss_str
        else:
            # No lattice examples in batch
            device = input_ids.device
            self_eval_loss = torch.tensor(0.0, device=device)
            logic_loss = torch.tensor(0.0, device=device)

        total_loss = (
            qa_loss_weight * lm_loss
            + self_loss_weight * self_eval_loss
            + logic_loss_weight * logic_loss
        )

        return {
            "loss": total_loss,
            "lm_loss": lm_loss,
            "self_eval_loss": self_eval_loss,
            "logic_loss": logic_loss,
        }


# ======================= Trainer wrapper =======================


class CoclacTrainer(Trainer):
    def __init__(
        self,
        qa_loss_weight: float = 1.0,
        self_loss_weight: float = 1.0,
        logic_loss_weight: float = 1.0,
        alpha_neg: float = 1.0,
        alpha_weak: float = 0.5,
        alpha_str: float = 0.5,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.qa_loss_weight = qa_loss_weight
        self.self_loss_weight = self_loss_weight
        self.logic_loss_weight = logic_loss_weight
        self.alpha_neg = alpha_neg
        self.alpha_weak = alpha_weak
        self.alpha_str = alpha_str

    def compute_loss(self, model, inputs, return_outputs=False):
        # Inputs contain the batch from collator
        labels = inputs.pop("labels")
        task_type = inputs.pop("task_type")
        claim_positions = inputs.pop("claim_positions")
        claim_labels = inputs.pop("claim_labels")

        outputs = model(
            labels=labels,
            task_type=task_type,
            claim_positions=claim_positions,
            claim_labels=claim_labels,
            qa_loss_weight=self.qa_loss_weight,
            self_loss_weight=self.self_loss_weight,
            logic_loss_weight=self.logic_loss_weight,
            alpha_neg=self.alpha_neg,
            alpha_weak=self.alpha_weak,
            alpha_str=self.alpha_str,
            **inputs,
        )
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss


# ======================= Main =======================


def main():
    # ---------- Config ----------
    base_model_name = os.environ.get("BASE_MODEL_NAME", "gpt2")  # CHANGE THIS
    multitask_path = "multitask_data.json"
    output_dir = "mt_coclac_checkpoints"

    num_train_epochs = 3
    train_batch_size = 1          # increase if GPU allows
    grad_accum_steps = 8
    lr = 5e-5
    max_length_qa = 512
    max_length_lattice = 512

    qa_loss_weight = 1.0
    self_loss_weight = 1.0
    logic_loss_weight = 1.0

    alpha_neg = 1.0
    alpha_weak = 0.5
    alpha_str = 0.5

    # ---------- Load data ----------
    items = load_multitask_data(multitask_path)
    print(f"Loaded {len(items)} multitask items from {multitask_path}")

    rng = np.random.RandomState(42)
    indices = np.arange(len(items))
    rng.shuffle(indices)
    split = int(0.9 * len(items))
    train_idx = indices[:split]
    eval_idx = indices[split:]

    train_items = [items[i] for i in train_idx]
    eval_items = [items[i] for i in eval_idx]

    train_dataset = MultiTaskCoclacDataset(train_items)
    eval_dataset = MultiTaskCoclacDataset(eval_items)

    # ---------- Tokenizer + special tokens ----------
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    special_tokens = [
        "[CLAIM_C]",
        "[CLAIM_PARA]",
        "[CLAIM_WEAK]",
        "[CLAIM_STR]",
        "[CLAIM_NEG]",
    ]
    tokenizer.add_tokens(special_tokens, special_tokens=True)

    # ---------- Model ----------
    model = CoclacMultiTaskModel(base_model_name, tokenizer)

    # ---------- Collator ----------
    collator = CoclacDataCollator(
        tokenizer=tokenizer,
        max_length_qa=max_length_qa,
        max_length_lattice=max_length_lattice,
    )

    # ---------- Training args ----------
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=train_batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=lr,
        logging_steps=50,
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=500,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),  # good for A100/4090 etc
        report_to=[],  # disable wandb/hf logging
    )

    # ---------- Trainer ----------
    trainer = CoclacTrainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        qa_loss_weight=qa_loss_weight,
        self_loss_weight=self_loss_weight,
        logic_loss_weight=logic_loss_weight,
        alpha_neg=alpha_neg,
        alpha_weak=alpha_weak,
        alpha_str=alpha_str,
    )

    # ---------- Train ----------
    trainer.train()

    # Save final model + tokenizer
    final_dir = os.path.join(output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print("Training finished. Model saved to", final_dir)


if __name__ == "__main__":
    main()
