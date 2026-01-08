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
