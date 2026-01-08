import argparse
from pathlib import Path
import yaml
from rich.console import Console
from oran_qa.retrieval.retriever import FaissRetriever
from oran_qa.modeling.generation import load_model_and_tokenizer, generate_answer
from oran_qa.modeling.prompt import build_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    console = Console()

    retr = FaissRetriever(
        index_dir=cfg["retrieval"]["index_dir"],
        embed_model=cfg["retrieval"]["embed_model"],
        top_k=int(cfg["retrieval"]["top_k"]),
    )

    model, tok = load_model_and_tokenizer(
        cfg["model"]["base_model"],
        lora_dir=cfg["model"].get("lora_dir"),
        use_4bit=True,
        bf16=True,
    )

    console.print("[bold]RAG chat ready. Type 'exit' to quit.[/bold]")
    while True:
        q = console.input("\n[bold cyan]Q> [/bold cyan]").strip()
        if not q or q.lower() in {"exit", "quit"}:
            break

        hits = retr.search(q)
        context = "\n\n".join(
            [f"[{h['doc_id']} | {h.get('section','?')}] {h['text']}" for h in hits]
        )

        prompt = build_prompt(q, context, style="evidence_qa_v1")
        ans = generate_answer(
            model,
            tok,
            prompt,
            max_new_tokens=int(cfg["generation"]["max_new_tokens"]),
            temperature=float(cfg["generation"]["temperature"]),
        )

        console.print("\n[bold green]Answer[/bold green]")
        console.print(ans)


if __name__ == "__main__":
    main()
