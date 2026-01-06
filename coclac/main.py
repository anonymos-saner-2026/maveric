# main.py
import os
from typing import List
from openai import OpenAI

from coclac import (
    LLMClient,
    QAExample,
    ClaimExample,
    DataPipeline,
    CoClaCPipeline,
)

class MyLLMClient(LLMClient):
    def __init__(self, model_name: str = "gpt-4o-mini-2024-07-18"):
        super().__init__()
        base_url = "https://api.yescale.io/v1"
        api_key = "sk-AOzQMlsMqmhCbXzCAOOOCkFuOGi9Yx4741EpvrsdWpceYdNM"
        if api_key is None:
            raise RuntimeError("OPENAI_API_KEY is not set")

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.model_name = model_name

    def _call_llm_text(self, prompt: str, max_tokens: int = 512) -> str:
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful, precise AI assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()


def main():
    # 1. Init LLM client
    llm = MyLLMClient()

    # 2. Load QA dataset
    qa_list: List[QAExample] = DataPipeline.load_qa_from_json("qa_dataset.json")
    print(f"Loaded {len(qa_list)} QA examples.")

    # 3. Build claim-level dataset (pseudo-label)
    data_pipeline = DataPipeline(llm)
    claim_examples: List[ClaimExample] = data_pipeline.build_claim_dataset(
        qa_list,
        max_qas=20,   # start small to debug
        shuffle=True,
    )
    DataPipeline.save_claim_dataset("claim_dataset.json", claim_examples)

    # 4. Train CoClaC
    coclac = CoClaCPipeline(llm, use_latent_truth=True)
    coclac.fit_calibrator(claim_examples)

    # 5. Demo
    question = "Einstein was born in which year?"
    answer = "Einstein was born in 1879 in Germany."
    res = coclac.get_answer_confidence(question, answer, agg="min")

    print("\n=== DEMO RESULT ===")
    print("Question:", question)
    print("Answer:", answer)
    print("Claims & confidences:")
    for c, p in zip(res["claims"], res["claim_confidences"]):
        print(f"  - {c} :: {p:.3f}")
    print("Answer-level confidence:", res["answer_confidence"])


if __name__ == "__main__":
    main()
