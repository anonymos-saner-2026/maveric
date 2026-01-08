def build_prompt(question: str, context: str = "", style: str = "evidence_qa_v1", options=None) -> str:
    if style == "mcq_v1":
        assert options is not None and len(options) >= 2, "mcq_v1 requires options"
        opt_lines = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])
        return (
            "You are taking a multiple-choice test.\n"
            "Choose the single best option.\n"
            "Respond with ONLY the option number (1,2,3,4).\n\n"
            f"Question: {question}\n"
            f"Options:\n{opt_lines}\n\n"
            "Answer:"
        )

    # default: evidence QA
    if style == "evidence_qa_v1":
        return (
            "You are a careful assistant for telecom standards QA.\n"
            "Answer ONLY using the provided context. If the answer is not in the context, say: "
            "\"I don't know based on the provided context.\"\n\n"
            "Output format:\n"
            "Answer: <1-4 sentences>\n"
            "Citations: <bullet list of [doc_id | section] you used>\n\n"
            f"Question:\n{question}\n\n"
            f"Context:\n{context}\n\n"
            "Answer:"
        )

    return f"Question: {question}\n\nContext:\n{context}\n\nAnswer:"
