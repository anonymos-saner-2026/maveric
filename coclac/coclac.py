# coclac.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import re
import json
import random
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
import math


# ===================== LLM client base =====================
import os
import hashlib

from openai import OpenAI

class SimpleLLMDiskCache:
    """
    Rất đơn giản: cache map từ key -> response, lưu ra 1 file JSON.
    Key = sha256(model_name + max_tokens + prompt).
    Dùng để tránh gọi lại LLM khi đã có kết quả (và survive được crash).
    """

    def __init__(self, cache_path: str = "llm_cache.json"):
        self.cache_path = cache_path
        self._data = {}
        self._dirty = False

        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
                print(f"[LLMCache] Loaded {len(self._data)} entries from {cache_path}")
            except Exception as e:
                print(f"[LLMCache] Failed to load cache from {cache_path}: {e}")
                self._data = {}

    def _make_key(self, model_name: str, prompt: str, max_tokens: int) -> str:
        h = hashlib.sha256()
        h.update(model_name.encode("utf-8"))
        h.update(str(max_tokens).encode("utf-8"))
        h.update(prompt.encode("utf-8"))
        return h.hexdigest()

    def get(self, model_name: str, prompt: str, max_tokens: int) -> Optional[str]:
        key = self._make_key(model_name, prompt, max_tokens)
        return self._data.get(key)

    def set(self, model_name: str, prompt: str, max_tokens: int, value: str):
        key = self._make_key(model_name, prompt, max_tokens)
        if key in self._data:
            return
        self._data[key] = value
        self._dirty = True
        # ghi luôn để nếu crash thì cache vẫn còn
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False)
        except Exception as e:
            print(f"[LLMCache] Failed to write cache: {e}")

class LLMClient:
    """
    Abstracts calls to your LLM.
    You MUST override _call_llm_text() with a real backend.
    """

    def __init__(self):
        pass

    def _call_llm_text(self, prompt: str, max_tokens: int = 512) -> str:
        """
        TODO: Replace with real LLM call (OpenAI, internal, etc.)
        """
        raise NotImplementedError("Implement this with your LLM backend")

    # --------- High-level helpers ---------

    def extract_claims(self, answer: str) -> List[str]:
        """
        Use the LLM to extract atomic factual claims from an answer.
        """
        prompt = f"""
You are a system for analyzing factual statements.

Task:
Given the answer below, extract all CLEAR FACTUAL CLAIMS.

Requirements:
- Each claim must be a SINGLE simple sentence that can stand on its own.
- Each claim should state one specific fact.
- Avoid using "and" / "or" when you can split into multiple sentences.
- Output ONLY the claims, one per line.
- Do NOT add numbering, explanations, or extra text.

Answer:
\"\"\"{answer}\"\"\"

Now list the factual claims, one claim per line.
"""
        raw = self._call_llm_text(prompt, max_tokens=256)
        claims = [line.strip() for line in raw.split("\n") if line.strip()]
        return claims

    def generate_variant(self, claim: str, variant_type: str) -> str:
        """
        variant_type ∈ {"paraphrase", "weakening", "strengthening", "negation"}
        """
        if variant_type == "paraphrase":
            prompt = f"""
Rewrite the following sentence with the SAME meaning.
Requirements:
- Keep it short and clear.
- Output EXACTLY ONE sentence.

Sentence:
\"\"\"{claim}\"\"\"
"""
        elif variant_type == "weakening":
            prompt = f"""
Write a LESS SPECIFIC sentence that is GUARANTEED to be TRUE if the following sentence is true.
Requirements:
- You may remove details to make the statement weaker.
- Do NOT introduce any new information.

Original sentence:
\"\"\"{claim}\"\"\"
"""
        elif variant_type == "strengthening":
            prompt = f"""
Write a MORE SPECIFIC sentence that ADDS DETAILS but MUST remain TRUE if the following sentence is true.
Requirements:
- You may only add details that are logically implied by the original sentence.
- Do NOT introduce new information that is not implied by the original sentence.

Original sentence:
\"\"\"{claim}\"\"\"
"""
        elif variant_type == "negation":
            prompt = f"""
Write a NEGATION of the following sentence in terms of its factual content.
Requirements:
- Keep the same subject and main content where possible.
- Flip the meaning to express that the original statement is NOT true.

Original sentence:
\"\"\"{claim}\"\"\"
"""
        else:
            raise ValueError(f"Unknown variant_type: {variant_type}")

        raw = self._call_llm_text(prompt, max_tokens=128)
        line = raw.strip().split("\n")[0].strip()
        return line

    def estimate_truth_prob(self, statement: str) -> float:
        """
        Ask the LLM to estimate P(True) for a statement. Return float ∈ [0, 1].
        """
        prompt = f"""
Consider the following statement:
\"\"\"{statement}\"\"\"

How likely do you think this statement is FACTUALLY CORRECT?

Answer with a SINGLE real number in [0, 1], and NOTHING ELSE.
Example: 0.73
"""
        raw = self._call_llm_text(prompt, max_tokens=16)
        match = re.search(r"([01](?:\.\d+)?)", raw)
        if not match:
            return 0.5
        try:
            val = float(match.group(1))
        except ValueError:
            val = 0.5
        val = max(0.0, min(1.0, val))
        return val


# ===================== Data structures =====================

@dataclass
class QAExample:
    question: str
    gold_answers: List[str]
    model_answer: Optional[str] = None


@dataclass
class ClaimExample:
    question: str
    answer: str
    claim_text: str
    label: int  # 1 = correct, 0 = incorrect


@dataclass
class ClaimLattice:
    original: str
    paraphrase: str
    weakening: str
    strengthening: str
    negation: str


@dataclass
class ClaimBeliefs:
    p_original: float
    p_paraphrase: float
    p_weakening: float
    p_strengthening: float
    p_negation: float

@dataclass
class BeliefVector:
    """
    Raw 5-D belief vector for a claim: [p_c, p_para, p_weak, p_str, p_neg]
    """
    vec: np.ndarray  # shape (5,)

# ===================== Labeler: QA -> ClaimExample =====================

class ClaimLabeler:
    """
    Use an LLM to label each claim as correct/incorrect,
    based on the question + gold answer(s).
    """

    def __init__(self, llm: LLMClient, sleep_sec: float = 0.0):
        self.llm = llm
        self.sleep_sec = sleep_sec

    def label_claim(self, qa: QAExample, claim: str) -> int:
        if len(qa.gold_answers) == 1:
            gold_str = qa.gold_answers[0]
        else:
            gold_str = "\n".join(f"- {a}" for a in qa.gold_answers)

        prompt = f"""
You are a system for checking the factual correctness of a claim.

Information:
- Question (Q):
\"\"\"{qa.question}\"\"\"

- Reference answer(s) (gold answers):
\"\"\"{gold_str}\"\"\"

- Claim to evaluate:
\"\"\"{claim}\"\"\"

Task:
1. Using ONLY the question and the gold answers, decide whether the claim is:
   - COMPLETELY CORRECT (consistent with the gold answers, no contradiction),
   - INCORRECT / CONTRADICTS the gold answers,
   - or there is NOT ENOUGH INFORMATION to decide.

2. Output exactly ONE of the following tokens:
   - CORRECT
   - INCORRECT
   - UNKNOWN

Do NOT output anything else.
"""
        raw = self.llm._call_llm_text(prompt, max_tokens=32).strip().upper()
        if self.sleep_sec > 0:
            time.sleep(self.sleep_sec)

        if "CORRECT" in raw and "INCORRECT" not in raw and "UNKNOWN" not in raw:
            return 1
        elif "INCORRECT" in raw:
            return 0
        else:
            # UNKNOWN -> treat as 0 (conservative), or you can choose to skip instead
            return 0

    def label_claims_for_qa(self, qa: QAExample, claims: List[str]) -> List[ClaimExample]:
        examples: List[ClaimExample] = []
        for c in claims:
            label = self.label_claim(qa, c)
            ex = ClaimExample(
                question=qa.question,
                answer=qa.model_answer or "",
                claim_text=c,
                label=label,
            )
            examples.append(ex)
        return examples


# ===================== Data pipeline: QA -> ClaimExample =====================

class DataPipeline:
    """
    From raw QA (question + gold) -> model_answer -> ClaimExample (with label)
    to train CoClaC.
    """

    def __init__(self, llm: LLMClient, max_answer_tokens: int = 256):
        self.llm = llm
        self.max_answer_tokens = max_answer_tokens
        self.claim_labeler = ClaimLabeler(llm)

    def generate_answer(self, qa: QAExample) -> str:
        prompt = f"""
Question:
\"\"\"{qa.question}\"\"\"

Provide a short, precise answer focusing on factual correctness.
"""
        answer = self.llm._call_llm_text(prompt, max_tokens=self.max_answer_tokens).strip()
        return answer

    def ensure_answers(self, qa_list: List[QAExample]) -> None:
        for qa in qa_list:
            if qa.model_answer is None:
                qa.model_answer = self.generate_answer(qa)

    def build_claim_dataset(
        self,
        qa_list: List[QAExample],
        max_qas: Optional[int] = None,
        shuffle: bool = True,
    ) -> List[ClaimExample]:
        if shuffle:
            random.shuffle(qa_list)
        if max_qas is not None:
            qa_list = qa_list[:max_qas]

        self.ensure_answers(qa_list)

        all_claim_examples: List[ClaimExample] = []
        for idx, qa in enumerate(qa_list):
            print(f"[DataPipeline] QA #{idx+1}/{len(qa_list)}")
            claims = self.llm.extract_claims(qa.model_answer or "")
            claim_examples = self.claim_labeler.label_claims_for_qa(qa, claims)
            all_claim_examples.extend(claim_examples)

        print(f"[DataPipeline] Total ClaimExamples: {len(all_claim_examples)}")
        return all_claim_examples

    @staticmethod
    def load_qa_from_json(path: str) -> List[QAExample]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        qa_list: List[QAExample] = []
        for item in data:
            q = item["question"]
            gold = item["gold_answers"]
            qa = QAExample(question=q, gold_answers=gold, model_answer=None)
            qa_list.append(qa)
        return qa_list

    @staticmethod
    def save_claim_dataset(path: str, claim_examples: List[ClaimExample]) -> None:
        out = []
        for ex in claim_examples:
            out.append(
                {
                    "question": ex.question,
                    "answer": ex.answer,
                    "claim_text": ex.claim_text,
                    "label": int(ex.label),
                }
            )
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)


# ===================== Lattice + beliefs + features =====================

class LatticeGenerator:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def build_lattice(self, claim: str) -> ClaimLattice:
        paraphrase = self.llm.generate_variant(claim, "paraphrase")
        weakening = self.llm.generate_variant(claim, "weakening")
        strengthening = self.llm.generate_variant(claim, "strengthening")
        negation = self.llm.generate_variant(claim, "negation")
        return ClaimLattice(
            original=claim,
            paraphrase=paraphrase,
            weakening=weakening,
            strengthening=strengthening,
            negation=negation,
        )


class BeliefElicitor:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def elicit_beliefs(self, lattice: ClaimLattice) -> ClaimBeliefs:
        p_orig = self.llm.estimate_truth_prob(lattice.original)
        p_para = self.llm.estimate_truth_prob(lattice.paraphrase)
        p_weak = self.llm.estimate_truth_prob(lattice.weakening)
        p_str = self.llm.estimate_truth_prob(lattice.strengthening)
        p_neg = self.llm.estimate_truth_prob(lattice.negation)
        return ClaimBeliefs(
            p_original=p_orig,
            p_paraphrase=p_para,
            p_weakening=p_weak,
            p_strengthening=p_str,
            p_negation=p_neg,
        )


class FeatureBuilder:
    def build_features(self, beliefs: ClaimBeliefs) -> np.ndarray:
        p_c = beliefs.p_original
        p_para = beliefs.p_paraphrase
        p_weak = beliefs.p_weakening
        p_str = beliefs.p_strengthening
        p_neg = beliefs.p_negation

        m_neg = p_c - p_neg
        g_weak = p_weak - p_c
        g_str = p_c - p_str
        g_para = abs(p_c - p_para)

        viol_neg = abs(p_c + p_neg - 1.0)
        viol_weak = max(0.0, p_c - p_weak)
        viol_str = max(0.0, p_str - p_c)

        feats = np.array([
            p_c,
            p_para,
            p_weak,
            p_str,
            p_neg,
            m_neg,
            g_weak,
            g_str,
            g_para,
            viol_neg,
            viol_weak,
            viol_str,
        ], dtype=np.float32)
        return feats


# ===================== Calibrator + CoClaC pipeline =====================

class Calibrator:
    def __init__(self, penalty: str = "l2", C: float = 1.0):
        self.model = LogisticRegression(
            penalty=penalty,
            C=C,
            solver="lbfgs",
            max_iter=1000,
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

class LatentTruthCalibrator:
    """
    Simple generative latent-truth model on raw 5-D belief vectors.

    Assumptions:
    - For T=1 (claim true), belief vectors ~ N(mu_pos, diag(sigma_pos^2))
    - For T=0 (claim false), belief vectors ~ N(mu_neg, diag(sigma_neg^2))
    - We estimate mu_pos, mu_neg, sigma_pos, sigma_neg from labeled data.
    - At inference, we compute P(T=1 | v) by Gaussian Naive Bayes.
    """

    def __init__(self, eps: float = 1e-3):
        self.eps = eps
        self.mu_pos = None
        self.mu_neg = None
        self.var_pos = None
        self.var_neg = None
        self.pi_pos = None  # prior P(T=1)
        self._fitted = False

    def fit(self, V: np.ndarray, y: np.ndarray):
        """
        V: shape (N, 5) belief vectors
        y: shape (N,), labels in {0,1}
        """
        y = y.astype(np.int32)
        pos_mask = y == 1
        neg_mask = y == 0

        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            raise ValueError("Need both positive and negative examples for latent-truth calibrator.")

        V_pos = V[pos_mask]
        V_neg = V[neg_mask]

        self.mu_pos = V_pos.mean(axis=0)
        self.mu_neg = V_neg.mean(axis=0)

        # unbiased var, add epsilon for numerical stability
        self.var_pos = V_pos.var(axis=0) + self.eps
        self.var_neg = V_neg.var(axis=0) + self.eps

        self.pi_pos = float(pos_mask.mean())
        self._fitted = True

    def _log_gaussian(self, V: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
        """
        Diagonal Gaussian log-likelihood for each row in V.
        """
        # log N(x; mu, diag(var)) = -0.5 * [ sum((x-mu)^2/var) + sum(log(2pi*var)) ]
        diff = V - mu[None, :]
        quad = (diff ** 2) / var[None, :]
        log_det = np.log(2 * np.pi * var).sum()
        log_prob = -0.5 * (quad.sum(axis=1) + log_det)
        return log_prob

    def predict_proba(self, V: np.ndarray) -> np.ndarray:
        """
        V: shape (N, 5)
        Return: shape (N,) = P(T=1 | V)
        """
        if not self._fitted:
            raise RuntimeError("LatentTruthCalibrator not fitted.")
        V = np.asarray(V, dtype=np.float32)

        log_p_pos = self._log_gaussian(V, self.mu_pos, self.var_pos) + math.log(self.pi_pos + self.eps)
        log_p_neg = self._log_gaussian(V, self.mu_neg, self.var_neg) + math.log(1.0 - self.pi_pos + self.eps)

        # normalize
        max_log = np.maximum(log_p_pos, log_p_neg)
        log_p_pos_norm = log_p_pos - max_log
        log_p_neg_norm = log_p_neg - max_log

        p_pos = np.exp(log_p_pos_norm)
        p_neg = np.exp(log_p_neg_norm)
        denom = p_pos + p_neg + self.eps
        return p_pos / denom

class CoClaCPipeline:
    def __init__(self, llm: LLMClient, use_latent_truth: bool = False):
        self.llm = llm
        self.lattice_gen = LatticeGenerator(llm)
        self.belief_elicitor = BeliefElicitor(llm)
        self.feature_builder = FeatureBuilder()
        self.use_latent_truth = use_latent_truth

        if use_latent_truth:
            self.latent_calibrator = LatentTruthCalibrator()
            self.calibrator = None
        else:
            self.calibrator = Calibrator()
            self.latent_calibrator = None

        self._fitted = False

    def build_features_for_claim(self, claim_text: str) -> Tuple[np.ndarray, ClaimLattice, ClaimBeliefs]:
        lattice = self.lattice_gen.build_lattice(claim_text)
        beliefs = self.belief_elicitor.elicit_beliefs(lattice)
        feats = self.feature_builder.build_features(beliefs)
        return feats, lattice, beliefs
    def build_belief_vector_for_claim(self, claim_text: str) -> Tuple[BeliefVector, ClaimLattice, ClaimBeliefs]:
        lattice = self.lattice_gen.build_lattice(claim_text)
        beliefs = self.belief_elicitor.elicit_beliefs(lattice)
        v = np.array([
            beliefs.p_original,
            beliefs.p_paraphrase,
            beliefs.p_weakening,
            beliefs.p_strengthening,
            beliefs.p_negation,
        ], dtype=np.float32)
        return BeliefVector(vec=v), lattice, beliefs
    def fit_calibrator(self, claim_examples: List[ClaimExample]) -> None:
        X_list, y_list = [], []

        for ex in claim_examples:
            if self.use_latent_truth:
                belief_vec, _, _ = self.build_belief_vector_for_claim(ex.claim_text)
                X_list.append(belief_vec.vec)
            else:
                feats, _, _ = self.build_features_for_claim(ex.claim_text)
                X_list.append(feats)
            y_list.append(ex.label)

        X = np.stack(X_list, axis=0)
        y = np.array(y_list, dtype=np.int32)

        if self.use_latent_truth:
            self.latent_calibrator.fit(X, y)
            train_pred = self.latent_calibrator.predict_proba(X)
        else:
            self.calibrator.fit(X, y)
            train_pred = self.calibrator.predict_proba(X)

        self._fitted = True
        brier = brier_score_loss(y, train_pred)
        print(f"[CoClaC] Training Brier score: {brier:.4f}, n={len(y)}")


    def predict_claim_confidence(self, claim_text: str) -> float:
        if not self._fitted:
            raise RuntimeError("Calibrator not fitted.")

        if self.use_latent_truth:
            belief_vec, _, _ = self.build_belief_vector_for_claim(claim_text)
            proba = self.latent_calibrator.predict_proba(belief_vec.vec[None, :])[0]
        else:
            feats, _, _ = self.build_features_for_claim(claim_text)
            proba = self.calibrator.predict_proba(feats[None, :])[0]

        return float(proba)


    def get_answer_confidence(self, question: str, answer: str, agg: str = "min") -> Dict:
        if not self._fitted:
            raise RuntimeError("Calibrator not fitted.")

        claims = self.llm.extract_claims(answer)
        if not claims:
            return {
                "claims": [],
                "claim_confidences": [],
                "answer_confidence": 0.0,
            }

        claim_confidences = []
        for c in claims:
            p = self.predict_claim_confidence(c)
            claim_confidences.append(p)

        arr = np.array(claim_confidences)
        if agg == "min":
            ans_conf = float(arr.min())
        elif agg == "mean":
            ans_conf = float(arr.mean())
        else:
            raise ValueError(f"Unknown agg: {agg}")

        return {
            "claims": claims,
            "claim_confidences": arr.tolist(),
            "answer_confidence": ans_conf,
        }


# ===================== OpenAI-based implementation (optional) =====================

import os
from openai import OpenAI
class OpenAILLMClient(LLMClient):
    """
    Concrete implementation of LLMClient using OpenAI-compatible API.

    It reads:
    - OPENAI_API_KEY   from environment
    - OPENAI_BASE_URL  from environment (defaults to https://api.yescale.io/v1)
    - OPENAI_MODEL     optional override for model name
    """

    def __init__(self, model_name: str = None, cache_path: str = "llm_cache.json"):
        super().__init__()

        base_url = "https://api.yescale.io/v1"
        api_key = "sk-AOzQMlsMqmhCbXzCAOOOCkFuOGi9Yx4741EpvrsdWpceYdNM"
        if api_key is None:
            raise RuntimeError("OPENAI_API_KEY is not set in environment variables")

        # model name: env > arg > default
        env_model = os.environ.get("OPENAI_MODEL")
        self.model_name = model_name or env_model or "gpt-4o-mini-2024-07-18"

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

        # disk cache
        self.cache = SimpleLLMDiskCache(cache_path=cache_path)

    def _call_llm_text(self, prompt: str, max_tokens: int = 512) -> str:
        """
        Low-level call used by all helper methods (extract_claims, generate_variant, ...).
        Thêm:
          - Disk cache (prompt-level)
          - Retry với backoff khi lỗi tạm thời (ví dụ: model unavailable)
        """
        # 1) thử cache trước
        cached = self.cache.get(self.model_name, prompt, max_tokens)
        if cached is not None:
            return cached

        # 2) không có cache -> gọi API với retry
        max_retries = 5
        backoff = 2.0  # giây

        last_err = None
        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful, precise AI assistant.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=0.0,  # make outputs stable for labeling/calibration
                )
                text = resp.choices[0].message.content.strip()

                # lưu cache
                self.cache.set(self.model_name, prompt, max_tokens, text)
                return text

            except Exception as e:
                last_err = e
                msg = str(e)

                # Nếu là lỗi kiểu "currently unavailable" thì retry với backoff
                # Cứ retry cho mọi lỗi tạm thời; nhưng nếu hết retry thì raise.
                print(
                    f"[OpenAILLMClient] Error on attempt {attempt+1}/{max_retries}: {msg}"
                )

                if attempt == max_retries - 1:
                    # hết retry, ném lỗi
                    raise

                # ngủ rồi thử lại
                time.sleep(backoff * (2 ** attempt))

        # theoretically không tới đây
        if last_err is not None:
            raise last_err
        raise RuntimeError("Unknown error in _call_llm_text")
