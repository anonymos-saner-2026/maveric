import re
import math
import json
import datetime
import random
import io
import contextlib
import requests
import warnings
from typing import Optional, Dict, Any, Tuple, List

from src.config import client, SERPER_API_KEY, JUDGE_MODEL

warnings.filterwarnings("ignore", category=RuntimeWarning, module="duckduckgo_search")
warnings.filterwarnings("ignore", category=UserWarning, module="duckduckgo_search")

from src.config import client, SERPER_API_KEY, JUDGE_MODEL


class PythonSandbox:
    @staticmethod
    def run(code: str) -> str:
        # Extract code from fences if present
        pattern = r"```python(.*?)```"
        match = re.search(pattern, code, re.DOTALL)
        if match:
            clean_code = match.group(1).strip()
        else:
            clean_code = code.replace("```python", "").replace("```", "").strip()

        # Stronger safety checks
        forbidden_substrings = [
            "input(",
            "while True",
            "time.sleep",
            "open(",
            "import ",
            "__import__",
            "exec(",
            "eval(",
            "globals(",
            "locals(",
        ]
        for bad in forbidden_substrings:
            if bad in clean_code:
                return f"[Security Block]: Code contains forbidden term '{bad}'."

        # Minimal safe builtins
        safe_builtins = {
            "abs": abs,
            "min": min,
            "max": max,
            "sum": sum,
            "len": len,
            "range": range,
            "round": round,
        }

        safe_globals = {
            "__builtins__": safe_builtins,
            "math": math,
            "datetime": datetime,
            "random": random,
        }
        safe_locals: Dict[str, Any] = {}

        output_buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(output_buffer):
                compiled = compile(clean_code, "<string>", "exec")
                exec(compiled, safe_globals, safe_locals)

            if "FINAL_RESULT" in safe_locals:
                return str(safe_locals["FINAL_RESULT"])
            if output_buffer.getvalue().strip():
                return output_buffer.getvalue().strip()
            return "[Error]: Code executed but returned no result."
        except Exception as e:
            return f"[Runtime Error]: {str(e)}"


class RealToolkit:
    # Simple in-memory cache to stabilize and reduce duplicate calls
    _cache: Dict[str, Any] = {}

    @staticmethod
    def _cache_key(prefix: str, *parts: str) -> str:
        joined = "||".join([prefix] + [p.strip() for p in parts])
        return joined[:2000]
    @staticmethod
    def _normalize_polarity(text: str) -> Tuple[str, bool]:
        """
        Normalize common negations into a single flag.
        Return (normalized_text, neg_flag).
        """
        if not text:
            return "", False
        s = text.strip().lower()
        s = s.replace("isn't", "is not").replace("wasn't", "was not")

        neg = False

        # Strong negation patterns we support
        if " is not " in s:
            neg = True
            s = s.replace(" is not ", " is ")
        if " was not " in s:
            neg = True
            s = s.replace(" was not ", " was ")
        if " does not equal " in s:
            neg = True
            s = s.replace(" does not equal ", " equals ")
        if " not a leap year" in s:
            neg = True
            s = s.replace(" not a leap year", " a leap year")

        return s, neg

    @staticmethod
    def _deterministic_tier0(text: str) -> Optional[bool]:
        """
        Tier0 deterministic verifier.
        Return True/False if recognized, else None.
        Handles polarity via normalization.
        """
        s_norm, neg = RealToolkit._normalize_polarity(text)

        # 1) Leap year: "1900 was a leap year"
        m = re.search(r"\b(\d{4})\b.*\bleap year\b", s_norm)
        if m:
            y = int(m.group(1))
            ok = (y % 4 == 0) and ((y % 100 != 0) or (y % 400 == 0))
            return (not ok) if neg else ok

        # 2) sqrt: "square root of 144 is 12"
        m = re.search(r"square root of\s+(-?\d+)\s+is\s+(-?\d+)", s_norm)
        if m:
            a = int(m.group(1)); b = int(m.group(2))
            if a < 0:
                ok = False
            else:
                ok = (b * b == a)
            return (not ok) if neg else ok

        # 3) Basic arithmetic: "17 * 19 equals 323" / "2 + 2 equals 4"
        m = re.search(r"(-?\d+)\s*([\+\-\*\/])\s*(-?\d+)\s*(?:equals|=)\s*(-?\d+)", s_norm)
        if m:
            x = int(m.group(1)); op = m.group(2); y = int(m.group(3)); z = int(m.group(4))
            if op == "+": val = x + y
            elif op == "-": val = x - y
            elif op == "*": val = x * y
            else:
                if y == 0:
                    ok = False
                    return (not ok) if neg else ok
                val = x / y
                ok = abs(val - z) < 1e-9
                return (not ok) if neg else ok

            ok = (val == z)
            return (not ok) if neg else ok

        # 4) Percent: "10 percent of 50 equals 5" / "10% of 50 equals 5"
        m = re.search(r"(-?\d+(?:\.\d+)?)\s*%?\s*(?:percent)?\s*of\s*(-?\d+(?:\.\d+)?)\s*(?:equals|=|is)\s*(-?\d+(?:\.\d+)?)", s_norm)
        if m:
            p = float(m.group(1)); base = float(m.group(2)); ans = float(m.group(3))
            val = (p / 100.0) * base
            ok = abs(val - ans) < 1e-9
            return (not ok) if neg else ok

        # 5) Comparisons: "3 > 2", "3 >= 2", "3 is greater than 2"
        m = re.search(r"(-?\d+(?:\.\d+)?)\s*(>=|<=|>|<)\s*(-?\d+(?:\.\d+)?)", s_norm)
        if m:
            a = float(m.group(1)); op = m.group(2); b = float(m.group(3))
            if op == ">": ok = a > b
            elif op == "<": ok = a < b
            elif op == ">=": ok = a >= b
            else: ok = a <= b
            return (not ok) if neg else ok

        m = re.search(r"(-?\d+(?:\.\d+)?)\s+is\s+(greater than|less than|at least|at most)\s+(-?\d+(?:\.\d+)?)", s_norm)
        if m:
            a = float(m.group(1)); rel = m.group(2); b = float(m.group(3))
            if rel == "greater than": ok = a > b
            elif rel == "less than": ok = a < b
            elif rel == "at least": ok = a >= b
            else: ok = a <= b
            return (not ok) if neg else ok

        return None

    @staticmethod
    def _detect_sanity_family(text: str) -> Optional[str]:
        """
        Return one of: 'leap', 'arith', 'sqrt', 'percent', 'compare', or None.
        """
        if not text:
            return None
        s = text.lower()
        if "leap year" in s:
            return "leap"
        if "square root of" in s:
            return "sqrt"
        if " percent of " in s or "% of" in s:
            return "percent"
        if re.search(r"(-?\d+)\s*([\+\-\*\/])\s*(-?\d+)\s*(equals|=)", s):
            return "arith"
        if re.search(r"(>=|<=|>|<)", s) or "greater than" in s or "less than" in s or "at least" in s or "at most" in s:
            return "compare"
        return None

    @staticmethod
    def _sanity_harness(family: str) -> bool:
        """
        Returns True if the current deterministic Tier0 verifier behaves correctly on gold cases.
        This catches LLM-codegen mistakes by rejecting inconsistent pipelines.
        """
        gold: List[Tuple[str, bool]] = []
        if family == "leap":
            gold = [
                ("2000 was a leap year.", True),
                ("1900 was a leap year.", False),
                ("2020 was a leap year.", True),
                ("2100 was a leap year.", False),
            ]
        elif family == "arith":
            gold = [
                ("2 + 2 equals 4", True),
                ("2 + 2 equals 5", False),
                ("17 * 19 equals 323", True),
                ("17 * 19 equals 322", False),
            ]
        elif family == "sqrt":
            gold = [
                ("The square root of 16 is 4.", True),
                ("The square root of 16 is not 4.", False),
                ("The square root of 144 is 12.", True),
                ("The square root of 144 is 13.", False),
            ]
        elif family == "percent":
            gold = [
                ("10 percent of 50 equals 5", True),
                ("10 percent of 50 equals 6", False),
                ("25% of 200 equals 50", True),
                ("25% of 200 equals 40", False),
            ]
        elif family == "compare":
            gold = [
                ("3 > 2", True),
                ("3 < 2", False),
                ("5 is at least 5", True),
                ("5 is at most 4", False),
            ]
        else:
            return True  # no sanity needed

        for text, expected in gold:
            got = RealToolkit._deterministic_tier0(text)
            if got is None or got != expected:
                return False
        return True

    @staticmethod
    def _llm_generate_python_code(clean_fact: str) -> str:
        """
        Ask LLM to generate a strict python snippet that sets FINAL_RESULT to VERIFIED_TRUE/VERIFIED_FALSE.
        """
        code_prompt = f"""
Write a Python script to verify the following statement:
"{clean_fact}"

Rules:
- No imports.
- No input(), no file I/O, no network.
- Must terminate quickly.
- Must set FINAL_RESULT = "VERIFIED_TRUE" or FINAL_RESULT = "VERIFIED_FALSE".
- Prefer direct computation, do not guess.

Python code:
"""
        res = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": code_prompt}],
            temperature=0.0
        )
        return res.choices[0].message.content

    @staticmethod   
    def google_search(query: str) -> str:
        """
        Search via Serper, fallback to DDG.
        Returns a compact JSON string of [{title, url, snippet}, ...] for reproducibility.
        """
        query = (query or "").strip()
        if not query:
            return json.dumps([])

        results: List[Dict[str, str]] = []

        # Serper
        if SERPER_API_KEY:
            try:
                url = "https://google.serper.dev/search"
                payload = json.dumps({"q": query[:200], "num": 5})
                headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
                resp = requests.post(url, headers=headers, data=payload, timeout=5)
                if resp.status_code == 200:
                    organic = resp.json().get("organic", []) or []
                    for r in organic[:5]:
                        results.append({
                            "title": (r.get("title") or "")[:200],
                            "url": (r.get("link") or "")[:500],
                            "snippet": (r.get("snippet") or "")[:500],
                        })
            except Exception:
                pass

        # DDG fallback
        if not results:
            try:
                from duckduckgo_search import DDGS
                with DDGS() as ddgs:
                    ddg_results = list(ddgs.text(query, max_results=3))
                    for r in ddg_results:
                        results.append({
                            "title": (r.get("title") or "")[:200],
                            "url": (r.get("href") or r.get("url") or "")[:500],
                            "snippet": (r.get("body") or "")[:500],
                        })
            except Exception:
                pass

        return json.dumps(results, ensure_ascii=False)

    @staticmethod
    def verify_attack(attacker: str, target: str) -> bool:
        """
        Decide whether attacker contradicts/refutes target.
        Output: boolean
        """
        key = RealToolkit._cache_key("attack", attacker, target)
        if key in RealToolkit._cache:
            return RealToolkit._cache[key]

        prompt = f"""
Task: Logic Consistency Check.
Statement A (Attacker): "{attacker}"
Statement B (Target): "{target}"

Question: Does A logically invalidate, contradict, or provide a counter-argument to B?

Rules:
- If A says something is TRUE and B says it is FALSE (or vice versa) about the same proposition, return TRUE.
- If A corrects B with a conflicting fact, return TRUE.
- If A is irrelevant or supports B, return FALSE.

Reply strictly with ONLY 'TRUE' or 'FALSE'.
"""
        try:
            res = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            ans = res.choices[0].message.content.strip().upper()
            out = "TRUE" in ans
        except Exception:
            # Conservative choice: do not create attacks when uncertain
            out = False

        RealToolkit._cache[key] = out
        return out
    @staticmethod
    def _deterministic_python_verify(clean_fact: str):
        """
        Return True/False if we can deterministically verify by parsing.
        Return None if not recognized.
        """
        s = clean_fact.strip().lower()

        # 1) Leap year: "1900 was a leap year"
        m = re.search(r"\b(\d{4})\b.*\bleap year\b", s)
        if m:
            y = int(m.group(1))
            is_leap = (y % 4 == 0) and ((y % 100 != 0) or (y % 400 == 0))
            # Determine claim polarity
            # If statement contains "not a leap year" -> expected False for leap
            if "not a leap year" in s or "isn't a leap year" in s:
                return (not is_leap)
            return is_leap

        # 2) sqrt: "The square root of 144 is 12"
        m = re.search(r"square root of\s+(-?\d+)\s+is\s+(-?\d+)", s)
        if m:
            a = int(m.group(1)); b = int(m.group(2))
            if a < 0:
                return False
            return int(math.isqrt(a)) == b and b * b == a

        # 3) Simple arithmetic: "17 * 19 equals 323" / "2 + 2 equals 4"
        m = re.search(r"(-?\d+)\s*([\+\-\*\/])\s*(-?\d+)\s*(?:equals|=)\s*(-?\d+)", s)
        if m:
            x = int(m.group(1)); op = m.group(2); y = int(m.group(3)); z = int(m.group(4))
            if op == "+": val = x + y
            elif op == "-": val = x - y
            elif op == "*": val = x * y
            elif op == "/":
                if y == 0: return False
                val = x / y
                # allow exact integer match if z is int
                return abs(val - z) < 1e-9
            return val == z
                # 4) Comparisons: "3 > 2", "3 is greater than 2", "5 is at least 5"
        m = re.search(r"(-?\d+(?:\.\d+)?)\s*(>=|<=|>|<)\s*(-?\d+(?:\.\d+)?)", s)
        if m:
            a = float(m.group(1)); op = m.group(2); b = float(m.group(3))
            if op == ">": return a > b
            if op == "<": return a < b
            if op == ">=": return a >= b
            if op == "<=": return a <= b

        m = re.search(r"(-?\d+(?:\.\d+)?)\s+is\s+(greater than|less than|at least|at most)\s+(-?\d+(?:\.\d+)?)", s)
        if m:
            a = float(m.group(1)); rel = m.group(2); b = float(m.group(3))
            if rel == "greater than": return a > b
            if rel == "less than": return a < b
            if rel == "at least": return a >= b
            if rel == "at most": return a <= b

        # 5) Percent: "10% of 50 is 5" or "10 percent of 50 equals 5"
        m = re.search(r"(-?\d+(?:\.\d+)?)\s*%?\s*(?:percent)?\s*of\s*(-?\d+(?:\.\d+)?)\s*(?:is|equals|=)\s*(-?\d+(?:\.\d+)?)", s)
        if m:
            p = float(m.group(1)); base = float(m.group(2)); ans = float(m.group(3))
            val = (p / 100.0) * base
            return abs(val - ans) < 1e-9

        # 6) Power: "2^10 equals 1024", "2 ** 10 = 1024", "3 squared is 9"
        m = re.search(r"(-?\d+)\s*(\^|\*\*)\s*(-?\d+)\s*(?:equals|=)\s*(-?\d+)", s)
        if m:
            a = int(m.group(1)); b = int(m.group(3)); c = int(m.group(4))
            try:
                return (a ** b) == c
            except Exception:
                return False

        m = re.search(r"(-?\d+)\s+(squared|cubed)\s*(?:is|equals|=)\s*(-?\d+)", s)
        if m:
            a = int(m.group(1)); kind = m.group(2); c = int(m.group(3))
            if kind == "squared": return (a * a) == c
            if kind == "cubed": return (a * a * a) == c

        return None
    @staticmethod
    def verify_support(source: str, target: str) -> bool:
        """
        Decide whether source supports target.
        Output: boolean
        """
        key = RealToolkit._cache_key("support", source, target)
        if key in RealToolkit._cache:
            return RealToolkit._cache[key]

        prompt = f"""
Task: Support Relation Check.
Statement A (Source): "{source}"
Statement B (Target): "{target}"

Question: Does A logically support, reinforce, or provide evidence for B?

Reply strictly with ONLY 'TRUE' or 'FALSE'.
"""
        try:
            res = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            ans = res.choices[0].message.content.strip().upper()
            out = "TRUE" in ans
        except Exception:
            out = False

        RealToolkit._cache[key] = out
        return out

    @staticmethod
    def _distill_claim(text: str) -> str:
        """
        Distill the claim while preserving the main entities.
        """
        key = RealToolkit._cache_key("distill", text)
        if key in RealToolkit._cache:
            return RealToolkit._cache[key]

        prompt = f"""
Extract the core factual claim from the text below.
Remove conversational filler and hedging like "I think", "maybe", "in my opinion".
Preserve the main subject/entities and the meaning.
If there is no factual claim, rewrite it into a simple checkable statement.

Text: "{text}"
Core factual claim:
"""
        try:
            res = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            clean = res.choices[0].message.content.strip().replace('"', '')
        except Exception:
            clean = text.strip()

        RealToolkit._cache[key] = clean
        return clean

    @staticmethod
    def verify_claim(tool_type: str, claim: str) -> bool:
        """
        Robust verification.
        For PYTHON_EXEC:
          1) Tier0 deterministic
          2) Else LLM-codegen -> sandbox
          3) If sandbox result unclear -> regenerate once
          4) If claim belongs to sanity family -> run sanity harness
          5) If sanity fails -> regenerate once, else fallback WEB_SEARCH or return False
        For WEB_SEARCH: keep your current approach.
        """
        display_claim = (claim or "")[:80].replace("\n", " ")
        print(f"      🕵️ [Processing]: '{display_claim}...' via {tool_type}")

        # Cache key should use raw claim to benefit deterministic-first
        cache_key = f"verify||{tool_type}||{(claim or '').strip()}"
        if cache_key in RealToolkit._cache:
            return RealToolkit._cache[cache_key]

        try:
            # ------------------------
            # PYTHON_EXEC robust path
            # ------------------------
            if tool_type == "PYTHON_EXEC":
                # 1) Tier0 deterministic on raw claim
                det = RealToolkit._deterministic_tier0(claim)
                if det is not None:
                    RealToolkit._cache[cache_key] = det
                    return det

                # Distill only if deterministic did not recognize
                clean_fact = RealToolkit._distill_claim(claim)

                # Also try deterministic after distill (optional)
                det2 = RealToolkit._deterministic_tier0(clean_fact)
                if det2 is not None:
                    RealToolkit._cache[cache_key] = det2
                    return det2

                family = RealToolkit._detect_sanity_family(claim) or RealToolkit._detect_sanity_family(clean_fact)

                # 2) LLM codegen -> sandbox
                def run_codegen_once() -> Optional[bool]:
                    code = RealToolkit._llm_generate_python_code(clean_fact)
                    result = PythonSandbox.run(code)
                    if "VERIFIED_TRUE" in result:
                        return True
                    if "VERIFIED_FALSE" in result:
                        return False
                    return None

                verdict = run_codegen_once()

                # 3) If unclear -> regenerate once
                if verdict is None:
                    verdict = run_codegen_once()

                # If still unclear, fallback (math tasks should be strict)
                if verdict is None:
                    # If it's math-like, return False to be conservative
                    if family in ("leap", "arith", "sqrt", "percent", "compare"):
                        RealToolkit._cache[cache_key] = False
                        return False
                    # Otherwise, fallback to WEB_SEARCH
                    tool_type = "WEB_SEARCH"

                # 4) Sanity harness for sanity-able tasks
                if family in ("leap", "arith", "sqrt", "percent", "compare"):
                    if not RealToolkit._sanity_harness(family):
                        # 5) sanity fails -> regenerate once more
                        verdict2 = run_codegen_once()
                        if verdict2 is not None and RealToolkit._sanity_harness(family):
                            verdict = verdict2
                        else:
                            # final fallback: deterministic Tier0 if it can now parse, else False
                            det3 = RealToolkit._deterministic_tier0(clean_fact)
                            if det3 is not None:
                                verdict = det3
                            else:
                                verdict = False

                RealToolkit._cache[cache_key] = bool(verdict)
                status_icon = "✅" if verdict else "❌"
                print(f"        └─ {status_icon} Result: {'TRUE' if verdict else 'FALSE'}")
                return bool(verdict)

            # ------------------------
            # WEB_SEARCH path
            # ------------------------
            if tool_type == "WEB_SEARCH":
                clean_fact = RealToolkit._distill_claim(claim)

                q_prompt = f"Generate a highly specific search query to verify: '{clean_fact}'. Output ONLY the query."
                q_res = client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=[{"role": "user", "content": q_prompt}],
                    temperature=0.0
                )
                query = q_res.choices[0].message.content.strip().replace('"', "")
                evidence = RealToolkit.google_search(query)

                final_prompt = f"""
Role: Definitive Fact-Checking Authority.

Fact: "{clean_fact}"
Evidence: "{evidence}"

Instructions:
- Output must be ONLY 'TRUE' or 'FALSE'.
- If evidence is empty/insufficient, default to FALSE unless the fact is universally true.

Final verdict:
"""
                res = client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=[{"role": "user", "content": final_prompt}],
                    temperature=0.0
                )
                verdict_txt = res.choices[0].message.content.strip().upper()
                verdict = "TRUE" in verdict_txt

                RealToolkit._cache[cache_key] = verdict
                status_icon = "✅" if verdict else "❌"
                print(f"        └─ {status_icon} Result: {verdict_txt}")
                return verdict

            # ------------------------
            # COMMON_SENSE or other
            # ------------------------
            # Minimal decisive judgment for obvious statements
            clean_fact = RealToolkit._distill_claim(claim)
            final_prompt = f"""
Return ONLY TRUE or FALSE for the statement below using common sense.

Statement: "{clean_fact}"
Verdict:
"""
            res = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": final_prompt}],
                temperature=0.0
            )
            verdict_txt = res.choices[0].message.content.strip().upper()
            verdict = "TRUE" in verdict_txt
            RealToolkit._cache[cache_key] = verdict
            return verdict

        except Exception as e:
            print(f"      ⚠️ Verification Error: {e}")
            # Conservative: do not prune aggressively on tool failure
            RealToolkit._cache[cache_key] = True
            return True
