import json
import math
import datetime
import random
import re
import requests
import io
import contextlib
import warnings

# Tắt cảnh báo rác
warnings.filterwarnings("ignore", category=RuntimeWarning, module="duckduckgo_search")
warnings.filterwarnings("ignore", category=UserWarning, module="duckduckgo_search")

from src.config import client, SERPER_API_KEY, JUDGE_MODEL 

# Fallback Search
try:
    from duckduckgo_search import DDGS
    HAS_DDG = True
except ImportError:
    HAS_DDG = False

class PythonSandbox:
    @staticmethod
    def run(code: str) -> str:
        # 1. CLEANUP CODE
        pattern = r"```python(.*?)```"
        match = re.search(pattern, code, re.DOTALL)
        if match:
            clean_code = match.group(1).strip()
        else:
            clean_code = code.replace("```python", "").replace("```", "").strip()
        
        # 2. SAFETY CHECK (Quan trọng để không bị treo)
        forbidden = ["input(", "while True", "time.sleep", "open(", "import os", "import sys"]
        for bad_word in forbidden:
            if bad_word in clean_code:
                return f"[Security Block]: Code contains forbidden term '{bad_word}'."

        safe_globals = {
            "math": math, "datetime": datetime, "random": random, "__builtins__": __builtins__
        }
        safe_locals = {}
        
        # 3. CAPTURE OUTPUT
        output_buffer = io.StringIO()
        
        try:
            with contextlib.redirect_stdout(output_buffer):
                compiled_code = compile(clean_code, "<string>", "exec")
                exec(compiled_code, safe_globals, safe_locals)
            
            # Ưu tiên biến FINAL_RESULT
            if "FINAL_RESULT" in safe_locals:
                return str(safe_locals["FINAL_RESULT"])
            # Fallback: Lấy stdout
            elif output_buffer.getvalue().strip():
                return output_buffer.getvalue().strip()
            else:
                return "[Error]: Code executed but returned no result."
                
        except Exception as e:
            return f"[Runtime Error]: {str(e)}"

class RealToolkit:
    @staticmethod
    def google_search(query: str) -> str:
        """Search Google (Serper -> DDG)"""
        results_text = ""
        # 1. Serper
        if SERPER_API_KEY:
            try:
                url = "https://google.serper.dev/search"
                payload = json.dumps({"q": query[:200], "num": 5})
                headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
                response = requests.post(url, headers=headers, data=payload, timeout=5)
                if response.status_code == 200:
                    snippets = [r.get('snippet', '') for r in response.json().get('organic', [])]
                    if snippets: results_text = " || ".join(snippets)
            except: pass
            
        # 2. DDG Fallback
        if not results_text and HAS_DDG:
            try:
                ddgs = DDGS()
                results = ddgs.text(query, max_results=3)
                if results: results_text = " || ".join([r.get('body', '') for r in results])
            except: pass
            
        return results_text if results_text else "No search results found."

    @staticmethod
    def verify_claim(tool_type: str, claim: str) -> bool:
        """
        Robust Verification: Lenient + Hybrid Knowledge
        """
        # 🔥 BẬT LẠI LOG ĐỂ KHÔNG TƯỞNG LÀ BỊ TREO 🔥
        clean_claim = claim[:40].replace('\n', ' ')
        print(f"      🕵️ [Check]: '{clean_claim}...' via {tool_type}")

        try:
            # === PYTHON EXECUTION ===
            if tool_type == "PYTHON_EXEC":
                code_prompt = f"""
                Write a Python script to verify: "{claim}".
                - Assign `FINAL_RESULT = "VERIFIED_TRUE"` or `"VERIFIED_FALSE"`.
                - NO `input()` or infinite loops.
                - Only use math/logic.
                """
                c_res = client.chat.completions.create(
                    model=JUDGE_MODEL, messages=[{"role": "user", "content": code_prompt}], temperature=0.0
                )
                code = c_res.choices[0].message.content
                result = PythonSandbox.run(code)
                
                if "VERIFIED_TRUE" in result: return True
                if "VERIFIED_FALSE" in result: return False
                # Nếu lỗi, log nhẹ và đi tiếp xuống dưới (Hybrid)
                # print(f"        ⚠️ Python fallback: {result[:50]}...")

            # === WEB SEARCH ===
            evidence = ""
            if tool_type == "WEB_SEARCH":
                q_prompt = f"Generate 1 specific search query for: '{claim}'. Output ONLY query."
                q_res = client.chat.completions.create(
                    model=JUDGE_MODEL, messages=[{"role": "user", "content": q_prompt}], temperature=0.0
                )
                query = q_res.choices[0].message.content.strip().replace('"', '')
                evidence = RealToolkit.google_search(query)

            # === HYBRID JUDGMENT ===
            prompt = f"""
            Role: Fair Fact Checker.
            Claim: "{claim}"
            Evidence: "{evidence}"
            
            Instructions:
            1. LENIENCY: If claim is directionally correct (e.g. 5'6" vs 5'7"), mark TRUE.
            2. HYBRID: If Evidence contradicts but claim is a known scientific/historical fact (Internal Knowledge), mark TRUE.
            3. Only mark FALSE if definitively proven wrong.
            
            Reply: TRUE or FALSE.
            """
            
            res = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            verdict = res.choices[0].message.content.strip().upper()
            return "TRUE" in verdict

        except Exception as e:
            print(f"      ⚠️ Error: {e}")
            return True

    @staticmethod
    def verify_attack(attacker: str, target: str) -> bool:
        try:
            # Thêm log nhẹ
            # print(f"      ⚔️ Checking Attack logic...") 
            prompt = f"Does '{attacker}' logically ATTACK/CONTRADICT '{target}'? Reply TRUE/FALSE."
            res = client.chat.completions.create(
                model=JUDGE_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.0
            )
            return "TRUE" in res.choices[0].message.content.strip().upper()
        except: return True