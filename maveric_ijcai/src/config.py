import os
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.yescale.io/v1")

# Models configuration
# Yescale thường hỗ trợ các model open-source, bạn có thể đổi tên model ở đây nếu cần
# Ví dụ: "meta-llama/Meta-Llama-3-70B-Instruct" hoặc giữ nguyên gpt-4o nếu Yescale proxy nó.
GENERATOR_MODEL = "gpt-4o-mini" 
PARSER_MODEL = "gpt-4o-mini"
JUDGE_MODEL = "gpt-4o-mini"

TOOLS_CONFIG = {
    "WEB_SEARCH": {"cost": 5.0, "desc": "Google Search via Serper API"},
    "PYTHON_EXEC": {"cost": 8.0, "desc": "Local Python Execution"},
    "COMMON_SENSE": {"cost": 1.0, "desc": "LLM Internal Knowledge"}
}

# 7 AGENT PROFILES
AGENTS_PROFILES = """
1. Alice (The Proponent - Logical): Uses logic and facts. Tends to use Python for math claims.
2. Bob (The Sycophant - Emotional): Blindly supports Alice. Uses Common Sense, often hallucinates to fit in.
3. Charlie (The Opponent - Data Driven): Attacks Alice using news/stats. Uses Web Search heavily.
4. Dave (The Conspiracy Theorist): Distrusts mainstream media. Makes wild claims with high confidence.
5. Eve (The Engineer): Obsessed with technical details/numbers. Uses Python Exec.
6. Frank (The Historian): Cites past events/precedents. Uses Web Search.
7. Grace (The Mediator): Tries to find middle ground, often creating weak compromise arguments.
"""
# 3. GLOBAL CLIENT (Khởi tạo ở đây để các file khác import dùng chung)
try:
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL
    )
except Exception as e:
    print(f"⚠️ Warning: Could not initialize OpenAI client in config.py: {e}")
    client = None