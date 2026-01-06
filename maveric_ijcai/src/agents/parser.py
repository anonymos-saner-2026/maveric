import json
from openai import OpenAI
from src.config import OPENAI_API_KEY, OPENAI_BASE_URL, PARSER_MODEL, TOOLS_CONFIG
from src.core.graph import ArgumentationGraph, ArgumentNode

# Init Client with Custom Base URL
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

def parse_debate(text: str) -> ArgumentationGraph:
    prompt = f"""
    Analyze this 7-agent debate and convert it to an Argumentation Graph JSON.
    
    RULES FOR TOOL ASSIGNMENT:
    - If the argument involves math/calculation/code (often from Eve/Alice) -> Assign "PYTHON_EXEC" (Cost {TOOLS_CONFIG['PYTHON_EXEC']['cost']}).
    - If the argument cites news, dates, or events (often from Charlie/Frank) -> Assign "WEB_SEARCH" (Cost {TOOLS_CONFIG['WEB_SEARCH']['cost']}).
    - If it's an opinion or general statement (Bob/Grace/Dave) -> Assign "COMMON_SENSE" (Cost {TOOLS_CONFIG['COMMON_SENSE']['cost']}).

    JSON Structure:
    {{
        "arguments": [
            {{"id": "A1", "speaker": "Alice", "content": "Bitcoin uses SHA-256...", "tool": "PYTHON_EXEC"}}
        ],
        "attacks": [
            {{"attacker": "A3", "target": "A1"}} 
        ]
    }}

    Debate Text:
    {text}
    """
    res = client.chat.completions.create(
        model=PARSER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    data = json.loads(res.choices[0].message.content)
    af = ArgumentationGraph()
    
    for item in data.get("arguments", []):
        tool = item.get("tool", "COMMON_SENSE")
        cost = TOOLS_CONFIG.get(tool, TOOLS_CONFIG["COMMON_SENSE"])["cost"]
        node = ArgumentNode(item['id'], item['content'], item.get('speaker'), tool, cost)
        af.add_node(node)
        
    for atk in data.get("attacks", []):
        af.add_attack(atk['attacker'], atk['target'])
    return af
