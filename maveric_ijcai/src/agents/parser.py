import json
from openai import OpenAI
from src.config import OPENAI_API_KEY, OPENAI_BASE_URL, PARSER_MODEL, TOOLS_CONFIG
from src.core.graph import ArgumentationGraph, ArgumentNode

client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

def parse_debate(text: str) -> ArgumentationGraph:
    prompt = f"""
    ROLE: Expert Logic & Argumentation Analyst.
    TASK: Convert the provided debate into a High-Granularity Atomic Argumentation Graph.
    
    DECOMPOSITION RULES:
    1. ATOMICITY: Break each speaker's turn into single, verifiable factual claims. 
       (e.g., if Alice says "Wall is visible because of isotopes", create two nodes: one for visibility, one for isotopes).
    2. KEYSTONE IDENTIFICATION: Identify the underlying technical assumptions (isotopes, formulas, laws) as separate nodes.
    3. TARGET MAPPING: The core topic is "A1". All top-level claims should eventually link to "A1".
    
    RELATION LOGIC (BIPOLAR):
    - SUPPORT: Evidence, validation, or expansion of another claim.
    - ATTACK: Contradiction, debunking, or pointing out physical impossibilities.
    
    TOOL ASSIGNMENT STRATEGY:
    - "PYTHON_EXEC": Numerical claims, ratios ($10^{{12}}$), physics formulas, or logical paradoxes.
    - "WEB_SEARCH": Historical facts, scientific consensus, or institutional data.
    - "COMMON_SENSE": Qualitative opinions or general subjective statements.

    JSON OUTPUT FORMAT:
    {{
        "arguments": [
            {{"id": "A1", "speaker": "Alice", "content": "The Great Wall is visible from the Moon.", "tool": "WEB_SEARCH"}},
            {{"id": "A2", "speaker": "Alice", "content": "Stones contain high Boron-10/11 isotope ratios.", "tool": "PYTHON_EXEC"}}
        ],
        "relations": [
            {{"from": "A2", "to": "A1", "type": "support"}},
            {{"from": "A3", "to": "A2", "type": "attack"}}
        ]
    }}

    DEBATE TRANSCRIPT:
    {text}
    """
    
    res = client.chat.completions.create(
        model=PARSER_MODEL,
        messages=[{"role": "system", "content": "You output strictly valid JSON for atomic claim extraction."},
                  {"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    
    data = json.loads(res.choices[0].message.content)
    af = ArgumentationGraph()
    
    # 1. Thêm Atomic Nodes
    for item in data.get("arguments", []):
        tool = item.get("tool", "COMMON_SENSE")
        # Lấy cost từ config
        cost = TOOLS_CONFIG.get(tool, {"cost": 5.0})["cost"] 
        
        node = ArgumentNode(
            id=item['id'], 
            content=item['content'], 
            speaker=item.get('speaker'), 
            tool_type=tool, 
            verification_cost=cost
        )
        # Khởi tạo mặc định để tránh lỗi logic filter
        node.is_verified = False 
        af.add_node(node)
        
    # 2. Thêm Bipolar Relations (Attack & Support)
    for rel in data.get("relations", []):
        source, target = rel['from'], rel['to']
        if rel['type'] == "attack":
            af.add_attack(source, target)
        elif rel['type'] == "support":
            af.add_support(source, target) # MaVERiC sử dụng cạnh này để tính Shield
            
    return af