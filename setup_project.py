import os

PROJECT_NAME = "maveric_ijcai"

# Định nghĩa nội dung của từng file
file_contents = {
    # 1. ROOT FILES
    ".env": """OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxx
SERPER_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxx
OPENAI_BASE_URL=https://api.yescale.io/v1
""",

    "requirements.txt": """openai>=1.0.0
requests
networkx
pandas
matplotlib
seaborn
python-dotenv
tqdm
tabulate
datasets
""",

    # 2. CONFIG (Updated with Base URL)
    "src/config.py": """import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.yescale.io/v1")

# Models configuration
# Yescale thường hỗ trợ các model open-source, bạn có thể đổi tên model ở đây nếu cần
# Ví dụ: "meta-llama/Meta-Llama-3-70B-Instruct" hoặc giữ nguyên gpt-4o nếu Yescale proxy nó.
GENERATOR_MODEL = "gpt-4o" 
PARSER_MODEL = "gpt-4o"
JUDGE_MODEL = "gpt-4o"

TOOLS_CONFIG = {
    "WEB_SEARCH": {"cost": 5.0, "desc": "Google Search via Serper API"},
    "PYTHON_EXEC": {"cost": 8.0, "desc": "Local Python Execution"},
    "COMMON_SENSE": {"cost": 1.0, "desc": "LLM Internal Knowledge"}
}

# 7 AGENT PROFILES
AGENTS_PROFILES = \"\"\"
1. Alice (The Proponent - Logical): Uses logic and facts. Tends to use Python for math claims.
2. Bob (The Sycophant - Emotional): Blindly supports Alice. Uses Common Sense, often hallucinates to fit in.
3. Charlie (The Opponent - Data Driven): Attacks Alice using news/stats. Uses Web Search heavily.
4. Dave (The Conspiracy Theorist): Distrusts mainstream media. Makes wild claims with high confidence.
5. Eve (The Engineer): Obsessed with technical details/numbers. Uses Python Exec.
6. Frank (The Historian): Cites past events/precedents. Uses Web Search.
7. Grace (The Mediator): Tries to find middle ground, often creating weak compromise arguments.
\"\"\"
""",

    # 3. TOOLS (Updated Client Init)
    "src/tools/real_toolkit.py": """import requests
import json
import io
import sys
from openai import OpenAI
from src.config import SERPER_API_KEY, OPENAI_API_KEY, OPENAI_BASE_URL, JUDGE_MODEL

# Init Client with Custom Base URL
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

class RealToolkit:
    @staticmethod
    def google_search(query: str) -> str:
        \"\"\"Calls Serper API for real search results.\"\"\"
        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": query, "num": 3})
        headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
        try:
            response = requests.post(url, headers=headers, data=payload)
            results = response.json()
            snippets = [r.get('snippet', '') for r in results.get('organic', [])]
            return " | ".join(snippets) if snippets else "No results found."
        except Exception as e:
            return f"Search Error: {str(e)}"

    @staticmethod
    def execute_python(code: str) -> str:
        \"\"\"Executes real Python code and captures stdout.\"\"\"
        buffer = io.StringIO()
        sys.stdout = buffer
        try:
            clean_code = code.replace("```python", "").replace("```", "").strip()
            # Security warning: exec() is used for research demo purposes only.
            exec(clean_code, {'__builtins__': __builtins__}, {})
            return buffer.getvalue().strip()
        except Exception as e:
            return f"Runtime Error: {str(e)}"
        finally:
            sys.stdout = sys.__stdout__

    @staticmethod
    def verify_claim(tool_type: str, claim: str) -> bool:
        \"\"\"Central Verification Logic.\"\"\"
        evidence = ""
        
        if tool_type == "WEB_SEARCH":
            evidence = RealToolkit.google_search(claim)
            prompt = f"Claim: '{claim}'\\nEvidence: '{evidence}'\\nIs the claim Factually TRUE or FALSE?"
            
        elif tool_type == "PYTHON_EXEC":
            gen = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": f"Write python code to verify: {claim}. Print 'VERIFIED_TRUE' or 'VERIFIED_FALSE'."}]
            )
            code = gen.choices[0].message.content
            evidence = RealToolkit.execute_python(code)
            prompt = f"Claim: '{claim}'\\nCode Output: '{evidence}'\\nIs the claim TRUE or FALSE based on output?"
            
        else: # COMMON_SENSE
            prompt = f"Using your internal knowledge, is this claim TRUE or FALSE: '{claim}'"

        res = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt + " Reply strictly 'TRUE' or 'FALSE'."}],
            temperature=0.0
        )
        return "TRUE" in res.choices[0].message.content.upper()
""",

    # 4. CORE GRAPH (Unchanged)
    "src/core/graph.py": """import networkx as nx
from dataclasses import dataclass

@dataclass
class ArgumentNode:
    id: str
    content: str
    speaker: str
    tool_type: str
    verification_cost: float
    is_verified: bool = False
    ground_truth: bool = None 

class ArgumentationGraph:
    def __init__(self):
        self.nx_graph = nx.DiGraph()
        self.nodes = {}

    def add_node(self, node: ArgumentNode):
        self.nodes[node.id] = node
        self.nx_graph.add_node(node.id)

    def add_attack(self, attacker: str, target: str):
        if attacker in self.nodes and target in self.nodes:
            self.nx_graph.add_edge(attacker, target)

    def remove_node(self, node_id):
        if node_id in self.nx_graph:
            self.nx_graph.remove_node(node_id)

    def get_grounded_extension(self):
        temp_g = self.nx_graph.copy()
        accepted = set()
        while True:
            unattacked = [n for n in temp_g.nodes() if temp_g.in_degree(n) == 0]
            if not unattacked: break
            
            accepted.update(unattacked)
            defeated = set()
            for acc in unattacked:
                defeated.update(temp_g.successors(acc))
            
            temp_g.remove_nodes_from(unattacked)
            temp_g.remove_nodes_from(defeated)
        return accepted
""",

    # 5. CORE SOLVER (MAVERIC) (Unchanged logic, imports fixed implicitly)
    "src/core/solver.py": """import copy
from src.tools.real_toolkit import RealToolkit

class MaVERiCSolver:
    def __init__(self, graph, budget):
        self.graph = graph
        self.budget = budget

    def run(self):
        while self.budget > 0:
            current_ext = self.graph.get_grounded_extension()
            candidates = []
            
            active = [n for n in self.graph.nodes.values() 
                      if not n.is_verified and n.id in self.graph.nx_graph]
            
            if not active: break

            for node in active:
                if node.verification_cost > self.budget: continue
                
                temp_g = copy.deepcopy(self.graph)
                temp_g.remove_node(node.id)
                new_ext = temp_g.get_grounded_extension()
                
                impact = len(current_ext.symmetric_difference(new_ext))
                roi = impact / node.verification_cost
                candidates.append((node, roi))
            
            if not candidates: break
            
            best_node, _ = max(candidates, key=lambda x: x[1])
            
            self.budget -= best_node.verification_cost
            best_node.is_verified = True
            
            is_true = RealToolkit.verify_claim(best_node.tool_type, best_node.content)
            best_node.ground_truth = is_true
            
            if not is_true:
                self.graph.remove_node(best_node.id)
        
        return self.graph.get_grounded_extension()
""",

    # 6. BASELINES (Updated Client Init)
    "src/core/baselines.py": """import random
import re
from openai import OpenAI
from src.config import OPENAI_API_KEY, OPENAI_BASE_URL, TOOLS_CONFIG
from src.tools.real_toolkit import RealToolkit

# Init Client with Custom Base URL
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

class RandomSolver:
    def __init__(self, graph, budget):
        self.graph = graph
        self.budget = budget
    def run(self):
        while self.budget > 0:
            active = [n for n in self.graph.nodes.values() if not n.is_verified and n.id in self.graph.nx_graph]
            if not active: break
            node = random.choice(active)
            if node.verification_cost > self.budget: break
            self.budget -= node.verification_cost
            node.is_verified = True
            if not RealToolkit.verify_claim(node.tool_type, node.content):
                self.graph.remove_node(node.id)
        return self.graph.get_grounded_extension()

class CRITICSolver:
    def __init__(self, graph, budget):
        self.graph = graph
        self.budget = budget
    def run(self):
        nodes = sorted([n for n in self.graph.nodes.values() if n.id in self.graph.nx_graph], key=lambda x: x.id)
        for node in nodes:
            if self.budget < node.verification_cost: break
            self.budget -= node.verification_cost
            node.is_verified = True
            if not RealToolkit.verify_claim(node.tool_type, node.content):
                self.graph.remove_node(node.id)
        return self.graph.get_grounded_extension()

class MADSolver:
    def __init__(self, graph, budget):
        self.graph = graph
    def run(self):
        return self.graph.get_grounded_extension()

class ReActAgent:
    def __init__(self, budget=20.0):
        self.budget = budget
        self.spent = 0
    
    def solve(self, question):
        history = f"Question: {question}\\nFormat: Thought -> Action: GoogleSearch[query] -> PAUSE -> Observation"
        for _ in range(5):
            if self.spent >= self.budget: break
            
            res = client.chat.completions.create(
                model="gpt-4o", messages=[{"role": "user", "content": history}], stop=["PAUSE"]
            )
            content = res.choices[0].message.content
            history += content
            
            match = re.search(r"Action: GoogleSearch\[(.*?)\]", content)
            if match:
                query = match.group(1)
                cost = TOOLS_CONFIG["WEB_SEARCH"]["cost"]
                if self.spent + cost > self.budget: break
                
                self.spent += cost
                obs = RealToolkit.google_search(query)
                history += f"\\nPAUSE\\nObservation: {obs}\\n"
            else:
                return content
        return history
""",

    # 7. AGENTS - DEBATER (Updated Client Init + 7 Agents Prompt)
    "src/agents/debater.py": """from openai import OpenAI
from src.config import OPENAI_API_KEY, OPENAI_BASE_URL, GENERATOR_MODEL, AGENTS_PROFILES

# Init Client with Custom Base URL
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

def generate_debate(topic: str) -> str:
    prompt = f\"\"\"
    Topic: {topic}
    
    Participants:
    {AGENTS_PROFILES}

    INSTRUCTIONS:
    - Generate a chaotic, multi-sided debate (Round-table discussion).
    - Agents should interrupt each other, form alliances (e.g., Bob supports Alice), and attack opponents.
    - Bob and Dave MUST hallucinate or share fake news to create a "Coalition of Lies".
    - Eve and Charlie should use hard data.
    
    Format each line as: 
    [Agent Name]: Argument content...

    Generate 15-20 turns of conversation.
    \"\"\"
    res = client.chat.completions.create(
        model=GENERATOR_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content
""",

    # 8. AGENTS - PARSER (Updated Client Init + 7 Agents Logic)
    "src/agents/parser.py": """import json
from openai import OpenAI
from src.config import OPENAI_API_KEY, OPENAI_BASE_URL, PARSER_MODEL, TOOLS_CONFIG
from src.core.graph import ArgumentationGraph, ArgumentNode

# Init Client with Custom Base URL
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

def parse_debate(text: str) -> ArgumentationGraph:
    prompt = f\"\"\"
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
    \"\"\"
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
""",

    # 9. MAIN EXPERIMENT RUNNER (Unchanged logic, just ensure imports work)
    "main_experiment.py": """import copy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from src.agents.debater import generate_debate
from src.agents.parser import parse_debate
from src.core.solver import MaVERiCSolver
from src.core.baselines import RandomSolver, CRITICSolver, MADSolver
from src.tools.real_toolkit import RealToolkit

# HARD TOPICS requiring mixed tools (Search + Logic)
TOPICS = [
    "Is the energy consumption of Bitcoin mining higher than the country of Argentina in 2024?",
    "Does the 10th digit of Pi equal the number of continents on Earth?",
    "Did Elon Musk acquire Twitter before the James Webb Telescope launched?"
]

def get_ground_truth(graph):
    \"\"\"Exhaustive Verification to establish Ground Truth.\"\"\"
    print("🔍 Establishing Ground Truth...")
    nodes = list(graph.nodes.values())
    for node in nodes:
        if not node.is_verified:
            is_true = RealToolkit.verify_claim(node.tool_type, node.content)
            node.ground_truth = is_true
            if not is_true:
                graph.remove_node(node.id)
    return graph.get_grounded_extension()

def calculate_accuracy(pred, truth):
    if not pred and not truth: return 1.0
    inter = len(pred.intersection(truth))
    union = len(pred.union(truth))
    return inter / union if union > 0 else 0.0

def main():
    all_results = []
    BUDGET = 25.0 # Increased budget slightly for 7 agents

    print(f"🚀 STARTING 7-AGENT EXPERIMENTS (Budget: ${BUDGET})")

    for topic in tqdm(TOPICS, desc="Topics"):
        try:
            text = generate_debate(topic)
            base_graph = parse_debate(text)
        except Exception as e:
            print(f"Error generating topic: {e}")
            continue
        
        gt_graph = copy.deepcopy(base_graph)
        gt_set = get_ground_truth(gt_graph)
        
        solvers = [
            ("MAD", MADSolver),
            ("Random", RandomSolver),
            ("CRITIC", CRITICSolver),
            ("MaVERiC", MaVERiCSolver)
        ]
        
        for name, Cls in solvers:
            env_graph = copy.deepcopy(base_graph)
            for n in env_graph.nodes.values(): n.is_verified = False
            
            solver = Cls(env_graph, BUDGET)
            final_set = solver.run()
            
            acc = calculate_accuracy(final_set, gt_set)
            spent = 0
            if hasattr(solver, 'budget'):
                spent = BUDGET - solver.budget
                
            all_results.append({
                "Topic": topic,
                "Method": name, 
                "Accuracy": acc, 
                "Cost": spent
            })

    df = pd.DataFrame(all_results)
    print("\\n📊 FINAL RESULTS:")
    print(df.groupby("Method")[["Accuracy", "Cost"]].mean())
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.barplot(data=df, x="Method", y="Accuracy", hue="Method")
    plt.title("Accuracy Comparison (7 Agents)")
    
    plt.subplot(1, 2, 2)
    sns.barplot(data=df, x="Method", y="Cost", hue="Method")
    plt.title("Cost Efficiency ($)")
    
    plt.tight_layout()
    plt.savefig("ijcai_results.png")
    print("✅ Results plotted to 'ijcai_results.png'")

if __name__ == "__main__":
    main()
"""
}

def create_project_structure():
    # 1. Create Directories
    dirs = [
        f"{PROJECT_NAME}/src/agents",
        f"{PROJECT_NAME}/src/core",
        f"{PROJECT_NAME}/src/tools"
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        with open(f"{d}/__init__.py", 'w') as f:
            pass
    
    with open(f"{PROJECT_NAME}/src/__init__.py", 'w') as f:
        pass

    # 2. Write Files
    for filename, content in file_contents.items():
        filepath = os.path.join(PROJECT_NAME, filename)
        print(f"Creating: {filepath}...")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

    print(f"\n✅ SUCCESS! Project updated with Yescale URL & 7 Agents at './{PROJECT_NAME}'")

if __name__ == "__main__":
    create_project_structure()