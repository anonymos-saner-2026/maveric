import random
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
        history = f"Question: {question}\nFormat: Thought -> Action: GoogleSearch[query] -> PAUSE -> Observation"
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
                history += f"\nPAUSE\nObservation: {obs}\n"
            else:
                return content
        return history
