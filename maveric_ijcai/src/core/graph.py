import networkx as nx
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
