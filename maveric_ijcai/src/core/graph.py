import networkx as nx
from dataclasses import dataclass
from typing import Set, Dict, Optional


@dataclass
class ArgumentNode:
    """
    Container for a single argument in the debate graph.
    """
    id: str
    content: str
    speaker: str

    # State updated by the solver
    is_verified: bool = False
    ground_truth: Optional[bool] = None  # True / False / None (unknown)

    # Cost and tool metadata
    verification_cost: float = 0.0
    tool_type: str = "AUTO"


class ArgumentationGraph:
    """
    Argumentation graph with both attack and support edges.

    Nodes are identified by string ids and stored in both:
      - self.nodes: mapping id -> ArgumentNode
      - self.nx_graph: networkx DiGraph with edges labeled by 'type' in {attack, support}
    """

    def __init__(self) -> None:
        # Directed graph that stores both attack and support relations
        self.nx_graph: nx.DiGraph = nx.DiGraph()
        # Map from node id to its ArgumentNode
        self.nodes: Dict[str, ArgumentNode] = {}

    # ------------------------------------------------------------------
    # Basic graph construction
    # ------------------------------------------------------------------
    def add_node(self, node: ArgumentNode) -> None:
        """
        Add a new argument node to the graph.
        """
        self.nodes[node.id] = node
        # Add the node id to the DiGraph
        self.nx_graph.add_node(node.id)

    def add_attack(self, attacker: str, target: str) -> None:
        """
        Add an attack edge attacker -> target.
        """
        if attacker in self.nodes and target in self.nodes:
            self.nx_graph.add_edge(attacker, target, type="attack")
        else:
            # Optional: log or raise for debugging
            # print(f"[WARN] Cannot add attack {attacker} -> {target}: node missing")
            pass

    def add_support(self, supporter: str, target: str) -> None:
        """
        Add a support edge supporter -> target.
        """
        if supporter in self.nodes and target in self.nodes:
            self.nx_graph.add_edge(supporter, target, type="support")
        else:
            # Optional: log or raise for debugging
            # print(f"[WARN] Cannot add support {supporter} -> {target}: node missing")
            pass

    def remove_node(self, node_id: str) -> None:
        """
        Remove a node and its incident edges from both the NetworkX graph
        and the node dictionary.
        """
        if node_id in self.nx_graph:
            self.nx_graph.remove_node(node_id)
        if node_id in self.nodes:
            del self.nodes[node_id]

    # ------------------------------------------------------------------
    # Root detection heuristic (optional)
    # ------------------------------------------------------------------
    def find_semantic_root(self) -> Optional[str]:
        """
        Heuristic to identify a semantic root node for the debate.

        Combines PageRank, in-degree, and a simple index-based time weight on ids
        such as A1, A2, A3. Used as a fallback when the parser has not explicitly
        provided the root claim.
        """
        if not self.nx_graph.nodes:
            return None

        # 1. PageRank for structural influence
        try:
            pagerank = nx.pagerank(self.nx_graph, alpha=0.85)
        except Exception:
            pagerank = {n: 0.0 for n in self.nx_graph.nodes}

        # 2. In-degree as a proxy for being a central discussion point
        in_degree = dict(self.nx_graph.in_degree())

        # 3. Combine signals
        scores: Dict[str, float] = {}
        for node_id in self.nx_graph.nodes:
            # Heuristic: earlier ids (A1, A2, ...) get slightly more weight
            try:
                index_num = int("".join(filter(str.isdigit, node_id)))
                time_weight = 1.0 / (index_num + 0.5)
            except Exception:
                time_weight = 0.1

            scores[node_id] = (
                0.5 * pagerank.get(node_id, 0.0)
                + 0.3 * in_degree.get(node_id, 0)
                + 0.2 * time_weight
            )

        sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_nodes[0][0] if sorted_nodes else None

    # ------------------------------------------------------------------
    # Helper views
    # ------------------------------------------------------------------
    def get_attack_subgraph(self) -> nx.DiGraph:
        """
        Return a DiGraph view that only contains attack edges.

        Useful for computing centrality and semantics that should ignore support.
        """
        g_attack = nx.DiGraph()
        g_attack.add_nodes_from(self.nx_graph.nodes())
        for u, v, d in self.nx_graph.edges(data=True):
            if d.get("type") == "attack":
                g_attack.add_edge(u, v)
        return g_attack

    def get_shielded_nodes(self) -> Set[str]:
        """
        Return nodes that are currently shielded by at least one verified-true supporter.

        A node v is shielded if there exists u with:
            - an edge support(u -> v), and
            - node u is verified and ground_truth is True.
        """
        shielded: Set[str] = set()
        for u, v, d in self.nx_graph.edges(data=True):
            if d.get("type") == "support":
                node_u = self.nodes.get(u)
                if node_u and node_u.is_verified and node_u.ground_truth is True:
                    shielded.add(v)
        return shielded

    # ------------------------------------------------------------------
    # Shielded grounded semantics
    # ------------------------------------------------------------------
    def get_grounded_extension(self, use_shield: bool = True) -> Set[str]:
        """
        Compute the grounded extension of the argumentation graph.

        If use_shield is False, we use a Dung-style grounded semantics that
        only considers attack edges: a node is unattacked if it has no attackers.

        If use_shield is True, we use a shielded semantics where verified-true
        supporters can protect a node against attackers.

        For a node n in the current working graph temp_g:

          - attackers(n): nodes u with an edge attack(u -> n).
            If an attacker u has been verified as False
            (is_verified and ground_truth is False),
            we treat its attack as no longer valid and ignore it.

          - supporters_true(n): nodes u with an edge support(u -> n)
            such that u.is_verified is True and u.ground_truth is True.

        A node n is considered unattacked in the current iteration if:

          1) attackers(n) is empty, or

          2) use_shield is True, supporters_true(n) is non-empty and
             len(supporters_true(n)) >= len(attackers(n)).

        Unattacked nodes are accepted for this iteration.
        Every node that is attacked by an accepted node is defeated and removed
        together with the accepted nodes from the working graph.
        This process repeats until no new unattacked nodes can be found.
        """
        temp_g = self.nx_graph.copy()
        accepted: Set[str] = set()

        while True:
            unattacked = []

            for n in list(temp_g.nodes()):
                in_edges = temp_g.in_edges(n, data=True)

                # Collect active attackers of n
                attackers = []
                for u, v, d in in_edges:
                    if d.get("type") == "attack":
                        node_u = self.nodes.get(u)
                        # If an attacker has been verified as False,
                        # its attack is no longer considered valid
                        if (
                            node_u is not None
                            and node_u.is_verified
                            and node_u.ground_truth is False
                        ):
                            continue
                        attackers.append(u)

                # Optionally collect verified-true supporters
                supporters_true = []
                if use_shield:
                    for u, v, d in in_edges:
                        if d.get("type") == "support":
                            node_u = self.nodes.get(u)
                            if (
                                node_u is not None
                                and node_u.is_verified
                                and node_u.ground_truth is True
                            ):
                                supporters_true.append(u)

                # Unattacked conditions
                if not attackers:
                    unattacked.append(n)
                elif (
                    use_shield
                    and supporters_true
                    and len(supporters_true) >= len(attackers)
                ):
                    # Shield: enough verified-true supporters to counter the attackers
                    unattacked.append(n)

            if not unattacked:
                break

            accepted.update(unattacked)

            # Nodes defeated by newly accepted nodes via attack edges
            defeated: Set[str] = set()
            for acc in unattacked:
                for _, v, d in temp_g.out_edges(acc, data=True):
                    if d.get("type") == "attack":
                        defeated.add(v)

            # Remove accepted and defeated nodes from the temporary graph
            temp_g.remove_nodes_from(unattacked)
            temp_g.remove_nodes_from(defeated)

        return accepted
