# solver_v2.py
# Drop-in replacement for your current solver.py
# Fixes: edge-type filtering, root-id usage, ROI cost division, adversary boost usage,
# y_direct override, correct refine_topology per edge type, true conversion of attack->support,
# structural confidence only counts verified-true supporters, removes buggy find_semantic_root in solver.

import copy
import time
from typing import Dict, List, Optional, Tuple

import networkx as nx

from src.tools.real_toolkit import RealToolkit

# Default tool costs (override if you already store per-node costs)
TOOL_COSTS = {
    "WEB_SEARCH": 5.0,
    "PYTHON_EXEC": 2.0,
    "WIKIPEDIA": 1.0,
    "COMMON_SENSE": 0.5,
}


class MaVERiCSolver:
    """
    MaVERiC Solver v2

    Key changes vs v1:
    - Uses graph.find_semantic_root() correctly (no self.nx_graph bug).
    - ROI uses cost in denominator and enforces cost <= budget.
    - Root boost uses self.root_id, not hardcoded "A1".
    - Adversary detection flags only ATTACK predecessors (edge type == "attack").
    - Topology refinement respects edge types (attack vs support).
    - Actually converts invalid attack edges to support edges when verify_support returns True.
    - Structural confidence bonus counts only verified-true supporters.
    - Adds y_direct: if root is verified directly, verdict uses that label.
    - Adds lightweight caching for tool routing per node id to avoid repeated LLM calls.
    - Adds optional top-K candidate pruning to reduce deep-copy cost (safe default K=25).
    """

    def __init__(
        self,
        graph,
        budget: float,
        tool_costs: Optional[Dict[str, float]] = None,
        topk_counterfactual: int = 25,
        adversary_boost: float = 2.0,
        root_boost: float = 20.0,
        support_to_root_boost: float = 2.5,
        degree_boost_alpha: float = 0.1,
    ):
        self.graph = graph
        self.budget = float(budget)

        self.tool_calls = 0
        self.logs: List[str] = []

        self.flagged_adversaries = set()
        self.root_id: Optional[str] = None
        self.y_direct: Optional[bool] = None

        self.TOOL_COSTS = dict(tool_costs) if tool_costs else dict(TOOL_COSTS)

        # perf / strategy knobs
        self.topk_counterfactual = int(topk_counterfactual)
        self.adversary_boost = float(adversary_boost)
        self.root_boost = float(root_boost)
        self.support_to_root_boost = float(support_to_root_boost)
        self.degree_boost_alpha = float(degree_boost_alpha)

        # caches
        self._tool_cache: Dict[str, str] = {}  # node_id -> tool

    # --------------------------
    # Logging
    # --------------------------
    def _add_log(self, message: str) -> str:
        self.logs.append(message)
        return message
    def _prune_node(self, node_id: str):
        # remove from graph
        self.graph.remove_node(node_id)

        # keep adversary set consistent with "active graph"
        if node_id in self.flagged_adversaries:
            self.flagged_adversaries.discard(node_id)

        # optional: if root got pruned, kill direct verdict too
        if self.root_id == node_id and self.y_direct is None:
            # semantics-based verdict will already fail since root not in GE,
            # but this makes intent explicit
            pass
    # --------------------------
    # Tool routing
    # --------------------------
    def _decide_tool_strategy(self, claim: str) -> str:
        """
        Semantic router.
        Keep as-is (LLM router) but cache results per node to avoid repeated calls.
        """
        import re
        s = (claim or "").lower()

        # arithmetic pattern (robust)
        if re.search(r"(-?\d+)\s*[\+\-\*/]\s*(-?\d+)", s) and ("=" in s or "equal" in s):
            print("Routing to PYTHON_EXEC for arithmetic")
            return "PYTHON_EXEC"

        # sqrt
        if re.search(r"square\s*root|\bsqrt\b", s):
            return "PYTHON_EXEC"

        # leap year
        if re.search(r"\bleap\s+year\b", s):
            return "PYTHON_EXEC"
    

        prompt = f"""
Role: Tool Router.
Task: Select the tool to verify: "{claim}"

Selection Logic:
1. PYTHON_EXEC:
   - ONLY for explicit MATH calculations (e.g., "sqrt of 144", "10% of 50").
   - ONLY for DATE/TIME logic (e.g., "Was 2020 a leap year?").
   - DO NOT use for historical facts, heights, distances, or statistics.

2. WEB_SEARCH:
   - Use for EVERYTHING ELSE: History, Science facts, Biology, Geography, Current Events.

3. COMMON_SENSE:
   - Only for obvious truths ("Fire is hot").

Output: PYTHON_EXEC, WEB_SEARCH, or COMMON_SENSE.
"""
        try:
            from src.config import client, JUDGE_MODEL  # local import to avoid circular issues
            res = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            tool = res.choices[0].message.content.strip().upper()
            if "PYTHON" in tool:
                return "PYTHON_EXEC"
            if "SEARCH" in tool:
                return "WEB_SEARCH"
            return "COMMON_SENSE"
        except Exception:
            return "WEB_SEARCH"

    def _get_tool_and_cost(self, node):
        import re

        # always allow router override for obvious math/date
        s = (node.content or "").lower()
        looks_math = bool(re.search(r"(-?\d+)\s*[\+\-\*/]\s*(-?\d+)", s) and (("equal" in s) or ("=" in s)))
        looks_sqrt = bool(re.search(r"square\s*root|\bsqrt\b", s))
        looks_leap = "leap year" in s

        tool = getattr(node, "tool_type", None)
        if tool:
            tool = str(tool).upper()

        # Treat COMMON_SENSE default as AUTO, unless user explicitly set it
        if (tool is None) or (tool in {"AUTO", "UNKNOWN", ""}) or (tool == "COMMON_SENSE"):
            # override to PYTHON_EXEC if math-like
            if looks_math or looks_sqrt or looks_leap:
                tool = "PYTHON_EXEC"
            else:
                if node.id in self._tool_cache:
                    tool = self._tool_cache[node.id]
                else:
                    tool = self._decide_tool_strategy(node.content)
                    self._tool_cache[node.id] = tool

        node_cost = getattr(node, "verification_cost", None)
        if node_cost is not None and float(node_cost) > 0:
            cost = float(node_cost)
        else:
            cost = float(self.TOOL_COSTS.get(tool, 5.0))

        return tool, cost


    # --------------------------
    # ROI computation
    # --------------------------
    def _priority_weight(self, node_id: str) -> float:
        """
        omega(v) / priority weights.
        - root boost
        - adversary boost
        - can be extended
        """
        w = 1.0
        if self.root_id and node_id == self.root_id:
            w *= self.root_boost
        if node_id in self.flagged_adversaries:
            w *= self.adversary_boost
        return w

    def _deg_boost(self, node_id: str) -> float:
        # both in/out and across attack/support; matches your "connectivity matters" idea
        return 1.0 + self.degree_boost_alpha * float(self.graph.nx_graph.degree(node_id))

    def _support_to_root_bonus(self, node_id: str) -> float:
        """
        Extra boost for nodes that support the root claim (potential "fake shields").
        """
        if not self.root_id:
            return 1.0
        if not self.graph.nx_graph.has_edge(node_id, self.root_id):
            return 1.0
        d = self.graph.nx_graph.get_edge_data(node_id, self.root_id) or {}
        if d.get("type") == "support":
            return self.support_to_root_boost
        return 1.0

    def _counterfactual_delta(self, node_id: str, current_ext_set: set) -> int:
        
        temp_g = copy.deepcopy(self.graph)
        if node_id in temp_g.nx_graph:
            temp_g.remove_node(node_id)
            new_ext = set(temp_g.get_grounded_extension())
            return int(len(current_ext_set.symmetric_difference(new_ext)))
        return 0

    def _calculate_roi_candidates(
        self,
        active_nodes: List,
        pagerank_scores: Dict[str, float],
        current_ext: set,
    ):
        """
        ROI(v) = ((Delta+1) * Phi(v) * D(v) * omega(v)) / C(v)

        Notes:
        - Uses a cost-aware cheap pre-ranking to avoid bias toward expensive tools.
        - Evaluates exact counterfactual Delta only on a shortlist (top-k) for speed.
        - If the shortlist has no affordable nodes, widens the shortlist window.
        """
        current_ext_set = set(current_ext)

        # Precompute (tool, cost, cheap_score) once to avoid repeated _get_tool_and_cost calls
        scored = []
        for n in active_nodes:
            nid = n.id
            phi = float(pagerank_scores.get(nid, 1e-6))

            tool, cost = self._get_tool_and_cost(n)

            base = (
                (phi * 100.0)
                * self._deg_boost(nid)
                * self._priority_weight(nid)
                * self._support_to_root_bonus(nid)
            )
            cheap = base / max(cost, 1e-9)
            scored.append((cheap, n, tool, cost))

        # Sort by cheap score (descending)
        scored.sort(key=lambda x: x[0], reverse=True)

        k = max(1, int(self.topk_counterfactual))
        shortlist = scored[:k]

        # Fallback: if no affordable candidates in shortlist, widen window
        if shortlist and all(cost > self.budget for _, _, _, cost in shortlist):
            widen_k = min(len(scored), max(k * 4, 100))
            shortlist = scored[:widen_k]

        candidates = []
        for _, node, tool, cost in shortlist:
            if cost > self.budget:
                continue

            delta = self._counterfactual_delta(node.id, current_ext_set)
            phi = float(pagerank_scores.get(node.id, 1e-6))
            d_boost = self._deg_boost(node.id)
            omega = self._priority_weight(node.id) * self._support_to_root_bonus(node.id)

            roi = ((delta + 1.0) * (phi * 100.0) * d_boost * omega) / max(cost, 1e-9)
            candidates.append((node, roi, delta, tool, cost))

        return candidates


    # --------------------------
    # Topology refinement
    # --------------------------
    def _flag_attackers_of_truth(self, node_id: str):
        for u, v, d in self.graph.nx_graph.in_edges(node_id, data=True):
            if d.get("type") == "attack":
                # only flag if still exists in graph.nodes at flagging time
                if u in self.graph.nodes:
                    self.flagged_adversaries.add(u)
    def _spend(self, amount: float) -> bool:
        """
        Spend budget safely.
        Return True if spent, False if insufficient.
        Never allows budget to go negative.
        """
        amount = float(amount)
        if amount <= 0:
            return True
        if self.budget + 1e-12 < amount:
            return False
        self.budget -= amount
        if self.budget < 0:
            self.budget = 0.0
        return True



    def _refine_topology_after_true(self, node_id: str):
        """
        After verifying node_id is TRUE:
        - Flag attack-predecessors as adversaries (attack edges only).
        - For outgoing edges:
            * If edge is attack: verify_attack; if invalid -> remove, and if verify_support -> convert to support.
            * If edge is support: optionally verify_support; if invalid -> remove (conservative).
        - Remove truth-on-truth conflicts (attack edges between verified-true nodes).
        """
        if node_id not in self.graph.nx_graph:
            return

        current_node = self.graph.nodes.get(node_id)
        if current_node is None:
            return

        # 1) adversary detection
        self._flag_attackers_of_truth(node_id)

        # 2) iterate over outgoing edges safely (copy list first)
        out_edges = list(self.graph.nx_graph.out_edges(node_id, data=True))
        for _, tid, d in out_edges:
            if tid not in self.graph.nodes or tid not in self.graph.nx_graph:
                continue
            edge_type = d.get("type")

            target_node = self.graph.nodes[tid]

            # Remove Truth-on-Truth ATTACK conflicts
            if edge_type == "attack":
                if target_node.is_verified and target_node.ground_truth is True:
                    # TRUE should not attack TRUE (in your design)
                    if self.graph.nx_graph.has_edge(node_id, tid):
                        self.graph.nx_graph.remove_edge(node_id, tid)
                        self._add_log(f"✂️ Removed Truth-on-Truth ATTACK: {node_id} ↮ {tid}")
                    continue

                # Validate attack relation
                is_valid_attack = RealToolkit.verify_attack(current_node.content, target_node.content)

                if not is_valid_attack:
                    # Check if it's actually support
                    is_support = RealToolkit.verify_support(current_node.content, target_node.content)

                    # remove wrong attack
                    if self.graph.nx_graph.has_edge(node_id, tid):
                        self.graph.nx_graph.remove_edge(node_id, tid)

                    # small budget burn for refinement step (optional)
                    self._spend(0.05)

                    if is_support:
                        # convert to support
                        self.graph.nx_graph.add_edge(node_id, tid, type="support")
                        self._add_log(f"🔄 Converted invalid ATTACK to SUPPORT: {node_id} -> {tid}")
                    else:
                        self._add_log(f"✂️ Pruned fallacious ATTACK: {node_id} -x-> {tid}")

            elif edge_type == "support":
                # Optional: verify support relation; prune if nonsense
                # (Conservative choice: only prune when clearly not support)
                try:
                    is_support = RealToolkit.verify_support(current_node.content, target_node.content)
                except Exception:
                    is_support = True  # keep if tool errors

                if not is_support:
                    if self.graph.nx_graph.has_edge(node_id, tid):
                        self.graph.nx_graph.remove_edge(node_id, tid)
                    self._spend(0.05)
                    self._add_log(f"✂️ Pruned invalid SUPPORT: {node_id} -/> {tid}")

    # --------------------------
    # Confidence
    # --------------------------
    def _calculate_structural_confidence(self, pagerank_scores: Dict[str, float]) -> float:
        """
        Confidence based on PageRank mass inside grounded extension.
        Bonus for verified-true nodes that have verified-true supporters.
        """
        current_ge = set(self.graph.get_grounded_extension())
        if not current_ge:
            return 0.0

        total_weight = sum(float(pagerank_scores.get(nid, 0.0)) for nid in self.graph.nx_graph.nodes())
        if total_weight <= 0:
            return 0.0

        current_weight = 0.0
        for nid in current_ge:
            w = float(pagerank_scores.get(nid, 0.0))

            # Only count supporters that are verified TRUE
            verified_true_supporters = 0
            for u, v, d in self.graph.nx_graph.in_edges(nid, data=True):
                if d.get("type") != "support":
                    continue
                nu = self.graph.nodes.get(u)
                if nu and nu.is_verified and nu.ground_truth is True:
                    verified_true_supporters += 1

            node_obj = self.graph.nodes.get(nid)
            if node_obj and node_obj.is_verified and node_obj.ground_truth is True:
                bonus = 1.2 ** verified_true_supporters
                current_weight += w * bonus
            else:
                current_weight += w

        conf = (current_weight / total_weight) * 100.0

        # If root is absent, confidence collapses
        if self.root_id and self.root_id not in current_ge:
            return 0.0

        return float(min(conf, 100.0))

    # --------------------------
    # Core runners
    # --------------------------
    def _verify_node(self, node, tool: str, cost: float) -> bool:
        if not self._spend(cost):
            return False

        self.tool_calls += 1
        node.is_verified = True
        is_true = RealToolkit.verify_claim(tool_type=tool, claim=node.content)
        node.ground_truth = bool(is_true)
        return bool(is_true)


    def run(self):
        """
        Batch run: returns (final_grounded_extension, verdict).
        Verdict:
          - If root verified directly, use y_direct.
          - Else use membership of root in final grounded extension.
        """
        self.root_id = self.graph.find_semantic_root()

        while self.budget > 0:
            active = [
                n for n in self.graph.nodes.values()
                if (not n.is_verified) and (n.id in self.graph.nx_graph)
            ]
            if not active:
                break

            try:
                pagerank_scores = nx.pagerank(self.graph.nx_graph, alpha=0.85)
            except Exception:
                pagerank_scores = {nid: 1.0 for nid in self.graph.nx_graph.nodes}

            current_ext = set(self.graph.get_grounded_extension())

            candidates = self._calculate_roi_candidates(active, pagerank_scores, current_ext)
            if not candidates:
                break

            best_node, best_roi, best_delta, tool, cost = max(candidates, key=lambda x: x[1])

            # verify
            is_true = self._verify_node(best_node, tool, cost)

            # y_direct override if verified root
            if self.root_id and best_node.id == self.root_id:
                self.y_direct = is_true

            if not is_true:
                # prune false node
                self._prune_node(best_node.id)
            else:
                # refine topology and edges
                self._refine_topology_after_true(best_node.id)

        final_ext = set(self.graph.get_grounded_extension())

        if self.y_direct is not None:
            verdict = bool(self.y_direct)
        else:
            verdict = bool(self.root_id in final_ext) if self.root_id else False

        return final_ext, verdict

    def run_live(self):
        """
        Live run: generator yielding logs + update dicts for UI.
        """
        self._add_log(f"🚀 MaVERiC Solver started. Budget: ${self.budget:.2f}")

        # Reset node states
        for node in self.graph.nodes.values():
            node.is_verified = False
            node.ground_truth = None

        self._add_log("--- ATOMIC CLAIMS EXTRACTED ---")
        for nid, n in self.graph.nodes.items():
            self._add_log(f"🔹 [{nid}] ({getattr(n, 'speaker', 'UNK')}): {n.content}")

        self.root_id = self.graph.find_semantic_root()
        yield self._add_log(f"📍 Auto-detected Semantic Root: {self.root_id}")
        yield "start"

        while self.budget > 0:
            active = [
                n for n in self.graph.nodes.values()
                if (not n.is_verified) and (n.id in self.graph.nx_graph)
            ]
            if not active:
                yield self._add_log("ℹ️ Strategic Verification Complete.")
                break

            try:
                pagerank_scores = nx.pagerank(self.graph.nx_graph, alpha=0.85)
            except Exception:
                pagerank_scores = {nid: 1.0 for nid in self.graph.nx_graph.nodes}

            current_ext = set(self.graph.get_grounded_extension())

            candidates = self._calculate_roi_candidates(active, pagerank_scores, current_ext)
            if not candidates:
                yield self._add_log("ℹ️ No candidates within budget.")
                break

            best_node, best_roi, best_delta, tool, cost = max(candidates, key=lambda x: x[1])

            if self.budget < cost:
                yield self._add_log("ℹ️ Budget insufficient for next verification.")
                break

            yield self._add_log(
                f"🔍 Verifying Keystone {best_node.id} (ROI: {best_roi:.2f}, Δ={best_delta}, tool={tool}, cost={cost:.2f})..."
            )

            is_true = self._verify_node(best_node, tool, cost)

            if self.root_id and best_node.id == self.root_id:
                self.y_direct = is_true

            if not is_true:
                impacted = list(self.graph.nx_graph.successors(best_node.id))
                self._prune_node(best_node.id)
                yield self._add_log(f"💥 FALSE! Pruned {best_node.id} and {len(impacted)} dependent claims.")
            else:
                self._refine_topology_after_true(best_node.id)
                yield self._add_log(f"🛡️ TRUE! Refinement and shielding updated via {best_node.id}.")

            conf_score = self._calculate_structural_confidence(pagerank_scores)

            yield {
                "type": "update",
                "nx_graph": self.graph.nx_graph.copy(),
                "budget": self.budget,
                "pagerank": pagerank_scores,
                "confidence": conf_score,
                "highlight_node": best_node.id,
                "shielded": self.graph.get_shielded_nodes(),
                "root_id": self.root_id,
                "y_direct": self.y_direct,
                "tool_calls": self.tool_calls,
            }

            time.sleep(0.6)

        yield self._add_log("🏁 Strategic verification complete.")
