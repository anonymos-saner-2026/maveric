import copy
from src.tools.real_toolkit import RealToolkit
from src.config import client, SERPER_API_KEY, JUDGE_MODEL

# Bảng giá thực tế (để trừ budget cho chuẩn)
TOOL_COSTS = {
    "WEB_SEARCH": 5.0,
    "PYTHON_EXEC": 2.0,
    "WIKIPEDIA": 1.0,
    "COMMON_SENSE": 0.5
}

class MaVERiCSolver:
    def __init__(self, graph, budget):
        self.graph = graph
        self.budget = budget
        self.tool_calls = 0

    def _decide_tool_strategy(self, claim: str) -> str:
        """
        🧠 Semantic Router: Chỉ dùng Python cho Toán/Logic. Còn lại đẩy sang Search.
        """
        prompt = f"""
        Role: Tool Router.
        Task: Select the tool to verify: "{claim}"
        
        Selection Logic:
        1. PYTHON_EXEC:
           - ONLY for explicit MATH calculations (e.g., "sqrt of 144", "10% of 50").
           - ONLY for DATE/TIME logic (e.g., "Was 2020 a leap year?").
           - DO NOT use for historical facts, heights, distances, or statistics (Python doesn't know these).
           
        2. WEB_SEARCH:
           - Use for EVERYTHING ELSE: History (Napoleon), Science (Great Wall), Biology, Geography, Current Events.
           
        3. COMMON_SENSE:
           - Only for obvious truths ("Fire is hot").
        
        Output: PYTHON_EXEC, WEB_SEARCH, or COMMON_SENSE.
        """
        try:
            res = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            tool = res.choices[0].message.content.strip().upper()
            
            if "PYTHON" in tool: return "PYTHON_EXEC"
            if "SEARCH" in tool: return "WEB_SEARCH"
            return "COMMON_SENSE"
        except:
            return "WEB_SEARCH"

    def run(self):
        print(f"🚀 [MaVERiC] Solver started with Budget=${self.budget}")
        
        while self.budget > 0:
            current_ext = self.graph.get_grounded_extension()
            candidates = []
            
            active = [n for n in self.graph.nodes.values() 
                      if not n.is_verified and n.id in self.graph.nx_graph]
            
            if not active: break

            # --- 🔥 NEW: TÍNH PAGERANK ĐỂ ĐO ĐỘ ẢNH HƯỞNG CẤU TRÚC 🔥 ---
            # Chuyển đổi sang graph vô hướng hoặc có hướng tùy chiến thuật
            # Ở đây dùng đồ thị gốc (có hướng) để tìm node bị tấn công/tấn công nhiều
            try:
                # PageRank đánh giá tầm quan trọng dựa trên liên kết
                pagerank_scores = nx.pagerank(self.graph.nx_graph, alpha=0.85)
            except:
                # Fallback nếu graph quá nhỏ hoặc lỗi
                pagerank_scores = {n.id: 1.0 for n in active}

            # 1. TÍNH ROI MỚI
            for node in active:
                if self.budget < 0.5: continue
                
                # A. Counterfactual Impact (Logic cũ)
                temp_g = copy.deepcopy(self.graph)
                if node.id in temp_g.nx_graph:
                    temp_g.remove_node(node.id)
                    new_ext = temp_g.get_grounded_extension()
                    raw_impact = len(current_ext.symmetric_difference(new_ext))
                else:
                    raw_impact = 0

                # B. Structural Influence (Logic mới)
                # Lấy điểm PageRank của node (mặc định 0 nếu lỗi)
                influence_score = pagerank_scores.get(node.id, 0.001)
                
                # C. Attack Centrality (Logic phụ: Ưu tiên node đang tấn công/bị tấn công)
                degree_boost = 1.0
                if node.id in self.graph.nx_graph:
                    deg = self.graph.nx_graph.degree(node.id)
                    degree_boost = 1.0 + (0.1 * deg) # Tăng nhẹ 10% mỗi cạnh

                # 🔥 CÔNG THỨC TỔNG HỢP (HYBRID ROI) 🔥
                # Kết hợp Impact thực tế + Tầm quan trọng cấu trúc
                # Hệ số 100 để scale PageRank lên cho cân xứng với raw_impact
                # influence_score thường rất nhỏ (0.0x), raw_impact là số nguyên (1, 2, 3...)
                
                combined_score = (raw_impact + 1.0) * (influence_score * 100) * degree_boost
                
                # Chia cho cost (cộng 1e-5 để tránh chia 0)
                # Dùng estimate cost ban đầu để ranking
                est_cost = 2.0 # Giả định trung bình
                if "height" in node.content or "date" in node.content: est_cost = 5.0 # Heuristic nhẹ
                
                roi = combined_score / est_cost
                
                candidates.append((node, roi, raw_impact)) # Lưu thêm raw để debug

            if not candidates: break
            
            # Chọn node tốt nhất
            best_node, best_roi, raw_imp = max(candidates, key=lambda x: x[1])
            
            print(f"   🎯 Target: '{best_node.content[:25]}...'")
            print(f"      📊 Stats: PageRank={pagerank_scores.get(best_node.id,0):.3f} | Imp={raw_imp} | ROI={best_roi:.2f}")
            
            # 2. 🧠 ROUTING: SOLVER QUYẾT ĐỊNH DÙNG TOOL GÌ
            selected_tool = self._decide_tool_strategy(best_node.content)
            actual_cost = TOOL_COSTS.get(selected_tool, 5.0)
            
            # Check lại budget lần cuối với giá thực tế
            if self.budget < actual_cost:
                print("   ⚠️ Not enough budget for selected tool. Stopping.")
                break

            print(f"   🎯 Target: '{best_node.content[:30]}...' | ROI: {best_roi:.2f} | Tool: {selected_tool}")
            
            # Trừ tiền
            self.budget -= actual_cost
            best_node.is_verified = True
            
            # 3. 🔨 EXECUTION: GỌI TOOLKIT ĐỂ VERIFY
            # Truyền selected_tool vào đây thay vì best_node.tool_type cũ
            is_true = RealToolkit.verify_claim(tool_type=selected_tool, claim=best_node.content)
            self.tool_calls += 1
            best_node.ground_truth = is_true
            
            if not is_true:
                # CASE A: Node sai -> Xóa Node khỏi graph
                print(f"      ❌ FALSE CLAIM -> Removing Node {best_node.id}")
                self.graph.remove_node(best_node.id)
            else:
                # CASE B: Node đúng -> Giữ Node -> Check Topology (Self-Healing)
                print(f"      ✅ TRUE CLAIM -> Checking outgoing attacks...")
                
                if best_node.id in self.graph.nx_graph:
                    targets = list(self.graph.nx_graph.successors(best_node.id))
                    
                    for target_id in targets:
                        # Phí check cạnh ($0.5)
                        if self.budget < 0.5: break 
                        
                        target_node = self.graph.nodes[target_id]
                        
                        # Kiểm tra xem best_node có thực sự attack target_node không
                        is_valid_attack = RealToolkit.verify_attack(best_node.content, target_node.content)
                        
                        if not is_valid_attack:
                            print(f"      ✂️ [TOPO-FIX]: Cutting invalid edge {best_node.id} -> {target_id}")
                            self.graph.nx_graph.remove_edge(best_node.id, target_id)
                            self.budget -= 0.5 
        
        return self.graph.get_grounded_extension()