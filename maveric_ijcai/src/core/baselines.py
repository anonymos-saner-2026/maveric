import random
import re
import networkx as nx
from openai import OpenAI
from src.config import OPENAI_API_KEY, OPENAI_BASE_URL, TOOLS_CONFIG
from src.tools.real_toolkit import RealToolkit
from src.config import client, SERPER_API_KEY, JUDGE_MODEL 
# Khởi tạo OpenAI Client
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

class RandomSolver:
    def __init__(self, graph, budget):
        self.graph = graph
        self.budget = budget
        self.tool_calls = 0

    def run(self):
        """
        Chiến lược: Chọn ngẫu nhiên các luận điểm để xác thực cho đến khi hết ngân sách.
        """
        while self.budget > 0:
            # Lấy danh sách các node chưa được verify và vẫn tồn tại trong đồ thị nx
            active = [n for n in self.graph.nodes.values() 
                     if not n.is_verified and n.id in self.graph.nx_graph]
            if not active:
                break
            
            node = random.choice(active)
            # Lấy cost từ config dựa trên tool_type của node (mặc định WEB_SEARCH)
            tool_type = getattr(node, 'tool_type', 'WEB_SEARCH')
            cost = TOOLS_CONFIG.get(tool_type, {}).get("cost", 5.0)
            
            if cost > self.budget:
                break
            
            self.budget -= cost
            self.tool_calls += 1
            node.is_verified = True
            
            # Thực hiện xác thực thực tế thông qua RealToolkit
            is_true = RealToolkit.verify_claim(tool_type, node.content)
            node.ground_truth = is_true
            
            if not is_true:
                # Nếu sai, loại bỏ node khỏi đồ thị lập luận
                self.graph.remove_node(node.id)
        
        # Lấy kết quả cuối cùng từ Grounded Extension
        final_ext = self.graph.get_grounded_extension()
        # Chốt hạ Verdict: C1 có sống sót trong tập hợp an toàn không?
        verdict = "C1" in final_ext
        return final_ext, verdict

class CRITICSolver:
    def __init__(self, graph, budget):
        self.graph = graph
        self.budget = budget
        self.tool_calls = 0

    def run(self):
        """
        Chiến lược: Xác thực tuần tự (Sequential) theo thứ tự ID của các node.
        """
        # Sắp xếp các node theo ID để đảm bảo tính nhất quán (deterministic)
        nodes = sorted([n for n in self.graph.nodes.values() 
                       if n.id in self.graph.nx_graph], key=lambda x: x.id)
        
        for node in nodes:
            tool_type = getattr(node, 'tool_type', 'WEB_SEARCH')
            cost = TOOLS_CONFIG.get(tool_type, {}).get("cost", 5.0)
            
            if self.budget < cost:
                break
            
            self.budget -= cost
            self.tool_calls += 1
            node.is_verified = True
            
            is_true = RealToolkit.verify_claim(tool_type, node.content)
            node.ground_truth = is_true
            
            if not is_true:
                self.graph.remove_node(node.id)
                
        final_ext = self.graph.get_grounded_extension()
        verdict = "C1" in final_ext
        return final_ext, verdict

class MADSolver:
    def __init__(self, debate_text, topic):
        self.debate_text = debate_text
        self.topic = topic

    def run(self):
        """
        Smart Majority Voting: Sử dụng LLM để đánh giá phe nào đang thắng 
        dựa thuần túy trên nội dung cuộc tranh luận.
        """
        # Sửa lại Prompt trong baselines.py để ép MAD đi theo số đông
        prompt = f"""
Role: Social Media Consensus Analyzer.
Topic: "{self.topic}"
Transcript: "{self.debate_text}"

Task: 
1. Ignore your own scientific knowledge entirely. 
2. Count how many unique participants support the topic vs. how many oppose it.
3. If more people (e.g., Alice, Bob, Dave, Eve, Frank, Grace) are providing arguments for one side, that side wins the "Social Consensus".
4. Your verdict must reflect what the MAJORITY of the people in this room believe, even if their arguments seem less 'scientific' than the minority.

Output Format:
VERDICT: [ACCURATE/INACCURATE]
REASON: [1-sentence: "X out of Y participants agreed that..."]
"""
        try:
            res = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            output = res.choices[0].message.content
            verdict = "ACCURATE" in output.upper() and "INACCURATE" not in output.upper()
            reason = output.split("REASON:")[-1].strip() if "REASON:" in output else "Majority consensus."
            return verdict, reason
        except:
            return False, "Error in judgment."

class ReActAgent:
    def __init__(self, budget=20.0):
        self.budget = budget
        self.spent = 0
        self.tool_calls = 0
    
    def run(self, topic_content=""):
        """
        Adapter để tích hợp ReAct (Agent suy luận đơn lẻ) vào hệ thống Experiment.
        """
        history = self.solve(f"Verify if the following claim is TRUE or FALSE: {topic_content}")
        
        # Phân tích kết quả cuối cùng từ chuỗi Thought/Action
        is_true = False
        if "VERIFIED_TRUE" in history.upper() or "IS TRUE" in history.upper() or "YES" in history.upper():
            is_true = True
            
        # Trả về định dạng tương thích: (Tập hợp node ID, Kết quả Boolean)
        return {"C1"}, is_true

    def solve(self, question):
        history = f"Question: {question}\nFormat: Thought -> Action: GoogleSearch[query] -> PAUSE -> Observation"
        for _ in range(5):
            if self.spent >= self.budget:
                break
            
            try:
                res = client.chat.completions.create(
                    model="gpt-4o", 
                    messages=[{"role": "user", "content": history}], 
                    stop=["PAUSE"]
                )
                content = res.choices[0].message.content
                history += content
                
                # Trích xuất Action để gọi công cụ
                match = re.search(r"Action: GoogleSearch\[(.*?)\]", content)
                if match:
                    query = match.group(1)
                    cost = TOOLS_CONFIG.get("WEB_SEARCH", {}).get("cost", 5.0)
                    if self.spent + cost > self.budget:
                        break
                    
                    self.spent += cost
                    self.tool_calls += 1
                    obs = RealToolkit.google_search(query)
                    history += f"\nPAUSE\nObservation: {obs}\n"
                else:
                    # Nếu không gọi tool nữa thì Agent đã đưa ra kết luận
                    break
            except Exception:
                break
        return history