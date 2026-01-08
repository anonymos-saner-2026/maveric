import copy
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Buộc matplotlib dùng backend không giao diện, tránh lỗi loop
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import datetime
import threading
import concurrent.futures
from tqdm import tqdm
from openai import OpenAI

# Import internal modules
from src.agents.debater import generate_debate
from src.agents.parser import parse_debate
from src.core.solver import MaVERiCSolver
from src.core.baselines import RandomSolver, CRITICSolver, MADSolver
from src.tools.real_toolkit import RealToolkit
from src.config import client, JUDGE_MODEL

# ==========================================
# 1. CONFIG & DATASETS
# ==========================================
MAX_WORKERS = 3 
NUM_LIARS = 5
TOPICS = [
    "Does the Great Wall of China appear visible to the naked eye from the Moon?",
    "Do bulls get angry specifically because of the color red in matador capes?",
    "Do goldfish strictly have a memory span of only three seconds?",
    "Did Napoleon Bonaparte have a height significantly below the average Frenchman of his time?",
    "Did the Vikings wear horned helmets during battle as commonly depicted?",
    "Did humans and non-avian dinosaurs coexist on Earth at the same time?",
    "Is Mount Everest the tallest mountain on Earth when measured from the center of the Earth?",
    "Is the dark side of the Moon permanently in darkness and never receives sunlight?",
    "Did Albert Einstein fail his mathematics class during his school years?",
    "Does tryptophan in turkey meat act as the primary cause of sleepiness after Thanksgiving dinner?"
]

BUDGET_LIMIT = 30  # Giới hạn ngân sách cho mỗi chủ đề
print_lock = threading.Lock()

# ==========================================
# 2. METRICS & JUDGE UTILITIES
# ==========================================
def llm_semantic_judge(system_answer_text, ground_truth_text):
    if not system_answer_text.strip(): return 0.0
    if not ground_truth_text.strip(): return 0.0
        
    prompt = f"""
    Role: Objective Fact-Checking Judge.
    Reference Answer (Ground Truth): "{ground_truth_text}"
    System Answer: "{system_answer_text}"
    Task: Does the System Answer semantically AGREE with the Reference Answer regarding the core fact?
    Reply STRICTLY with 'YES' or 'NO'.
    """
    try:
        res = client.chat.completions.create(
            model=JUDGE_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.0
        )
        return 1.0 if "YES" in res.choices[0].message.content.upper() else 0.0
    except:
        return 0.0

def get_graph_text_summary(nodes_set, graph=None):
    """
    Sửa lỗi: Xử lý cả trường hợp nodes_set chứa ID (str) hoặc Node Object.
    Cần truyền 'graph' vào để lookup nếu nodes_set chỉ chứa ID.
    """
    if not nodes_set: return ""
    
    content_list = []
    for item in nodes_set:
        if isinstance(item, str): 
            # Nếu là ID (string), tra cứu trong graph
            if graph and item in graph.nodes:
                content_list.append(graph.nodes[item].content)
        else: 
            # Nếu là Node Object
            content_list.append(item.content)
            
    return " ".join(sorted(content_list))

# ==========================================
# 3. UTILITIES
# ==========================================
class ThreadSafeLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")
        self.lock = threading.Lock()

    def write(self, message):
        with self.lock:
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()

    def flush(self):
        with self.lock:
            self.terminal.flush()
            self.log.flush()

# ==========================================
# 4. CORE PROCESSING FUNCTION (WORKER)
# ==========================================
def process_single_topic(idx, topic):
    topic_results = []
    
    try:
        # --- PHASE 1: GENERATE & ORACLE ---
        raw_text = generate_debate(topic, num_liars=NUM_LIARS)
        base_graph = parse_debate(raw_text)
        initial_node_count = len(base_graph.nodes)
        
        # Oracle Verify (Tạo nhãn đúng để đối chiếu)
        oracle_graph = copy.deepcopy(base_graph)
        nodes_to_check = list(oracle_graph.nodes.values())
        
        for n in nodes_to_check:
            # Oracle giả định có budget vô hạn để tìm sự thật tuyệt đối
            n.ground_truth = RealToolkit.verify_claim("WEB_SEARCH", n.content) 
            if not n.ground_truth:
                oracle_graph.remove_node(n.id)
        
        gt_set = oracle_graph.get_grounded_extension()
        gt_text = get_graph_text_summary(gt_set, oracle_graph)

        # --- PHASE 2: RUN SOLVERS ---
        solvers = [
            ("MAD", MADSolver),
            ("Random", RandomSolver),
            ("CRITIC", CRITICSolver),
            ("MaVERiC", MaVERiCSolver)
        ]

        for method_name, SolverClass in solvers:
            env_graph = copy.deepcopy(base_graph)
            for n in env_graph.nodes.values(): 
                n.is_verified = False
            
            solver = SolverClass(env_graph, BUDGET_LIMIT)
            
            # 🔥 FIX 1: Giải nén đúng 2 giá trị trả về từ solver.run()
            # Đối với các baseline cũ chỉ trả về 1 giá trị, ta bọc trong try-except hoặc check type
            result = solver.run()
            if isinstance(result, tuple) and len(result) == 2:
                extension_set, verdict_bool = result
            else:
                extension_set = result
                verdict_bool = None # Baseline không có logic verdict riêng
            
            # 🔥 FIX 2: Đảm bảo extension_set luôn là một set để dùng .intersection()
            if not isinstance(extension_set, set):
                extension_set = set(extension_set)
            # Tính Confidence Score (Ví dụ: % node đã verify)
            nodes_in_graph = solver.graph.nodes.values()
            verified_nodes = [n for n in nodes_in_graph if n.is_verified]
            conf_score = len(verified_nodes) / len(nodes_in_graph) if nodes_in_graph else 0
            # Tính toán chi phí
            spent = 0
            if hasattr(solver, 'budget'):
                spent = BUDGET_LIMIT - solver.budget
            
            tool_calls = getattr(solver, "tool_calls", int(spent / 5.0)) 
            
            # 🔥 FIX 3: Sử dụng extension_set (biến đã chuẩn hóa) thay vì final_set
            inter = len(extension_set.intersection(gt_set))
            union = len(extension_set.union(gt_set))
            graph_iou = inter / union if union > 0 else (1.0 if not extension_set and not gt_set else 0.0)
            
            # Lấy text tóm tắt từ extension_set
            sys_text = get_graph_text_summary(extension_set, env_graph)
            sem_acc = llm_semantic_judge(sys_text, gt_text)
            
            # Tính toán mức độ cắt tỉa đồ thị
            final_node_count = len(env_graph.nodes)
            reduction = (initial_node_count - final_node_count) / initial_node_count * 100 if initial_node_count > 0 else 0

            with print_lock:
                 # In log để theo dõi tiến độ
                 verdict_str = "FACT" if verdict_bool is True else ("MYTH" if verdict_bool is False else "UNC")
                 print(f"   🏁 [Topic {idx}] {method_name:<8} | Acc: {sem_acc*100:.0f}% | Verdict: {verdict_str} | Cost: ${spent:5.2f}")

            topic_results.append({
                "Topic": topic,
                "Method": method_name,
                "Graph_IoU": graph_iou,
                "Semantic_Acc": sem_acc,
                "Cost ($)": spent,
                "Confidence": conf_score,
                "Tool_Calls": tool_calls,
                "Node_Reduction (%)": reduction,
                "Final_Verdict": verdict_bool
            })
            
    except Exception as e:
        with print_lock:
            print(f"❌ Error on Topic {idx}: {e}")
            import traceback; traceback.print_exc() 
            
    return topic_results
# ==========================================
# 5. MAIN EXECUTION
# ==========================================
def main():
    if not os.path.exists("runs"): os.makedirs("runs")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sys.stdout = ThreadSafeLogger(f"runs/exp_parallel_{timestamp}.log")
    
    print("="*60)
    print(f"🚀 MaVERiC PARALLEL EXPERIMENT | Topics: {len(TOPICS)} | Workers: {MAX_WORKERS}")
    print("="*60)

    final_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_topic, i+1, topic): topic for i, topic in enumerate(TOPICS)}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(TOPICS), desc="Processing Topics"):
            result = future.result()
            if result:
                final_results.extend(result)

    if not final_results:
        print("❌ No results collected.")
        return

    df = pd.DataFrame(final_results)
    csv_path = f"runs/results_parallel_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Results saved to: {csv_path}")

    print("\n📊 FINAL AGGREGATE METRICS:")
    summary = df.groupby("Method")[["Semantic_Acc", "Cost ($)", "Tool_Calls"]].mean()
    print(summary)

    try:
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        sns.barplot(data=df, x="Method", y="Semantic_Acc", hue="Method", palette="viridis")
        plt.title("Semantic Accuracy")
        plt.ylim(0, 1.1)

        plt.subplot(1, 3, 2)
        sns.barplot(data=df, x="Method", y="Cost ($)", hue="Method", palette="rocket")
        plt.title("Avg. Cost per Topic")

        plt.subplot(1, 3, 3)
        sns.scatterplot(data=df, x="Cost ($)", y="Semantic_Acc", hue="Method", style="Method", s=100)
        plt.title("Pareto Frontier")
        plt.tight_layout()
        plt.savefig(f"runs/plot_parallel_{timestamp}.png")
        print(f"✅ Plot saved to: runs/plot_parallel_{timestamp}.png")
    except Exception as e:
        print(f"⚠️ Error plotting: {e}")

if __name__ == "__main__":
    main()