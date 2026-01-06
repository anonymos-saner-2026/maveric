import copy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import datetime
from tqdm import tqdm

# Import modules
from src.agents.debater import generate_debate
from src.agents.parser import parse_debate
from src.core.solver import MaVERiCSolver
from src.core.baselines import RandomSolver, CRITICSolver, MADSolver
from src.tools.real_toolkit import RealToolkit

# ==========================================
# 1. DANH SÁCH 10 TOPICS (FACT VS MYTH)
# ==========================================
TOPICS = [
    # 1. Science Myth
    "Does the Great Wall of China appear visible to the naked eye from the Moon?",
    
    # 2. Historical Fact vs Fiction
    "Did Napoleon Bonaparte have a height significantly below the average Frenchman of his time?",
    
    # 3. Biology/Common Sense
    "Do bulls get angry specifically because of the color red in matador capes?",
    
    # 4. History/Technology
    "Did the Vikings wear horned helmets during battle as commonly depicted?",
    
    # 5. Scientific Timeline (Hard)
    "Did humans and non-avian dinosaurs coexist on Earth at the same time?",
    
    # 6. Biography/Myth
    "Did Albert Einstein fail his mathematics class during his school years?",
    
    # 7. Geography/Technicality
    "Is Mount Everest the tallest mountain on Earth when measured from the center of the Earth?",
    
    # 8. Biology/Memory
    "Do goldfish strictly have a memory span of only three seconds?",
    
    # 9. Health/Chemistry
    "Does tryptophan in turkey meat act as the primary cause of sleepiness after Thanksgiving dinner?",
    
    # 10. Physics/Space
    "Is the dark side of the Moon permanently in darkness and never receives sunlight?"
]

# ==========================================
# 2. LOGGING UTILITY (Ghi cả ra màn hình và file)
# ==========================================
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Ghi ngay lập tức

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# ==========================================
# 3. EXPERIMENT FUNCTIONS
# ==========================================
def get_ground_truth(graph):
    """Exhaustive Verification để lấy đáp án đúng tuyệt đối (Oracle)."""
    print("\n🔍 Establishing Ground Truth (Oracle Mode)...")
    nodes = list(graph.nodes.values())
    for node in nodes:
        if not node.is_verified:
            # Verify thật để làm Ground Truth
            # Lưu ý: Ground Truth không tính vào Cost của thuật toán
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
    # --- SETUP LOGGING ---
    if not os.path.exists("runs"):
        os.makedirs("runs")
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"runs/experiment_{timestamp}.log"
    
    # Redirect print to both console and file
    sys.stdout = Logger(log_filename)
    
    print(f"📄 LOG FILE CREATED: {log_filename}")
    print("="*60)
    print(f"🚀 STARTING BATCH EXPERIMENT (N={len(TOPICS)} Topics)")
    print("="*60)

    all_results = []
    BUDGET = 30.0 # Tăng nhẹ budget vì topic đa dạng

    for i, topic in enumerate(tqdm(TOPICS, desc="Progress")):
        print("\n" + "#"*60)
        print(f"📌 TOPIC {i+1}: {topic}")
        print("#"*60)

        try:
            # 1. Generate & Parse
            text = generate_debate(topic)
            
            print("\n--- 🗣️ DEBATE CONTENT ---")
            print(text)
            print("-------------------------\n")

            base_graph = parse_debate(text)
            
            # 2. Ground Truth
            gt_graph = copy.deepcopy(base_graph)
            gt_set = get_ground_truth(gt_graph)
            
            # 3. Run Solvers
            solvers = [
                ("MAD", MADSolver),
                ("Random", RandomSolver),
                ("CRITIC", CRITICSolver),
                ("MaVERiC", MaVERiCSolver)
            ]
            
            for name, Cls in solvers:
                print(f"\n👉 Running Method: {name}")
                env_graph = copy.deepcopy(base_graph)
                # Reset verify status
                for n in env_graph.nodes.values(): 
                    n.is_verified = False
                
                solver = Cls(env_graph, BUDGET)
                final_set = solver.run()
                
                acc = calculate_accuracy(final_set, gt_set)
                
                spent = 0
                if hasattr(solver, 'budget'):
                    spent = BUDGET - solver.budget
                
                print(f"   🏁 Result: Accuracy={acc:.2f} | Cost=${spent:.2f}")

                all_results.append({
                    "Topic ID": i+1,
                    "Topic": topic,
                    "Method": name, 
                    "Accuracy": acc, 
                    "Cost": spent
                })

        except Exception as e:
            print(f"❌ ERROR processing topic {i+1}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 4. Save & Plot
    df = pd.DataFrame(all_results)
    
    # Lưu CSV kết quả
    csv_filename = f"runs/results_{timestamp}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\n💾 Data saved to: {csv_filename}")

    print("\n📊 AGGREGATE RESULTS:")
    summary = df.groupby("Method")[["Accuracy", "Cost"]].mean()
    print(summary)
    
    # Vẽ biểu đồ
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.barplot(data=df, x="Method", y="Accuracy", hue="Method")
    plt.title("Average Accuracy (10 Topics)")
    plt.ylim(0, 1.1)
    
    plt.subplot(1, 2, 2)
    sns.barplot(data=df, x="Method", y="Cost", hue="Method")
    plt.title("Average Cost ($)")
    
    plt.tight_layout()
    plt.savefig(f"runs/plot_{timestamp}.png")
    print(f"✅ Plot saved to: runs/plot_{timestamp}.png")

if __name__ == "__main__":
    main()