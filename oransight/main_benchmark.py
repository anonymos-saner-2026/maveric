import json
import os
import re
import time
from tqdm import tqdm  # Thư viện tạo thanh tiến trình
from oransight import ORANSightRAG

# --- CÁC HÀM HỖ TRỢ ---

def load_benchmark_file(filepath):
    """Đọc file JSON hoặc JSON Lines một cách an toàn."""
    if not os.path.exists(filepath):
        return []
        
    with open(filepath, 'r', encoding='utf-8') as file:
        try:
            return json.load(file)
        except json.JSONDecodeError:
            file.seek(0)
            data = []
            for line in file:
                if line.strip():
                    try: 
                        data.append(json.loads(line))
                    except: pass
            return data

def format_question_for_llm(item):
    question_text = item[0]
    options = item[1]
    formatted_options = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])
    
    # Prompt này chỉ định dạng câu hỏi trắc nghiệm, 
    # Logic dùng context hay không đã nằm trong class ORANSightRAG
    return f"Question: {question_text}\nOptions:\n{formatted_options}\n\nSelect the correct option number (1, 2, 3, or 4). Respond with the number ONLY."

def run_single_category(rag_system, benchmark_file):
    """
    Chạy đánh giá có thanh tiến trình (Progress Bar).
    """
    category_name = os.path.basename(benchmark_file)
    print(f"\n--- Đang xử lý: {category_name} ---")
    
    data = load_benchmark_file(benchmark_file)
    if not data:
        print(f"⚠️ File rỗng hoặc không tồn tại: {benchmark_file}")
        return 0, 0

    # LƯU Ý: Chạy toàn bộ dữ liệu (Bỏ comment dòng dưới nếu muốn test nhanh 5 câu)
    # test_data = data[:5] 
    test_data = data
    
    correct = 0
    total = 0
    
    # Sử dụng tqdm để hiện thanh loading
    # desc: Tiêu đề, unit: đơn vị tính
    pbar = tqdm(test_data, desc=f"Evaluating {category_name}", unit="qs")
    
    for item in pbar:
        try:
            prompt = format_question_for_llm(item)
            correct_idx = str(item[2])
            
            # Gọi RAG (Hàm này giờ đã thông minh hơn, tự fallback nếu thiếu context)
            response = rag_system.generate_response(prompt)
            
            # Regex bắt số (Ưu tiên bắt số đứng đầu câu hoặc đứng riêng lẻ)
            match = re.search(r'\b([1-4])\b', response)
            ai_choice = match.group(1) if match else "Unknown"
            
            if ai_choice == correct_idx:
                correct += 1
            total += 1
            
            # Cập nhật thông tin ngay trên thanh loading (Accuracy hiện tại)
            current_acc = (correct / total) * 100
            pbar.set_postfix({"Acc": f"{current_acc:.2f}%"})
            
        except KeyboardInterrupt:
            print("\n🛑 Người dùng đã dừng chương trình.")
            break
        except Exception as e:
            # Không in lỗi ra màn hình để tránh vỡ layout của thanh loading, chỉ pass hoặc log file
            pass

    final_acc = (correct/total*100) if total > 0 else 0
    print(f"\n✅ Kết quả {category_name}: {correct}/{total} ({final_acc:.2f}%)")
    
    return correct, total

def calculate_final_metrics(results):
    print("\n" + "="*40)
    print(" 📊 BẢNG TỔNG HỢP KẾT QUẢ (ORAN-BENCH-13K)")
    print("="*40)

    total_correct_all = 0
    total_questions_all = 0
    accuracies = []

    # Định nghĩa thứ tự in cho đẹp
    order = ['Easy', 'Intermediate', 'Difficult']
    
    for category in order:
        if category in results:
            correct, total = results[category]
            if total > 0:
                acc = correct / total
                accuracies.append(acc)
                total_correct_all += correct
                total_questions_all += total
                print(f"• {category:<15}: {acc*100:6.2f}% ({correct}/{total})")
            else:
                print(f"• {category:<15}: N/A (0/0)")

    print("-" * 40)

    # 1. Macro Accuracy
    if len(accuracies) > 0:
        macro_acc = sum(accuracies) / len(accuracies)
        print(f"👉 Macro Accuracy    : {macro_acc:.4f} ({(macro_acc*100):.2f}%)")
    else:
        print("👉 Macro Accuracy    : N/A")

    # 2. Weighted Accuracy
    if total_questions_all > 0:
        weighted_acc = total_correct_all / total_questions_all
        print(f"👉 Weighted Accuracy : {weighted_acc:.4f} ({(weighted_acc*100):.2f}%)")
    else:
        print("👉 Weighted Accuracy : N/A")
    print("="*40)

# --- MAIN ---

if __name__ == "__main__":
    # 1. Setup RAG
    print("🚀 Khởi động hệ thống ORANSight...")
    oransight = ORANSightRAG()
    
    # Nạp documents
    specs_folder = "./oran_specs"
    if os.path.exists(specs_folder):
        file_list = [f for f in os.listdir(specs_folder) if f.endswith(('.txt', '.pdf'))]
        if file_list:
            print(f"📂 Đang nạp {len(file_list)} tài liệu từ '{specs_folder}'...")
            # Dùng tqdm cho việc nạp file nếu số lượng file lớn
            for f in tqdm(file_list, desc="Indexing Docs"):
                path = os.path.join(specs_folder, f)
                try:
                    with open(path, "r", encoding="utf-8") as file:
                        oransight.process_documents(file.read())
                except: pass
        else:
            print("⚠️ Thư mục specs rỗng. Hệ thống sẽ chạy hoàn toàn bằng Internal Knowledge.")
    else:
        print("⚠️ Không tìm thấy thư mục 'oran_specs'. Hệ thống sẽ chạy hoàn toàn bằng Internal Knowledge.")
    
    # 2. Chạy đánh giá
    benchmark_dir = "./benchmark"
    final_results = {}
    
    # Map tên file với Category chuẩn
    tasks = [
        ("Easy", f"{benchmark_dir}/fin_E.json"),
        ("Intermediate", f"{benchmark_dir}/fin_M.json"),
        ("Difficult", f"{benchmark_dir}/fin_H.json") # Chú ý: Code cũ bạn để fin_H, check lại file thực tế
    ]

    for category, filepath in tasks:
        c, t = run_single_category(oransight, filepath)
        final_results[category] = (c, t)

    # 3. Tính toán Metric
    calculate_final_metrics(final_results)