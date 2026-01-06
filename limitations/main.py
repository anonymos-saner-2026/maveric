import streamlit as st
import openreview
import fitz  # PyMuPDF
from openai import OpenAI
import os
import tempfile

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="AI Research Gap Finder", page_icon="🔍", layout="wide")

# --- SIDEBAR: Cấu hình API ---
with st.sidebar:
    st.title("Settings")
    api_key = st.text_input("Yescale API Key", value="sk-AOzQMlsMqmhCbXzCAOOOCkFuOGi9Yx4741EpvrsdWpceYdNM", type="password")
    base_url = st.text_input("Base URL", value="https://api.yescale.io/v1")
    model_name = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gemini-1.5-pro"])
    max_papers = st.slider("Số lượng bài báo tối đa", 1, 10, 3)

# Khởi tạo clients
client_ai = OpenAI(api_key=api_key, base_url=base_url)
client_or = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')

# --- HÀM XỬ LÝ ---
def search_papers(topic, limit_search=300):
    invitation = 'ICLR.cc/2026/Conference/-/Submission'
    try:
        notes = client_or.get_notes(invitation=invitation, limit=limit_search) 
        results = []
        for note in notes:
            content = note.content
            title = content.get('title', {}).get('value', '')
            keywords = content.get('keywords', {}).get('value', [])
            keywords_str = " ".join([str(k) for k in keywords])
            
            if topic.lower() in title.lower() or topic.lower() in keywords_str.lower():
                pdf_url = f"https://openreview.net/pdf?id={note.id}"
                results.append({'title': title, 'pdf_url': pdf_url, 'id': note.id})
                if len(results) >= max_papers: break
        return results
    except Exception as e:
        st.error(f"Lỗi OpenReview: {e}")
        return []

def extract_limitations(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        # Lấy 4 trang cuối
        text = "".join([page.get_text() for page in doc[max(0, len(doc)-4):]])
        
        response = client_ai.chat.completions.create(
            model=model_name, 
            messages=[
                {"role": "system", "content": "You are a research assistant specialized in academic analysis."},
                {"role": "user", "content": f"Extract the 'Limitations' or 'Future Work' section from this text. If not explicitly found, summarize the main weaknesses mentioned by the authors: {text[:15000]}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Lỗi trích xuất: {e}"

# --- GIAO DIỆN CHÍNH ---
st.title("🚀 AI Research Gap Finder")
st.markdown("Hệ thống tự động tìm kiếm 'Limitations' từ các bài báo **ICLR 2026** mới nhất để gợi ý hướng nghiên cứu.")

topic = st.text_input("Nhập chủ đề bạn quan tâm (ví dụ: Knowledge Distillation, LLM Reasoning...)", "")

if st.button("Bắt đầu phân tích"):
    if not topic:
        st.warning("Vui lòng nhập chủ đề!")
    else:
        with st.status("Đang thực hiện quy trình...", expanded=True) as status:
            # Bước 1: Tìm kiếm
            st.write("🔍 Đang quét OpenReview cho ICLR 2026...")
            papers = search_papers(topic)
            
            if not papers:
                st.error("Không tìm thấy bài báo nào phù hợp.")
                status.update(label="Thất bại", state="error")
            else:
                st.write(f"✅ Tìm thấy {len(papers)} bài báo phù hợp.")
                
                full_context = ""
                # Bước 2: Xử lý từng bài
                for i, paper in enumerate(papers):
                    st.write(f"📄 Đang xử lý bài {i+1}: {paper['title']}")
                    
                    # Tải file tạm
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        os.system(f"curl -L -k {paper['pdf_url']} -o {tmp_file.name}")
                        limit_text = extract_limitations(tmp_file.name)
                        full_context += f"### {paper['title']}\n{limit_text}\n\n"
                        os.unlink(tmp_file.name) # Xóa file tạm
                
                # Bước 3: Tổng hợp
                st.write("🧠 Đang dùng AI để phân tích tổng hợp các lỗ hổng...")
                final_response = client_ai.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "user", "content": f"Dựa trên các phần limitations sau đây về chủ đề '{topic}', hãy thực hiện: 1. Chỉ ra 3 vấn đề lớn nhất chưa được giải quyết. 2. Gợi ý 3 hướng nghiên cứu mới cụ thể cho luận văn/bài báo. Trình bày bằng tiếng Việt. \n\n {full_context}"}
                    ]
                )
                
                status.update(label="Hoàn thành phân tích!", state="complete")

                # HIỂN THỊ KẾT QUẢ
                st.divider()
                st.header("🎯 Kết quả phân tích hướng nghiên cứu")
                st.markdown(final_response.choices[0].message.content)
                
                with st.expander("Xem chi tiết dữ liệu thô đã trích xuất"):
                    st.markdown(full_context)

# --- FOOTER ---
st.caption("Dữ liệu được lấy trực tiếp từ OpenReview API v2 (ICLR 2026).")