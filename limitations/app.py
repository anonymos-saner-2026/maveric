import streamlit as st
import openreview
import fitz  # PyMuPDF
from openai import OpenAI
import os
import tempfile
import datetime
import requests

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="AI Research Gap Finder Ultra", page_icon="🧪", layout="wide")

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Configuration")
    api_key = st.text_input("Yescale API Key", value="sk-AOzQMlsMqmhCbXzCAOOOCkFuOGi9Yx4741EpvrsdWpceYdNM", type="password")
    base_url = st.text_input("Base URL", value="https://api.yescale.io/v1")
    model_name = st.selectbox("LLM Model", ["gpt-4o", "gpt-4o-mini", "gemini-1.5-pro"])
    max_papers = st.slider("Max Papers to Analyze", 1, 15, 5)
    st.divider()
    check_novelty = st.checkbox("Double-check Novelty (Semantic Scholar)", value=True)

client_ai = OpenAI(api_key=api_key, base_url=base_url)
client_or = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')

# --- HÀM SEARCH SEMANTIC SCHOLAR ---
def check_existing_works(query):
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit=3&fields=title,authors,year,abstract"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json().get('data', [])
    except: return []
    return []

# --- HÀM XỬ LÝ CHÍNH ---
def search_papers(topic, max_results=10):
    invitation = 'ICLR.cc/2026/Conference/-/Submission'
    results, offset = [], 0
    batch_size = 500
    search_status = st.empty()
    while len(results) < max_results and offset < 3000:
        search_status.info(f"🔍 Scanning ICLR 2026 papers {offset} to {offset + batch_size}...")
        try:
            notes = client_or.get_notes(invitation=invitation, limit=batch_size, offset=offset)
            if not notes: break
            for note in notes:
                content = note.content
                title = content.get('title', {}).get('value', 'Untitled')
                keywords = content.get('keywords', {}).get('value', [])
                if topic.lower() in title.lower() or any(topic.lower() in str(k).lower() for k in keywords):
                    results.append({'title': title, 'pdf_url': f"https://openreview.net/pdf?id={note.id}", 'id': note.id})
                if len(results) >= max_results: break
            offset += batch_size
        except: break
    search_status.empty()
    return results

def extract_limitations(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = "".join([page.get_text() for page in doc[max(0, len(doc)-4):]])
        response = client_ai.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": f"Extract ONLY the research limitations and gaps from this text: {text[:12000]}"}]
        )
        return response.choices[0].message.content
    except: return "Extraction failed."

# --- GIAO DIỆN ---
st.title("🧪 AI Research Gap Finder Ultra")
st.markdown("### Deep Search (ICLR 2026) + Expert Synthesis + Novelty Verification")

topic = st.text_input("Research Topic:", "Reinforcement Learning efficiency")

if st.button("Generate Verified Research Report"):
    if not topic: st.warning("Please enter a topic.")
    else:
        with st.status("Executing Research Pipeline...", expanded=True) as status:
            papers = search_papers(topic, max_results=max_papers)
            if not papers:
                st.error("No papers found matching the topic.")
            else:
                full_context = ""
                for i, paper in enumerate(papers):
                    st.write(f"📂 Processing paper {i+1}: {paper['title']}")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        os.system(f"curl -L -k {paper['pdf_url']} -o {tmp.name}")
                        limit_text = extract_limitations(tmp.name)
                        full_context += f"### {paper['title']}\n{limit_text}\n\n"
                        os.unlink(tmp.name)

                st.write("🧠 Synthesizing Paradigm-Shifting Proposals...")
                
                # --- PROMPT CHUYÊN GIA ---
                system_instruction = (
                    "You are a Distinguished Research Professor at a world-leading AI Lab. "
                    "You excel at synthesizing SOTA gaps into paradigm-shifting research methodologies."
                )

                user_prompt = f"""
                ### TASK:
                Analyze the provided 'Limitations' and 'Future Work' sections from ICLR 2026 submissions regarding the topic: '{topic}'. 
                Synthesize these gaps into 3 unique, high-novelty research proposals.

                ### INPUT DATA FROM ICLR 2026:
                {full_context}

                ### OUTPUT REQUIREMENTS (Response in English):

                #### 1. CRITICAL PROBLEM TAXONOMY
                - Identify 3 core technical challenges. Explain the 'Root Cause' of why current SOTA methods fail.

                #### 2. NOVEL RESEARCH PROPOSALS
                Follow this strict format for EACH proposal:
                - **METHOD NAME:** [Clear, descriptive name]
                - **CONCEPTUAL HYPOTHESIS:** What is the fundamental intuition?
                - **TECHNICAL ARCHITECTURE:** Describe the mathematical objective or structural change.
                - **NOVELTY JUSTIFICATION:** Why is this a step-change, not an incremental improvement?

                #### 3. EXECUTION BLUEPRINT
                - Suggest specific datasets and metrics. Identify 2 potential technical pitfalls.
                """

                ai_response = client_ai.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.75
                ).choices[0].message.content

                # --- BƯỚC CHECK NOVELTY ---
                novelty_report = ""
                if check_novelty:
                    st.write("🛡️ Verifying Proposals against Semantic Scholar...")
                    lines = ai_response.split('\n')
                    for line in lines:
                        if "**METHOD NAME:**" in line:
                            m_name = line.split('**METHOD NAME:**')[-1].strip()
                            existing = check_existing_works(f"{topic} {m_name}")
                            novelty_report += f"\n#### Verification for: {m_name}\n"
                            if existing:
                                novelty_report += "⚠️ *Potential overlap with existing literature found:* \n"
                                for p in existing:
                                    novelty_report += f"- **{p['title']}** ({p.get('year', 'N/A')}) [Link](https://www.semanticscholar.org/paper/{p.get('paperId')})\n"
                            else:
                                novelty_report += "✅ *High Novelty Potential: No direct matches found.*\n"

                status.update(label="Analysis Complete!", state="complete")

                # HIỂN THỊ KẾT QUẢ
                st.divider()
                st.header("🎯 Verified Expert Research Report")
                st.markdown(ai_response)
                
                if check_novelty:
                    st.divider()
                    st.header("🛡️ Novelty Verification Insights")
                    st.markdown(novelty_report)

                # DOWNLOAD REPORT
                report_md = f"# Research Gap Report: {topic.upper()}\n\n## AI Analysis\n{ai_response}\n\n## Novelty Check\n{novelty_report}\n\n## Raw Context\n{full_context}"
                st.download_button("📥 Download Full Report (.md)", report_md, f"Research_Report_{topic.replace(' ', '_')}.md")