import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# --- CONFIGURATION ---
# STRICTLY using the model name you requested
MODEL_NAME = "Qwen/Qwen3-4B"

class ORANSightRAG:
    def __init__(self, model_name=MODEL_NAME):
        print(f"🚀 Initializing ORANSight with local vLLM model: {model_name}...")
        
        # 1. Initialize vLLM
        # trust_remote_code=True is often required for Qwen architectures
        # gpu_memory_utilization=0.7 leaves 30% VRAM for the embedding model and other overhead
        try:
            self.llm = LLM(
                model=model_name, 
                trust_remote_code=True, 
                gpu_memory_utilization=0.3,
                dtype="auto" # Automatically use float16 or bfloat16 based on GPU
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        except Exception as e:
            print(f"❌ Error loading model '{model_name}': {e}")
            print("Please check if the model name is correct and you have access/internet connection.")
            raise e
        
        # [cite_start]2. Initialize Embedding Model (BAAI/bge-small-en-v1.5) [cite: 100]
        print("📥 Loading Embedding Model BAAI/bge-small-en-v1.5...")
        self.embed_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
        self.embedding_dim = 384 

        # [cite_start]3. Initialize Vector Database (FAISS) [cite: 102]
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.chunks = [] 

    def process_documents(self, text_content):
        """
        Process text: Chunking and Vectorization.
        [cite_start]Based on paper: Chunk size = 1024, Overlap = 256[cite: 119].
        """
        print("⚙️ Processing and chunking documents...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=256,
            length_function=len,
        )
        
        splits = text_splitter.split_text(text_content)
        
        if not splits:
            print("⚠️ No content to process.")
            return

        print(f"   - Embedding {len(splits)} chunks...")
        embeddings = self.embed_model.encode(splits)
        self.index.add(np.array(embeddings).astype('float32'))
        self.chunks.extend(splits)
        
        print(f"✅ Added {len(splits)} chunks to FAISS database.")

    def retrieve(self, query, k=5):
        """
        Retrieve relevant documents.
        [cite_start]Based on paper: Top 5 relevant documents[cite: 120].
        """
        query_vector = self.embed_model.encode([query])
        distances, indices = self.index.search(np.array(query_vector).astype('float32'), k)
        
        results = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.chunks):
                results.append(self.chunks[idx])
        
        return results

    def generate_response(self, query):
        """
        Generate response using vLLM with RAG and Chain-of-Thought prompts.
        """
        # --- STEP 1: RETRIEVE ---
        retrieved_docs = self.retrieve(query)
        
        # --- STEP 2: CONSTRUCT PROMPT (Chain-of-Thought) ---
        if retrieved_docs:
            # CASE 1: CONTEXT FOUND (RAG MODE)
            context_block = "\n\n".join(retrieved_docs)
            
            # Specialized System Role
            messages = [
                {"role": "system", "content": "You are an O-RAN Specification Specialist acting as a precise technical assistant."},
                {"role": "user", "content": f"""CONTEXT INFORMATION:
---------------------
{context_block}
---------------------

QUERY: {query}

INSTRUCTIONS:
You must answer the query based **STRICTLY** on the provided context information above.

**Reasoning Process (Chain-of-Thought):**
1.  **Analyze the Query:** Identify key O-RAN components (e.g., O-CU, O-DU, RIC, Interfaces).
2.  **Scan Context:** Locate definitions or descriptions in the provided text.
3.  **Synthesize:** Connect facts from the context.
4.  **Conclusion:** State the final answer clearly.

**Final Answer:**
Provide the final result below."""}
            ]
            
        else:
            # CASE 2: NO CONTEXT (FALLBACK MODE)
            # Expert System Role
            messages = [
                {"role": "system", "content": "You are a Senior O-RAN System Architect with deep knowledge of 3GPP and O-RAN Alliance standards."},
                {"role": "user", "content": f"""QUERY: {query}

SYSTEM NOTICE: No specific documents were retrieved from the knowledge base. Rely on your internal training data.

INSTRUCTIONS:
Answer using your expert knowledge of O-RAN specifications.

**Reasoning Process (Chain-of-Thought):**
1.  **Recall Standards:** Identify the relevant O-RAN Work Group (WG) or 3GPP spec.
2.  **Define Concepts:** Define the technical terms in the query.
3.  **Logical Deduction:** Explain the architecture or protocol flow.
4.  **Conclusion:** Formulate a precise answer.

**Final Answer:**
Provide the final result below."""}
            ]
            print(f"[System] ⚠️ No context found. Switching to Internal Knowledge Mode.")

        # --- STEP 3: FORMAT PROMPT FOR QWEN ---
        # Apply strict chat template for Qwen
        prompt_text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        # --- STEP 4: GENERATE WITH vLLM ---
        sampling_params = SamplingParams(
            temperature=0.1,    # Low temperature for factual accuracy
            top_p=0.95,
            max_tokens=1024,    # Allow enough tokens for Chain-of-Thought
            stop=["<|im_end|>", "<|endoftext|>"] # Common stop tokens for Qwen
        )

        outputs = self.llm.generate([prompt_text], sampling_params)
        generated_text = outputs[0].outputs[0].text
        return generated_text

# --- TEST EXECUTION ---
if __name__ == "__main__":
    # Initialize
    try:
        oransight = ORANSightRAG()
        
        # [cite_start]Test Data from Paper Appendix B [cite: 240, 246]
        sample_text = """
        The O-RAN fronthaul interface is the connection between the O-DU (Distributed Unit) and the O-RU (Remote Unit). 
        It is responsible for transporting user data and control information between the O-RU and the O-DU.
        In the context of O-RAN, virtualization refers to the process of running network functions (such as O-RU and O-DU) on software rather than dedicated hardware.
        """
        oransight.process_documents(sample_text)

        # Test Query
        query = "What is the function of the O-RAN fronthaul interface?"
        print(f"\nQUERY: {query}\n")
        
        ans = oransight.generate_response(query)
        print(f"\nFINAL ANSWER:\n{ans}")
        
    except RuntimeError as e:
        print(f"\nCRITICAL ERROR: {e}")
        print("Ensure you have a GPU available and CUDA installed for vLLM.")