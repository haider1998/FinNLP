import streamlit as st
import pdfplumber
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

# Load embedding model (runs locally)
embedder = SentenceTransformer("intfloat/e5-small-v2")

# Load LLaMA 3 model (runs locally)
llm = Llama(
    model_path="models/llama-3.gguf",
    n_ctx=4096,
    n_threads=8,          # Adjust based on your CPU
    use_mlock=True,       # Pin memory (optional)
    verbose=False
)

st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("📄 Company Annual Report Chatbot")

# Upload PDF
uploaded_file = st.file_uploader("Upload an annual report (PDF)", type="pdf")

if uploaded_file:
    with pdfplumber.open(uploaded_file) as pdf:
        full_text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

    # Chunk text
    chunk_size = 500
    chunk_overlap = 100
    chunks = []
    for i in range(0, len(full_text), chunk_size - chunk_overlap):
        chunk = full_text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)

    # Embed chunks
    st.info("🔍 Indexing PDF...")
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    st.success("✅ PDF indexed. You can now ask questions!")

    question = st.text_input("Ask a question about the report:")

    if question:
        question_embedding = embedder.encode([question])
        _, indices = index.search(question_embedding, k=3)
        context = "\n\n".join([chunks[i] for i in indices[0]])

        # Create prompt
        prompt = f"""[INST] You are a helpful assistant. Use the following context from a company's annual report to answer the user's question.

Context:
{context}

Question: {question}
Answer: [/INST]"""

        st.info("💬 Generating answer...")
        result = llm(prompt, max_tokens=768, temperature=0.7, stop=["</s>", "Question:", "\nQuestion:"])
        raw_answer = result["choices"][0]["text"]

        # Truncate cleanly at the last full stop
        last_period_index = raw_answer.rfind(".")
        if last_period_index != -1:
            answer = raw_answer[:last_period_index + 1]
        else:
            answer = raw_answer

        st.markdown("### 📘 Answer")
        st.write(answer.strip())

        # --------------------------
        # Generate Follow-up Questions
        # --------------------------
        followup_prompt = f"""[INST] You are a helpful assistant. Based on the following question and answer from a company's annual report, suggest two relevant follow-up questions the user might ask next.

Question: {question}
Answer: {answer}

Follow-up Questions:
1."""
        followup_response = llm(followup_prompt, max_tokens=100, temperature=0.7, stop=["\n\n", "</s>", "\nQuestion:"])
        followup_text = followup_response["choices"][0]["text"]

        # Clean and extract two follow-up questions
        lines = followup_text.strip().split("\n")
        followups = []

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if any(char.isalpha() for char in line):  # skip garbage lines
                question = line.strip("1234567890. ").strip()
                followups.append(question)
            if len(followups) == 2:
                break

        if followups:
            st.markdown("### 🧠 Follow-up Questions")
            for i, fq in enumerate(followups, 1):
                st.write(f"{i}. {fq}")
