import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np
from transformers import pipeline

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    full_text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        full_text += f"\n\n--- Page {page_num + 1} ---\n\n{text}"
    return full_text

# Function to chunk text
def chunk_text(text, chunk_size=500):
    chunks = []
    current_chunk = ""
    for line in text.split("\n"):
        if len(current_chunk) + len(line) < chunk_size:
            current_chunk += " " + line
        else:
            chunks.append(current_chunk.strip())
            current_chunk = line
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# Function to visualize embeddings
def visualize_embeddings(embeddings, labels):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], palette="viridis")
    for i, label in enumerate(labels):
        plt.text(reduced[i, 0], reduced[i, 1], str(i), fontsize=9)
    plt.title("Chunk Embeddings Visualized with PCA", fontsize=16, weight="bold")
    st.pyplot(plt)

# Enhanced Streamlit UI
st.set_page_config(page_title="🧠 PDF Insight Analyzer", layout="wide")
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', sans-serif;
    }
    .main {
        background: linear-gradient(135deg, #e0f7fa 0%, #f0f4c3 100%);
        padding: 2rem;
        border-radius: 12px;
    }
    h1, h2, h3 {
        color: #2E3A59;
    }
    .stButton > button {
        background: linear-gradient(90deg, #4CAF50, #81C784);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
    }
    .stTextInput > div > input {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title(" PDF Insight Analyzer")

uploaded_file = st.file_uploader("📎 Upload a PDF for Semantic Analysis", type=["pdf"])

if uploaded_file:
    st.info(" Extracting text from the uploaded PDF...")
    extracted_text = extract_text_from_pdf(uploaded_file)
    st.subheader(" Extracted PDF Text")
    st.text_area("Text Content", extracted_text, height=300)

    if st.button(" Chunk, Embed & Visualize"):
        chunks = chunk_text(extracted_text)
        st.session_state["chunks"] = chunks
        st.subheader(" Text Chunks")

        with st.spinner("Embedding text chunks using sentence-transformers..."):
            model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = model.encode(chunks, convert_to_tensor=True)
            st.session_state["embeddings"] = embeddings
            embeddings_np = embeddings.cpu().numpy()

        st.success("✅ Embeddings created. Visualizing...")
        visualize_embeddings(embeddings_np, labels=[f"Chunk {i+1}" for i in range(len(chunks))])

# Semantic search functionality if session data is available
if "chunks" in st.session_state and "embeddings" in st.session_state:
    query = st.text_input("🔍 Ask a question or search for relevant info:")
    if query:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        query_embedding = model.encode(query, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(query_embedding, st.session_state["embeddings"])[0]
        top_k = torch.topk(similarities, k=3)

        top_chunks = [st.session_state["chunks"][i] for i in top_k.indices]
        combined_text = "\n".join(top_chunks)

        st.subheader("  Answer : -- ")
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        summary = summarizer(combined_text, max_length=120, min_length=40, do_sample=False)[0]["summary_text"]
        st.markdown(f"** {summary}**")

        st.subheader(" Top Raw Chunks (for reference)")
        for idx, score in zip(top_k.indices, top_k.values):
            st.markdown(f"**🔹 Score: {score.item():.4f}**")
            st.write(st.session_state["chunks"][idx])
else:
    st.warning(" Please upload and process a PDF to enable search.")
