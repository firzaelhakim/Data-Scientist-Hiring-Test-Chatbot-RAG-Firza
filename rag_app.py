import streamlit as st, os, numpy as np, re
from sentence_transformers import SentenceTransformer
import faiss
try:
    import PyPDF2 as pypdf
except Exception:
    import pypdf as pypdf

st.set_page_config(page_title="RAG over PDFs", layout="wide")
st.title("💬 Chatbot — Retrieval-Augmented Generation (PDFs)")

docs_dir = st.sidebar.text_input("Docs folder", "sample_docs")
model_name = st.sidebar.selectbox("Embedding model", [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "intfloat/multilingual-e5-base",
    "sentence-transformers/all-MiniLM-L6-v2"
])
topk = st.sidebar.slider("Top-K passages", 2, 8, 4)

def read_pdf_text(path):
    chunks = []
    with open(path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                chunks.append((i+1, text))
    return chunks

def chunk_text(text, chunk_size=180, overlap=40):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

@st.cache_resource
def build_index(docs_dir, model_name):
    docs = []
    for fname in sorted(os.listdir(docs_dir)):
        if fname.lower().endswith(".pdf"):
            for page, text in read_pdf_text(os.path.join(docs_dir, fname)):
                for ch in chunk_text(text):
                    docs.append({"file": fname, "page": page, "text": ch})
    model = SentenceTransformer(model_name)
    embs = model.encode([d["text"] for d in docs], show_progress_bar=False, normalize_embeddings=True)
    embs = np.array(embs).astype("float32")
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    return docs, model, index

if os.path.isdir(docs_dir):
    docs, model, index = build_index(docs_dir, model_name)
    st.success(f"Indexed {len(docs)} chunks from PDFs in '{docs_dir}'")
    q = st.text_input("Ask a question")
    if q:
        q_emb = model.encode([q], normalize_embeddings=True).astype("float32")
        D, I = index.search(q_emb, topk)
        hits = [docs[int(i)] | {"score": float(s)} for i, s in zip(I[0], D[0])]
        st.subheader("Retrieved Contexts with Citations")
        for h in hits:
            st.markdown(f"**({h['file']}:{h['page']})** {h['text']}")
        st.caption("Note: You can plug a small seq2seq model (e.g., flan-t5-base) to generate a final answer from these contexts.")
else:
    st.warning("Docs folder not found. Make sure it exists and contains PDFs.")
