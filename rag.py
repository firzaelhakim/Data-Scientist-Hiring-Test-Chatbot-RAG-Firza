import argparse, os, re
import numpy as np
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer
import faiss
try:
    import PyPDF2 as pypdf
except Exception:
    import pypdf as pypdf

SEP = "\n---\n"

def read_pdf_text(path: str) -> List[Tuple[int, str]]:
    chunks = []
    with open(path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                chunks.append((i+1, text))
    return chunks

def chunk_text(text: str, chunk_size: int = 180, overlap: int = 40) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

def load_corpus(doc_dir: str) -> List[Dict]:
    docs = []
    for fname in os.listdir(doc_dir):
        if not fname.lower().endswith(".pdf"):
            continue
        path = os.path.join(doc_dir, fname)
        for page_num, text in read_pdf_text(path):
            for ch in chunk_text(text):
                docs.append({"file": fname, "page": page_num, "text": ch})
    if not docs:
        raise SystemExit(f"No PDF chunks found in {doc_dir}")
    return docs

def build_index(docs: List[Dict], model_name: str):
    model = SentenceTransformer(model_name)
    embs = model.encode([d['text'] for d in docs], show_progress_bar=True, normalize_embeddings=True)
    embs = np.array(embs).astype("float32")
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    return index, embs, model

def retrieve(query: str, index, model, docs: List[Dict], top_k: int = 4):
    q_emb = model.encode([query], normalize_embeddings=True)
    q_emb = np.array(q_emb).astype("float32")
    D, I = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        d = docs[int(idx)]
        results.append({**d, "score": float(score)})
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=str, default="sample_docs")
    ap.add_argument("--model", type=str, default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--query", type=str, default="Apa tantangan utama transisi energi di Indonesia?")
    args = ap.parse_args()

    docs = load_corpus(args.docs)
    index, embs, model = build_index(docs, args.model)
    hits = retrieve(args.query, index, model, docs, args.topk)

    print("=== QUERY ===")
    print(args.query)
    print("\n=== CONTEXT (CITED) ===")
    for h in hits:
        print(f"({h['file']}:{h['page']}) {h['text']}\n")

    print("Tip: For generative answer, combine with a small seq2seq model (e.g., flan-t5-base) or simply present the cited contexts above.")

if __name__ == "__main__":
    main()
