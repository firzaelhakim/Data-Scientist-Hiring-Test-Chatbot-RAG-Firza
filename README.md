
---

## 💬 Chatbot — RAG over PDFs

**Files**: `rag.py`, `rag_app.py`, `sample_docs/` (contains example PDFs).  
**How to run**:
```bash
python rag.py --docs sample_docs --query "Apa tantangan utama transisi energi di Indonesia?" --topk 4
streamlit run rag_app.py
```
Output includes retrieved **contexts with citations** `(file:page)`. You can extend `rag.py` to use a generator model (e.g., `google/flan-t5-base`) for abstractive answers.
