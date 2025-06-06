# app/rag.py (Updated for Groq + llama3-70b-8192 + Multi-document support)

import os
import faiss
import pickle
import requests
from sentence_transformers import SentenceTransformer
from app.utils import chunk_text

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
INDEX_DIR = "indexes"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def get_answer(query, doc_paths, k=2):
    if not isinstance(doc_paths, list):
        doc_paths = [doc_paths]

    all_chunks = []

    for doc_path in doc_paths:
        index_file = os.path.join(INDEX_DIR, os.path.basename(doc_path) + ".index")
        if not os.path.exists(index_file):
            continue

        index = faiss.read_index(index_file)
        with open(index_file + ".meta", "rb") as f:
            chunks = pickle.load(f)

        query_embedding = embedding_model.encode([query])
        _, indices = index.search(query_embedding, k)
        relevant_chunks = [chunks[i] for i in indices[0] if i < len(chunks)]
        all_chunks.extend(relevant_chunks)

    if not all_chunks:
        return "No relevant content found across uploaded documents."

    # Prepare context
    context = "\n---\n".join(all_chunks)
    prompt = f"""You are a helpful assistant. Use the context below to answer the question.

Context:
{context}

Question: {query}
Answer:"""

    # Call Groq LLM
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": "You answer questions based on document context."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }

    response = requests.post(url, headers=headers, json=payload)
    try:
        return response.json()["choices"][0]["message"]["content"]
    except Exception:
        return "Groq API error: " + str(response.json())
