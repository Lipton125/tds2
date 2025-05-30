from sentence_transformers import SentenceTransformer
import faiss
import json
import os

model = SentenceTransformer("all-MiniLM-L6-v2")

def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def build_faiss_index(docs, index_path):
    texts = [doc['content'] for doc in docs]
    embeddings = model.encode(texts, show_progress_bar=True)
    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    return index, texts

def search_index(index_path, query, k=3):
    index = faiss.read_index(index_path)
    query_vector = model.encode([query])
    D, I = index.search(query_vector, k)
    return I[0]

