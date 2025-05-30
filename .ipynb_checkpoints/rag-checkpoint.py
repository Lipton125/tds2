import json
import os
import faiss
import numpy as np
import uuid
import openai
from tqdm import tqdm

openai.api_key = os.getenv("AIPIPE_TOKEN")  # or hardcode it
openai.api_base = "https://aipipe.org/openai/v1"

EMBED_MODEL = "text-embedding-3-small"
INDEX_FILE = "tds_index.faiss"
META_FILE = "tds_metadata.jsonl"

def load_chunks(filepaths):
    chunks = []
    for path in filepaths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                text = data.get("text") or data.get("content") or ""
                if not text.strip(): continue
                # Split large chunks into ~500 token segments
                for i in range(0, len(text), 2000):
                    chunk = text[i:i+2000]
                    chunks.append({
                        "id": str(uuid.uuid4()),
                        "text": chunk,
                        "source": data.get("source") or data.get("post_url"),
                        "title": data.get("title") or "Discourse Post"
                    })
    return chunks

def embed_chunks(chunks):
    embeddings = []
    batch_size = 50
    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding"):
        batch = chunks[i:i+batch_size]
        texts = [c["text"] for c in batch]
        resp = openai.Embedding.create(model=EMBED_MODEL, input=texts)
        for j, e in enumerate(resp["data"]):
            chunks[i + j]["embedding"] = e["embedding"]
            embeddings.append(e["embedding"])
    return chunks, np.array(embeddings).astype("float32")

def save_index(chunks, embeddings):
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(embeddings)
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    print(f"✅ Saved FAISS index to {INDEX_FILE}")
    print(f"✅ Saved metadata to {META_FILE}")

if __name__ == "__main__":
    chunks = load_chunks(["CourseContentData.jsonl", "DicourseData.jsonl"])
    chunks, embeddings = embed_chunks(chunks)
    save_index(chunks, embeddings)
