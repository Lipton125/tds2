from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional
import base64
import os
from retrieval import load_jsonl, search_index
from answer_generator import generate_answer

app = FastAPI()

class Query(BaseModel):
    question: str
    image: Optional[str] = None  # base64-encoded

discourse_docs = load_jsonl("data/DiscourseData.jsonl")
course_docs = load_jsonl("data/CourseContentData.jsonl")
all_docs = discourse_docs + course_docs
index_path = "tds_index.faiss"

# Build FAISS index once
if not os.path.exists(index_path):
    from retrieval import build_faiss_index
    build_faiss_index(all_docs, index_path)

@app.post("/api/")
async def answer_question(query: Query):
    indices = search_index(index_path, query.question, k=3)
    context = [all_docs[i]['content'] for i in indices]
    answer = generate_answer(query.question, context)

    # Dummy links (replace with actual discourse URLs if present)
    links = [{"url": doc.get("url", ""), "text": doc.get("content", "")[:80]} for i, doc in enumerate(all_docs) if i in indices]

    return {
        "answer": answer,
        "links": links
    }

