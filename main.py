from fastapi import FastAPI, Form, File, UploadFile
from typing import Optional
import base64
import os
from retrieval import load_jsonl, search_index
from answer_generator import generate_answer

app = FastAPI()

discourse_docs = load_jsonl("data/DiscourseData.jsonl")
course_docs = load_jsonl("data/CourseContentData.jsonl")
all_docs = discourse_docs + course_docs
index_path = "tds_index.faiss"

if not os.path.exists(index_path):
    from retrieval import build_faiss_index
    build_faiss_index(all_docs, index_path)

@app.post("/api/")
async def answer_question(
    question: str = Form(...),
    file: Optional[UploadFile] = File(None)
):
    # Read and convert uploaded file to base64 if present
    image_b64 = None
    if file:
        contents = await file.read()
        image_b64 = base64.b64encode(contents).decode('utf-8')
    
    # Use question and image_b64 for answer generation as needed
    indices = search_index(index_path, question, k=3)
    context = [all_docs[i]['content'] for i in indices]
    
    # Pass image_b64 if your generate_answer supports it, else just question & context
    answer = generate_answer(question, context, image_b64=image_b64)

    links = [{"url": doc.get("url", ""), "text": doc.get("content", "")[:80]} for i, doc in enumerate(all_docs) if i in indices]

    return {
        "answer": answer,
        "links": links
    }

