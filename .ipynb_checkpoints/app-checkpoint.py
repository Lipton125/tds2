from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import numpy as np
import openai
import os

# Make sure your OpenAI API key is set as an environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

# Request payload schema
class SimilarityRequest(BaseModel):
    docs: List[str]
    query: str

# Response payload schema
class SimilarityResponse(BaseModel):
    matches: List[str]

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@app.post("/similarity", response_model=SimilarityResponse)
async def similarity(request: SimilarityRequest):
    # Embed all texts using OpenAI embedding model
    texts = request.docs + [request.query]
    response = openai.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    
    embeddings = [np.array(e.embedding) for e in response.data]
    doc_embeddings = embeddings[:-1]
    query_embedding = embeddings[-1]

    # Compute cosine similarity for each document
    scores = [
        (doc, cosine_similarity(query_embedding, emb))
        for doc, emb in zip(request.docs, doc_embeddings)
    ]

    # Sort by similarity score (highest first)
    top_matches = sorted(scores, key=lambda x: x[1], reverse=True)[:3]
    return SimilarityResponse(matches=[doc for doc, _ in top_matches])
