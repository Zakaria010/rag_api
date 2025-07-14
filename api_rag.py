import os
from fastapi import FastAPI, Request
from pydantic import BaseModel
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from utils.retrieval import retrieve_relevant_chunks

# ---- CONFIG ----
PDF_FOLDER = "./data"
CHUNK_SIZE = 500
OVERLAP = 100
INDEX_TYPE = "innerproduct"
TOP_K = 5
EMBEDDER_NAME = "all-MiniLM-L6-v2"

# ---- INIT ----
app = FastAPI()

# Load index and chunks at startup
def prepare_index_and_chunks(pdf_folder, chunk_size, overlap, index_type, embedder_name):
    folder_name = f"{embedder_name} ; {index_type}_chunk{chunk_size}_overlap{overlap}"
    folder_path = Path(folder_name)
    faiss_index_path = str(folder_path / f"index_{index_type}.idx")
    chunks_path = folder_path / f"chunks_{chunk_size}_{overlap}.npy"
    return faiss_index_path, chunks_path

embedder = SentenceTransformer(EMBEDDER_NAME)
faiss_index, chunks_path = prepare_index_and_chunks(
    PDF_FOLDER, CHUNK_SIZE, OVERLAP, INDEX_TYPE, EMBEDDER_NAME
)

# ---- FastAPI Models ----
class QueryRequest(BaseModel):
    query: str
    top_k: int = TOP_K

@app.post("/query")
def query_rag(request: QueryRequest):
    retrieved_chunks = retrieve_relevant_chunks(
        request.query, embedder, TOP_K, faiss_index, chunks_path
    )
    return {"context": retrieved_chunks}
