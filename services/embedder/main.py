import os
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import bigquery
from vertexai import init as vertex_init
from vertexai.language_models import TextEmbeddingModel

# --- Config (no hard crash at import) ---

PROJECT_ID = (
    os.environ.get("GCP_PROJECT")
    or os.environ.get("GOOGLE_CLOUD_PROJECT")
    or "rate-case-app-hackathon"  # fallback for safety
)

BQ_DATASET = os.environ.get("BQ_DATASET", "rc")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "us-east4")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-004")

app = FastAPI(title="Embedder (Vertex AI + BigQuery)")


# --- Lazy clients/helpers so startup never crashes ---

def require_project_id() -> str:
    if not PROJECT_ID:
        # For Cloud Run, this should be set automatically, but if not,
        # we fail at *request* time, not import time.
        raise HTTPException(
            status_code=500,
            detail="GCP_PROJECT or GOOGLE_CLOUD_PROJECT env var must be set for embedder",
        )
    return PROJECT_ID


def get_bq_client() -> bigquery.Client:
    project_id = require_project_id()
    return bigquery.Client(project=project_id)


def get_embedding_model() -> TextEmbeddingModel:
    project_id = require_project_id()
    vertex_init(project=project_id, location=VERTEX_LOCATION)
    return TextEmbeddingModel.from_pretrained(EMBED_MODEL)


# --- Pydantic models ---

class EmbedRequest(BaseModel):
    limit: int = 50  # max number of new chunks to embed


class EmbedResponse(BaseModel):
    embedded_count: int


# --- API Endpoints ---

@app.post("/embed", response_model=EmbedResponse)
async def embed_new_chunks(req: EmbedRequest):
    """
    - Finds chunks in rc.document_chunks that do NOT yet have embeddings
      in rc.embeddings.
    - Calls Vertex AI Embeddings.
    - Writes vectors into rc.embeddings.
    """
    if req.limit <= 0:
        raise HTTPException(status_code=400, detail="limit must be > 0")

    bq_client = get_bq_client()

    query = f"""
    SELECT c.chunk_id, c.text
    FROM `{PROJECT_ID}.{BQ_DATASET}.document_chunks` c
    LEFT JOIN `{PROJECT_ID}.{BQ_DATASET}.embeddings` e
    ON c.chunk_id = e.chunk_id
    WHERE e.chunk_id IS NULL
    LIMIT {req.limit}
    """
    rows = list(bq_client.query(query))

    if not rows:
        return EmbedResponse(embedded_count=0)

    texts = [r["text"] for r in rows]
    chunk_ids = [r["chunk_id"] for r in rows]

    model = get_embedding_model()
    embeddings = model.get_embeddings(texts)

    table_id = f"{PROJECT_ID}.{BQ_DATASET}.embeddings"
    to_insert = []
    for cid, emb in zip(chunk_ids, embeddings):
        to_insert.append(
            {
                "chunk_id": cid,
                "emb": emb.values,  # ARRAY<FLOAT64>
            }
        )

    errors = bq_client.insert_rows_json(table_id, to_insert)
    if errors:
        raise HTTPException(
            status_code=500,
            detail=f"BigQuery insert errors: {errors}",
        )

    return EmbedResponse(embedded_count=len(to_insert))


@app.get("/")
async def root():
    return {
        "service": "embedder",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health():
    """
    Health + config visibility endpoint.
    """
    return {
        "status": "ok",
        "project": PROJECT_ID or "(missing)",
        "dataset": BQ_DATASET,
        "vertex_location": VERTEX_LOCATION,
        "embed_model": EMBED_MODEL,
    }
