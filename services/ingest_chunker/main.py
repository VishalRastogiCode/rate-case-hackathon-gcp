import os
import uuid

from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from google.cloud import storage, bigquery
import httpx  # <-- add this

# Try to resolve project ID, but DO NOT crash if missing.
PROJECT_ID = (
    os.environ.get("GCP_PROJECT")
    or os.environ.get("GOOGLE_CLOUD_PROJECT")
    or "rate-case-app-hackathon"  # safe default for Cloud Run in your project
)

BQ_DATASET = os.environ.get("BQ_DATASET", "rc")
GCS_BUCKET = os.environ.get("GCS_BUCKET", "")  # empty string if not set
EMBEDDER_URL = os.environ.get("EMBEDDER_URL")  # <-- add this

app = FastAPI(title="Ingest & Chunker (GCS + BigQuery)")


def get_storage_client() -> storage.Client:
    return storage.Client(project=PROJECT_ID)


def get_bq_client() -> bigquery.Client:
    return bigquery.Client(project=PROJECT_ID)


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    page_start: int
    page_end: int
    text: str


def upload_to_gcs(file_bytes: bytes, filename: str) -> str:
    if not GCS_BUCKET:
        raise RuntimeError("GCS_BUCKET env var must be set for ingest-chunker")

    client = get_storage_client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(filename)
    blob.upload_from_string(file_bytes)
    return f"gs://{GCS_BUCKET}/{filename}"


def write_chunks_to_bq(chunks: List[Chunk]) -> int:
    client = get_bq_client()
    table_id = f"{PROJECT_ID}.{BQ_DATASET}.document_chunks"
    rows = [
        {
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "page_start": c.page_start,
            "page_end": c.page_end,
            "text": c.text,
        }
        for c in chunks
    ]
    errors = client.insert_rows_json(table_id, rows)
    if errors:
        raise RuntimeError(f"BigQuery insert errors: {errors}")
    return len(rows)

async def trigger_embedding(limit: int = 50):
    """
    Call the embedder service to embed any new chunks.
    This is best-effort: failure should NOT break the ingest request.
    """
    if not EMBEDDER_URL:
        # No embedder configured; just skip
        return

    url = EMBEDDER_URL.rstrip("/") + "/embed"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json={"limit": limit})
            resp.raise_for_status()
            # Optional: log or print response if needed
            print(f"Triggered embedder: {resp.json()}")
    except Exception as e:
        # Best-effort: log but don't raise, so ingest still succeeds
        print(f"Failed to trigger embedder: {e}")

@app.post("/ingest", response_model=List[Chunk])
async def ingest_pdf(file: UploadFile = File(...)):
    """
    Minimal viable ingest for hackathon:
    - Uploads the raw file to GCS
    - Decodes text (best-effort) from bytes
    - Stores one chunk per file into BigQuery.rc.document_chunks
    """

    if not GCS_BUCKET:
        raise HTTPException(
            status_code=500,
            detail="GCS_BUCKET env var must be set for ingest-chunker",
        )

    content = await file.read()
    if not content:
        return []

    # Store raw file in GCS for traceability
    filename = f"uploads/{uuid.uuid4()}_{file.filename}"
    gcs_uri = upload_to_gcs(content, filename)

    # For hackathon, do simple text decoding instead of full DocAI OCR
    try:
        text = content.decode("utf-8", errors="ignore")
    except Exception:
        text = "Unable to decode file contents as text."

    doc_id = str(uuid.uuid4())
    chunk = Chunk(
        chunk_id=str(uuid.uuid4()),
        doc_id=doc_id,
        page_start=1,
        page_end=1,
        text=f"[GCS:{gcs_uri}]\n\n{text[:8000]}",
    )

    write_chunks_to_bq([chunk])

    # Trigger embedding asynchronously (best-effort)
    await trigger_embedding(limit=50)
    
    return [chunk]


@app.get("/")
def root():
    return {
        "service": "ingest-chunker",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health():
    """
    Lightweight health check that also shows what config
    the service is currently seeing.
    """
    return {
        "status": "ok",
        "project": PROJECT_ID,
        "dataset": BQ_DATASET,
        "bucket": GCS_BUCKET,
    }
