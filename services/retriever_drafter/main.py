import os
import math
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import bigquery
from vertexai import init as vertex_init
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel

# --- Config: hard-code Vertex location to us-central1 for Gemini ---

PROJECT_ID = (
    os.environ.get("GCP_PROJECT")
    or os.environ.get("GOOGLE_CLOUD_PROJECT")
    or "rate-case-app-hackathon"
)

BQ_DATASET = os.environ.get("BQ_DATASET", "rc")

# 🔥 Force Vertex calls (embeddings + Gemini) to us-central1
VERTEX_LOCATION = "us-central1"

EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-004")
GEN_MODEL = os.environ.get("GEN_MODEL", "gemini-1.5-flash")

app = FastAPI(title="Retriever & Drafter (Vertex + BigQuery)")


def get_bq_client() -> bigquery.Client:
    return bigquery.Client(project=PROJECT_ID)


def get_embedding_model() -> TextEmbeddingModel:
    vertex_init(project=PROJECT_ID, location=VERTEX_LOCATION)
    return TextEmbeddingModel.from_pretrained(EMBED_MODEL)


def get_gen_model() -> GenerativeModel:
    vertex_init(project=PROJECT_ID, location=VERTEX_LOCATION)
    return GenerativeModel(GEN_MODEL)


class Question(BaseModel):
    question: str
    k: int = 5  # number of chunks to retrieve


class Answer(BaseModel):
    answer: str
    supporting_chunks: List[str]


def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb + 1e-8)


@app.post("/ask", response_model=Answer)
async def ask(q: Question):
    """
    - Embeds the incoming question with Vertex AI.
    - Loads embedded chunks from BigQuery.
    - Ranks by cosine similarity.
    - RETURNS a simple heuristic "answer" plus supporting chunk IDs.
      (No Gemini call – project does not have access to gemini-1.5-flash.)
    """
    if not q.question or not q.question.strip():
        raise HTTPException(status_code=400, detail="question must be non-empty")

    # 1) Embed the question
    emb_model = get_embedding_model()
    q_emb = emb_model.get_embeddings([q.question])[0].values

    # 2) Fetch embedded chunks from BigQuery
    bq_client = get_bq_client()
    query = f"""
    SELECT e.chunk_id, e.emb, c.text
    FROM `{PROJECT_ID}.{BQ_DATASET}.embeddings` e
    JOIN `{PROJECT_ID}.{BQ_DATASET}.document_chunks` c
      ON e.chunk_id = c.chunk_id
    LIMIT 200
    """
    rows = list(bq_client.query(query))

    if not rows:
        return Answer(
            answer=(
                "No embedded context is available in BigQuery. "
                "Make sure ingest + embedder have populated "
                f"`{PROJECT_ID}.{BQ_DATASET}.embeddings`."
            ),
            supporting_chunks=[],
        )

    # 3) Score by cosine similarity
    scored = []
    for r in rows:
        emb = r["emb"]  # ARRAY<FLOAT64>
        score = cosine(q_emb, emb)
        scored.append((score, r))

    scored.sort(reverse=True, key=lambda x: x[0])
    top = scored[: max(1, q.k)]

    top_texts = [r["text"] for _, r in top]
    top_ids = [r["chunk_id"] for _, r in top]

    # 4) Build a simple "answer" by stitching the top chunks
    # (For hackathon: NO Gemini call, just retrieval.)
    context_preview = "\n\n".join(top_texts)

    answer_text = (
        "Here are the most relevant context chunks I found for your question. "
        "In a full RAG setup, these would be passed to a generative model like Gemini "
        "to draft a natural language answer:\n\n"
        f"{context_preview[:4000]}"  # guardrail on size
    )

    return Answer(
        answer=answer_text,
        supporting_chunks=top_ids,
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "project": PROJECT_ID,
        "dataset": BQ_DATASET,
        "vertex_location": VERTEX_LOCATION,
        "embed_model": EMBED_MODEL,
        "gen_model": GEN_MODEL,
    }
