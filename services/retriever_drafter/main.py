import os
import math
import logging

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
GEN_MODEL = os.environ.get("GEN_MODEL", "gemini-2.5-flash-lite")

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
    - Calls Gemini (GEN_MODEL, e.g. gemini-2.5-flash-lite) to draft a natural language answer
      using the top-k chunks as context.
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
                f"Make sure ingest + embedder have populated "
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

    # 4) Build context string for Gemini
    context = "\n\n".join(f"- {t}" for t in top_texts)

    # 5) Call Gemini to synthesize the final answer
    gen_model = get_gen_model()
    prompt = (
        "You are a regulatory rate case assistant. Use ONLY the context below to answer.\n\n"
        "If the context does not contain the answer, say so explicitly and do not hallucinate.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{q.question}\n\n"
        "Provide a clear, concise answer in English, with a short explanation that references "
        "key drivers (e.g., major cost categories, test year vs. historical trend, and any "
        "one-time or ongoing adjustments)."
    )

    try:
        resp = gen_model.generate_content(prompt)
        # For the Vertex AI Python SDK, resp.text is the usual convenience property:
        answer_text = getattr(resp, "text", None) or str(resp)
    except Exception as e:
        # If Gemini fails for any reason, fall back to returning context only
        answer_text = (
            "Gemini call failed, but here are the top relevant context chunks:\n\n"
            f"{context[:4000]}\n\n"
            f"(Error from Gemini: {e})"
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

logger = logging.getLogger("retriever")
logging.basicConfig(level=logging.INFO)

@app.post("/ask", response_model=Answer)
async def ask(q: Question):
    """
    - Embeds the incoming question with Vertex AI.
    - Loads embedded chunks from BigQuery.
    - Ranks by cosine similarity.
    - Calls Gemini (GEN_MODEL) to draft a natural language answer.
    """
    if not q.question or not q.question.strip():
        raise HTTPException(status_code=400, detail="question must be non-empty")

    logger.info(f"[ASK] Received question: {q.question!r} (k={q.k})")

    # 1) Embed the question
    emb_model = get_embedding_model()
    logger.info(f"[ASK] Using embedding model: {EMBED_MODEL} in {VERTEX_LOCATION}")
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
    logger.info(f"[ASK] Querying BigQuery: {PROJECT_ID}.{BQ_DATASET}.embeddings + document_chunks")
    rows = list(bq_client.query(query))

    if not rows:
        logger.warning("[ASK] No rows returned from embeddings join; returning fallback message.")
        return Answer(
            answer=(
                "No embedded context is available in BigQuery. "
                f"Make sure ingest + embedder have populated "
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

    logger.info(f"[ASK] Using top {len(top_ids)} chunks as context for Gemini. Chunk IDs: {top_ids}")

    # 4) Build context string for Gemini
    context = "\n\n".join(f"- {t}" for t in top_texts)

    # 5) Call Gemini to synthesize the final answer
    gen_model = get_gen_model()
    logger.info(f"[ASK] Calling Gemini model: {GEN_MODEL} in {VERTEX_LOCATION}")

    prompt = (
        "You are a regulatory rate case assistant. Use ONLY the context below to answer.\n\n"
        "If the context does not contain the answer, say so explicitly and do not hallucinate.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{q.question}\n\n"
        "Provide a clear, concise answer in English, with a short explanation that references "
        "key drivers (e.g., major cost categories, test year vs. historical trend, and any "
        "one-time or ongoing adjustments)."
    )

    try:
        resp = gen_model.generate_content(prompt)
        answer_text = getattr(resp, "text", None) or str(resp)
        answer_text = f"[Generated by {GEN_MODEL} @ {VERTEX_LOCATION}]\n\n" + answer_text
        logger.info("[ASK] Gemini call succeeded.")
    except Exception as e:
        logger.exception("[ASK] Gemini call failed; falling back to raw context.")
        answer_text = (
            "Gemini call failed, but here are the top relevant context chunks:\n\n"
            f"{context[:4000]}\n\n"
            f"(Error from Gemini: {e})"
        )

    return Answer(
        #answer=answer_text,
        answer_text = f"[Generated by {GEN_MODEL} @ {VERTEX_LOCATION}]\n\n{resp.text}",
        supporting_chunks=top_ids,
    )