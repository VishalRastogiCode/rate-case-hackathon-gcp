# Rate Case Advisor – GCP Hackathon (BigQuery + Vertex)

This repo contains a minimum viable **end-to-end GCP implementation** for the Rate Case Advisor:

- `ingest-chunker`: accepts file uploads, stores to GCS, chunks text, and writes chunks into BigQuery.
- `embedder`: reads chunks from BigQuery, calls **Vertex AI Embeddings**, and writes embeddings back to BigQuery.
- `retriever-drafter`: embeds the user's question, retrieves top chunks from BigQuery, and calls **Gemini** to draft an answer.

All three services are designed to run on **Cloud Run** with Dockerfiles included.

## GCP prerequisites

Before deploying:

1. Enable APIs:
   - Cloud Run
   - Cloud Build
   - BigQuery
   - Vertex AI
   - Cloud Storage

2. Create a BigQuery dataset, e.g.:

   ```sql
   CREATE SCHEMA rc;
   ```

3. Create tables:

   ```sql
   CREATE TABLE rc.document_chunks (
     chunk_id STRING,
     doc_id STRING,
     page_start INT64,
     page_end INT64,
     text STRING
   );

   CREATE TABLE rc.embeddings (
     chunk_id STRING,
     emb ARRAY<FLOAT64>
   );
   ```

4. Create a GCS bucket, e.g. `rate-case-docs-us`.

5. Configure the following env vars on each Cloud Run service:

   - `GCP_PROJECT`       – your project ID
   - `BQ_DATASET`        – e.g. `rc`
   - `GCS_BUCKET`        – for ingest-chunker only
   - `VERTEX_LOCATION`   – e.g. `us-central1` (for embedder & retriever)
   - Optionally: `GEN_MODEL`, `EMBED_MODEL` to override defaults.

See each service's `main.py` for details.
