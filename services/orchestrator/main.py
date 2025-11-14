import os
from typing import List, Dict, Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# ------------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------------

RETRIEVER_URL = os.environ.get("RETRIEVER_URL")
if not RETRIEVER_URL:
    print("WARNING: RETRIEVER_URL not set. Orchestrator will not be able to call retriever.")
RETRIEVER_URL = (RETRIEVER_URL or "").rstrip("/")

app = FastAPI(title="Rate Case Orchestrator (Multi-Agent Demo)")


# ------------------------------------------------------------------------------------
# Pydantic models
# ------------------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str


class SubQueryResult(BaseModel):
    subquery: str
    retriever_answer: str
    supporting_chunks: List[str]


class AgentReview(BaseModel):
    agent_name: str
    status: str
    comments: List[str]


class OrchestratedResponse(BaseModel):
    original_question: str
    subqueries: List[str]
    subquery_results: List[SubQueryResult]
    final_answer: str
    reviews: List[AgentReview]


# ------------------------------------------------------------------------------------
# "Agents" (implemented as helper functions)
# ------------------------------------------------------------------------------------

def decompose_question(question: str) -> List[str]:
    """
    Decomposition Agent:
    Very simple heuristic decomposition for demo purposes.
    - If ' and ' present, split into 2 subquestions.
    - Otherwise, return the full question as a single subquery.
    """
    q = question.strip()
    if not q:
        return []

    if " and " in q.lower():
        parts = q.split(" and ", 1)
        sub1 = parts[0].strip()
        sub2 = parts[1].strip().rstrip("?")
        subqs: List[str] = []
        if sub1:
            subqs.append(sub1 + "?")
        if sub2:
            subqs.append(sub2 + "?")
        return subqs

    return [q]


async def call_retriever(subquery: str, k: int = 5) -> Dict[str, Any]:
    """
    Retrieval Agent:
    Calls your existing retriever service's /ask endpoint.
    Expected retriever API:
        POST /ask  { "question": str, "k": int }
        -> { "answer": str, "supporting_chunks": [chunk_id, ...] }
    """
    if not RETRIEVER_URL:
        raise RuntimeError("RETRIEVER_URL is not configured.")

    url = f"{RETRIEVER_URL}/ask"
    payload = {"question": subquery, "k": k}

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(url, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Retriever returned {resp.status_code}: {resp.text}"
            )
        return resp.json()


def synthesize_answer(question: str, sub_results: List[SubQueryResult]) -> str:
    """
    Synthesis Agent:
    Combines the per-subquery retriever answers into one narrative.
    For demo: simple concatenation with headings.
    """
    lines: List[str] = []
    lines.append("### Final Synthesized Answer")
    lines.append("")
    lines.append(f"**Original question:** {question}")
    lines.append("")

    for idx, r in enumerate(sub_results, start=1):
        lines.append(f"**Sub-question {idx}:** {r.subquery}")
        lines.append(r.retriever_answer.strip() or "(no answer returned)")
        lines.append("")

    lines.append(
        "This answer was synthesized from multiple sub-questions. "
        "In a production setup, this synthesis could be driven by a more advanced LLM-based agent."
    )

    return "\n".join(lines)


def response_validator_agent(sub_results: List[SubQueryResult], final_answer: str) -> AgentReview:
    """
    Response Validator Agent:
    Very light checks for demo purposes.
    """
    comments: List[str] = []

    if not final_answer.strip():
        comments.append("Final answer is empty.")
        status = "needs_changes"
    else:
        if len(sub_results) == 0:
            comments.append("No subquery results were available.")
            status = "needs_changes"
        else:
            comments.append("Answer contains content for at least one sub-question.")
            status = "pass"

    return AgentReview(
        agent_name="Response Validator Agent",
        status=status,
        comments=comments,
    )


def business_reviewer_agent(final_answer: str) -> AgentReview:
    """
    Business Reviewer Agent:
    Stubbed for demo, but you can enrich prompts later.
    """
    comments: List[str] = [
        "High-level check: answer mentions O&M and test year concepts.",
        "For production, this agent would validate alignment with accounting/financial guidelines."
    ]
    status = "pass" if "O&M" in final_answer or "operating" in final_answer.lower() else "needs_review"

    return AgentReview(
        agent_name="Business Reviewer Agent",
        status=status,
        comments=comments,
    )


def legal_reviewer_agent(final_answer: str) -> AgentReview:
    """
    Legal Reviewer Agent:
    Stubbed for demo. In a full system, this would compare against legal/regulatory narratives.
    """
    comments: List[str] = [
        "Automated legal pass: no explicit contradictory regulatory statements detected in this simple check.",
        "In production, this would cross-check against testimony, orders, and regulatory narratives."
    ]
    # Always "needs_review" in demo to show that human review is recommended
    status = "needs_review"

    return AgentReview(
        agent_name="Legal Reviewer Agent",
        status=status,
        comments=comments,
    )


# ------------------------------------------------------------------------------------
# FastAPI endpoints
# ------------------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "retriever_url": RETRIEVER_URL or "(missing)",
    }


@app.post("/answer", response_model=OrchestratedResponse)
async def answer(req: AskRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="question must be non-empty")

    # 1) Decomposition Agent
    subqueries = decompose_question(req.question)
    if not subqueries:
        raise HTTPException(status_code=400, detail="failed to decompose question")

    # 2) Retrieval Agents (one call to retriever per subquery)
    sub_results: List[SubQueryResult] = []
    for sq in subqueries:
        try:
            retriever_resp = await call_retriever(sq, k=5)
            retriever_answer = retriever_resp.get("answer", "")
            supporting_chunks = retriever_resp.get("supporting_chunks", [])
        except Exception as e:
            retriever_answer = f"[Error calling retriever: {e}]"
            supporting_chunks = []

        sub_results.append(
            SubQueryResult(
                subquery=sq,
                retriever_answer=retriever_answer,
                supporting_chunks=supporting_chunks,
            )
        )

    # 3) Synthesis Agent
    final_answer = synthesize_answer(req.question, sub_results)

    # 4) Validator + Business + Legal Agents
    reviews: List[AgentReview] = []
    reviews.append(response_validator_agent(sub_results, final_answer))
    reviews.append(business_reviewer_agent(final_answer))
    reviews.append(legal_reviewer_agent(final_answer))

    return OrchestratedResponse(
        original_question=req.question,
        subqueries=subqueries,
        subquery_results=sub_results,
        final_answer=final_answer,
        reviews=reviews,
    )


# ------------------------------------------------------------------------------------
# Simple HTML UI
# ------------------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def ui():
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <title>Rate Case Agentic Orchestrator Demo</title>
        <style>
            body {
                font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                margin: 0;
                padding: 0;
                background: #0f172a;
                color: #e5e7eb;
            }
            .container {
                max-width: 1100px;
                margin: 0 auto;
                padding: 24px 16px 48px;
            }
            h1 {
                font-size: 1.8rem;
                margin-bottom: 0.25rem;
            }
            h2 {
                font-size: 1.3rem;
                margin-top: 1.5rem;
                margin-bottom: 0.5rem;
            }
            .subtitle {
                color: #9ca3af;
                margin-bottom: 1.5rem;
            }
            .card {
                background: #020617;
                border-radius: 16px;
                padding: 20px;
                box-shadow: 0 15px 35px rgba(0,0,0,0.35);
                border: 1px solid #1f2937;
                margin-bottom: 1.5rem;
            }
            textarea {
                width: 100%;
                min-height: 120px;
                border-radius: 12px;
                border: 1px solid #374151;
                padding: 10px 12px;
                font-size: 0.95rem;
                background: #020617;
                color: #e5e7eb;
                resize: vertical;
            }
            textarea:focus {
                outline: none;
                border-color: #38bdf8;
                box-shadow: 0 0 0 1px #38bdf8;
            }
            button {
                margin-top: 12px;
                padding: 10px 18px;
                border-radius: 999px;
                border: none;
                font-size: 0.95rem;
                font-weight: 600;
                background: linear-gradient(135deg, #22c55e, #06b6d4);
                color: white;
                cursor: pointer;
                display: inline-flex;
                align-items: center;
                gap: 8px;
            }
            button:disabled {
                opacity: 0.6;
                cursor: wait;
            }
            .pill {
                display: inline-flex;
                align-items: center;
                padding: 4px 10px;
                border-radius: 999px;
                background: #111827;
                border: 1px solid #1f2937;
                font-size: 0.75rem;
                color: #9ca3af;
                margin-right: 6px;
                margin-bottom: 4px;
            }
            .pill span {
                width: 8px;
                height: 8px;
                border-radius: 999px;
                margin-right: 6px;
            }
            .pill-success span { background: #22c55e; }
            .pill-warn span { background: #eab308; }
            .pill-fail span { background: #ef4444; }
            pre {
                background: #020617;
                border-radius: 12px;
                padding: 12px 14px;
                font-size: 0.85rem;
                overflow-x: auto;
                border: 1px solid #1f2937;
            }
            .section {
                margin-bottom: 1.25rem;
            }
            .label {
                font-size: 0.75rem;
                text-transform: uppercase;
                letter-spacing: 0.06em;
                color: #9ca3af;
                margin-bottom: 0.25rem;
            }
            .muted {
                color: #9ca3af;
                font-size: 0.85rem;
            }
            .flex {
                display: flex;
                gap: 16px;
                flex-wrap: wrap;
            }
            .flex-2 {
                flex: 2 1 380px;
            }
            .flex-1 {
                flex: 1 1 260px;
            }
            .badge {
                display: inline-block;
                padding: 2px 8px;
                border-radius: 999px;
                font-size: 0.7rem;
                background: #111827;
                border: 1px solid #1f2937;
                margin-left: 6px;
                color: #9ca3af;
            }
            ul {
                padding-left: 1.1rem;
                margin: 0.35rem 0 0.5rem;
            }
            li {
                margin-bottom: 0.1rem;
            }
            a {
                color: #38bdf8;
            }
            .small {
                font-size: 0.78rem;
                color: #6b7280;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <header style="margin-bottom: 1.5rem;">
                <h1>Rate Case Agentic Orchestrator</h1>
                <div class="subtitle">
                    Multi-agent pipeline on top of your RAG backend: Decomposer → Retriever → Synthesizer → Validators.
                </div>
            </header>

            <div class="card">
                <div class="label">Intervenor Question</div>
                <textarea id="questionInput" placeholder="Ask something like: What is the justification for the proposed increase in ConEd's O&M expenses during the test year, and how does it compare with prior years' actual expenses?"></textarea>
                <button id="askBtn">
                    <span id="btnIcon">▶</span>
                    <span id="btnText">Run Agentic Pipeline</span>
                </button>
                <div class="small" style="margin-top: 6px;">
                    This will: (1) decompose the question, (2) call the retriever for each sub-question, (3) synthesize a final answer, and (4) run validator, business, and legal review agents.
                </div>
            </div>

            <div class="flex">
                <div class="flex-2">
                    <div class="card" id="finalAnswerCard" style="display:none;">
                        <div class="label">Final Synthesized Answer</div>
                        <div id="finalAnswer" style="white-space: pre-wrap; font-size: 0.9rem;"></div>
                    </div>

                    <div class="card" id="subqueriesCard" style="display:none;">
                        <div class="label">Decomposition & Sub-Results</div>
                        <div id="subqueriesContent"></div>
                    </div>
                </div>

                <div class="flex-1">
                    <div class="card" id="reviewsCard" style="display:none;">
                        <div class="label">Agent Reviews</div>
                        <div id="reviewsContent"></div>
                    </div>

                    <div class="card">
                        <div class="label">How to talk about this in demos</div>
                        <ul>
                            <li><b>Decomposition Agent</b> &mdash; splits complex questions into focused sub-questions.</li>
                            <li><b>Retriever Agents</b> &mdash; each sub-question calls the RAG retriever microservice.</li>
                            <li><b>Synthesis Agent</b> &mdash; merges sub-answers into a single narrative.</li>
                            <li><b>Validator / Business / Legal Agents</b> &mdash; apply layered checks before responding.</li>
                        </ul>
                        <div class="small">
                            The UI is just a thin demo layer. All orchestration runs in the backend orchestrator service.
                        </div>
                    </div>
                </div>
            </div>

            <div id="errorBox" style="display:none;" class="card">
                <div class="label">Error</div>
                <div id="errorText" class="muted"></div>
            </div>
        </div>

        <script>
            const btn = document.getElementById('askBtn');
            const questionInput = document.getElementById('questionInput');
            const finalAnswerCard = document.getElementById('finalAnswerCard');
            const finalAnswerEl = document.getElementById('finalAnswer');
            const subqueriesCard = document.getElementById('subqueriesCard');
            const subqueriesContent = document.getElementById('subqueriesContent');
            const reviewsCard = document.getElementById('reviewsCard');
            const reviewsContent = document.getElementById('reviewsContent');
            const errorBox = document.getElementById('errorBox');
            const errorText = document.getElementById('errorText');
            const btnIcon = document.getElementById('btnIcon');
            const btnText = document.getElementById('btnText');

            function setLoading(isLoading) {
                btn.disabled = isLoading;
                if (isLoading) {
                    btnIcon.textContent = '⏳';
                    btnText.textContent = 'Running agents...';
                } else {
                    btnIcon.textContent = '▶';
                    btnText.textContent = 'Run Agentic Pipeline';
                }
            }

            function statusPill(status, name) {
                let cls = 'pill';
                if (status === 'pass') cls += ' pill-success';
                else if (status === 'needs_review' || status === 'needs_changes') cls += ' pill-warn';
                else cls += ' pill-fail';

                return `<span class="${cls}"><span></span>${name}: ${status}</span>`;
            }

            async function runPipeline() {
                const question = questionInput.value.trim();
                errorBox.style.display = 'none';
                finalAnswerCard.style.display = 'none';
                subqueriesCard.style.display = 'none';
                reviewsCard.style.display = 'none';

                if (!question) {
                    errorText.textContent = 'Please enter a question.';
                    errorBox.style.display = 'block';
                    return;
                }

                try {
                    setLoading(true);
                    const resp = await fetch('/answer', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({ question })
                    });

                    if (!resp.ok) {
                        const txt = await resp.text();
                        throw new Error(`HTTP ${resp.status}: ${txt}`);
                    }

                    const data = await resp.json();

                    // Final answer
                    finalAnswerEl.textContent = data.final_answer || '(no final answer)';
                    finalAnswerCard.style.display = 'block';

                    // Subqueries & sub-results
                    const sqs = data.subquery_results || [];
                    let sqHtml = '';
                    sqs.forEach((r, idx) => {
                        sqHtml += `<div class="section">`;
                        sqHtml += `<div><b>Sub-question ${idx+1}:</b> ${r.subquery}</div>`;
                        sqHtml += `<div class="label" style="margin-top: 4px;">Retriever Answer</div>`;
                        sqHtml += `<div class="muted" style="white-space: pre-wrap; font-size: 0.85rem;">${(r.retriever_answer || '(no answer)').replace(/</g, '&lt;')}</div>`;
                        if (r.supporting_chunks && r.supporting_chunks.length) {
                            sqHtml += `<div class="label" style="margin-top: 4px;">Supporting Chunks</div>`;
                            sqHtml += `<div class="small">${r.supporting_chunks.join(', ')}</div>`;
                        }
                        sqHtml += `</div>`;
                    });
                    subqueriesContent.innerHTML = sqHtml || '<div class="muted">No sub-results.</div>';
                    subqueriesCard.style.display = 'block';

                    // Reviews
                    const revs = data.reviews || [];
                    let revHtml = '';
                    revs.forEach(r => {
                        revHtml += `<div class="section">`;
                        revHtml += statusPill(r.status, r.agent_name);
                        if (r.comments && r.comments.length) {
                            revHtml += `<ul>`;
                            r.comments.forEach(c => {
                                revHtml += `<li class="small">${c.replace(/</g, '&lt;')}</li>`;
                            });
                            revHtml += `</ul>`;
                        }
                        revHtml += `</div>`;
                    });
                    reviewsContent.innerHTML = revHtml || '<div class="muted">No reviews.</div>';
                    reviewsCard.style.display = 'block';

                } catch (err) {
                    console.error(err);
                    errorText.textContent = err.message || String(err);
                    errorBox.style.display = 'block';
                } finally {
                    setLoading(false);
                }
            }

            btn.addEventListener('click', runPipeline);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)
