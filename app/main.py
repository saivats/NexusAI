import re
import csv
import io
import uuid
import time
import logging
from collections import defaultdict

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from app.config import load_config, reload_config
from app.router import get_router
from app.retrieval import get_retriever
from app.azure_layer import is_azure_enabled, rephrase_answer
from app import database as db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("nexusai")

app = FastAPI(title="NexusAI", version="2.0.0", description="One Front Door for Everything")

config = load_config()
server_config = config.get("server", {})
app.add_middleware(
    CORSMiddleware,
    allow_origins=server_config.get("cors_origins", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions = {}
rate_limits = defaultdict(list)

SPLIT_PATTERN = re.compile(
    r"""
    \band\s+also\b |
    \balso\b |
    \bplus\b |
    \band\b |
    [;] |
    [,]\s*(?:and\b|also\b|plus\b|\d+[.)]) |
    [,](?=\s+(?:how|what|when|where|why|can|do|is|are|i\s))|
    \?\s* |
    (?:^|\n)\s*\d+[.)]\s*
    """,
    re.VERBOSE | re.IGNORECASE,
)
CONTACT_EMAIL = "support@university.edu"
CONTACT_PHONE = "ext. 1234"
MAX_MESSAGE_LENGTH = 500
RATE_LIMIT_PER_MINUTE = 30
SESSION_MAX_TURNS = 5

DOMAIN_DISPLAY = {
    "it": "IT Support",
    "hr": "Human Resources",
    "finance": "Finance & Fees",
    "facilities": "Facilities",
    "academics": "Academics & Exams",
    "admissions": "Admissions & Student Services",
    "sensitive": "Welfare & Support",
    "unknown": "Unknown",
}


class ChatRequest(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    message: str

    @field_validator("message")
    @classmethod
    def validate_message(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Message cannot be empty")
        if len(v) > MAX_MESSAGE_LENGTH:
            raise ValueError(f"Message exceeds {MAX_MESSAGE_LENGTH} character limit")
        return v


class FeedbackRequest(BaseModel):
    session_id: str
    faq_id: str
    rating: int = Field(ge=-1, le=1)


class PartResponse(BaseModel):
    domain: str
    domain_label: str = ""
    confidence: float
    calibrated_confidence: float = 0.0
    confidence_label: str = ""
    action: str
    answer: str = ""
    source: str = ""
    source_title: str = ""
    options: list[str] = []
    option_labels: list[str] = []
    alternatives: list[dict] = []
    follow_up: str | None = None
    ticket_id: str | None = None
    sensitive_contact: dict | None = None


class ChatResponse(BaseModel):
    session_id: str
    parts: list[PartResponse]
    azure_mode: bool = False
    request_id: str = ""


class MetricsResponse(BaseModel):
    answered: int = 0
    clarified: int = 0
    handed_off: int = 0
    sensitive: int = 0
    total: int = 0
    avg_confidence: float = 0.0
    avg_latency_ms: float = 0.0
    domain_distribution: dict = {}
    azure_enabled: bool = False


def _split_multi_topic(message):
    parts = SPLIT_PATTERN.split(message)
    cleaned = [p.strip() for p in parts if p and p.strip() and len(p.strip()) > 3]
    return cleaned if cleaned else [message.strip()]


def _generate_ticket_id():
    return f"TKT-{uuid.uuid4().hex[:8].upper()}"


def _sanitize_output(text):
    text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.replace("<", "&lt;").replace(">", "&gt;")
    return text


def _check_rate_limit(session_id):
    now = time.time()
    window = [t for t in rate_limits[session_id] if now - t < 60]
    rate_limits[session_id] = window
    if len(window) >= RATE_LIMIT_PER_MINUTE:
        return False
    rate_limits[session_id].append(now)
    return True


def _get_session(session_id):
    return sessions.setdefault(session_id, {
        "pending_query": None,
        "pending_domains": None,
        "history": [],
        "last_domain": None,
    })


def _update_session_history(session, domain, query):
    session["history"].append({"domain": domain, "query": query})
    if len(session["history"]) > SESSION_MAX_TURNS:
        session["history"] = session["history"][-SESSION_MAX_TURNS:]
    session["last_domain"] = domain


def _resolve_context(query, session):
    context_words = ["that", "this", "it", "those", "them", "same", "there"]
    query_lower = query.lower().strip()
    words = query_lower.split()

    if len(words) <= 5 and any(w in context_words for w in words):
        if session.get("last_domain"):
            return session["last_domain"]
    return None


@app.on_event("startup")
def startup():
    db.init_db()
    get_router()
    get_retriever()
    logger.info("NexusAI started — Azure mode: %s", is_azure_enabled())


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())[:12]
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "version": "2.0.0",
        "azure_enabled": is_azure_enabled(),
    }


@app.get("/domains")
def get_domains():
    router = get_router()
    return {
        "domains": [
            {"id": d, "label": DOMAIN_DISPLAY.get(d, d), "faq_count": len(router.domain_data.get(d, []))}
            for d in router.domain_names
        ]
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, req: Request):
    request_id = getattr(req.state, "request_id", str(uuid.uuid4())[:12])
    start_time = time.time()

    if not _check_rate_limit(request.session_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please wait a moment.")

    router = get_router()
    retriever = get_retriever()
    azure_active = is_azure_enabled()

    session = _get_session(request.session_id)

    if (session["pending_query"]
            and request.message.lower().strip() in [d.lower() for d in (session["pending_domains"] or [])]):
        chosen_domain = request.message.lower().strip()
        original_query = session["pending_query"]
        result = retriever.retrieve(original_query, chosen_domain)
        session["pending_query"] = None
        session["pending_domains"] = None

        if result:
            answer_text = result["answer"]
            if azure_active:
                answer_text, _ = rephrase_answer(answer_text, result["source"])

            latency_ms = (time.time() - start_time) * 1000
            _update_session_history(session, chosen_domain, original_query)
            db.log_query(request_id, request.session_id, original_query, chosen_domain,
                         1.0, 1.0, "answer", result["faq_id"], latency_ms)

            return ChatResponse(
                session_id=request.session_id,
                request_id=request_id,
                azure_mode=azure_active,
                parts=[PartResponse(
                    domain=chosen_domain,
                    domain_label=DOMAIN_DISPLAY.get(chosen_domain, chosen_domain),
                    confidence=1.0,
                    calibrated_confidence=1.0,
                    confidence_label="High",
                    action="answer",
                    answer=_sanitize_output(answer_text),
                    source=result["source"],
                    source_title=result.get("source_title", ""),
                    alternatives=result.get("alternatives", []),
                    follow_up=result.get("follow_up"),
                )],
            )

    message_parts = _split_multi_topic(request.message)
    response_parts = []

    for part_text in message_parts:
        part_start = time.time()

        context_domain = _resolve_context(part_text, session)

        route_result = router.route(part_text)
        action = route_result["action"]

        if context_domain and action in ("handoff", "clarify"):
            route_result["top_domain"] = context_domain
            route_result["action"] = "answer"
            action = "answer"

        calibrated = route_result.get("calibrated_confidence", 0.0)
        label = route_result.get("confidence_label", "Low")

        if action == "sensitive":
            sensitive_resp = route_result.get("sensitive_response", {})
            latency_ms = (time.time() - part_start) * 1000
            db.log_sensitive_query(request_id, request.session_id)

            response_parts.append(PartResponse(
                domain="sensitive",
                domain_label=DOMAIN_DISPLAY.get("sensitive", "Welfare & Support"),
                confidence=1.0,
                calibrated_confidence=1.0,
                confidence_label="High",
                action="sensitive",
                answer=sensitive_resp.get("message", ""),
                sensitive_contact={
                    "email": sensitive_resp.get("email", ""),
                    "phone": sensitive_resp.get("phone", ""),
                },
            ))

        elif action == "answer":
            domain = route_result["top_domain"]
            result = retriever.retrieve(part_text, domain)
            latency_ms = (time.time() - part_start) * 1000

            answer_text = result["answer"] if result else "No matching FAQ found for this query."
            source = result["source"] if result else ""
            faq_id = result["faq_id"] if result else None

            if azure_active and result:
                answer_text, _ = rephrase_answer(answer_text, source)

            _update_session_history(session, domain, part_text)
            db.log_query(request_id, request.session_id, part_text, domain,
                         route_result["confidence"], calibrated, "answer", faq_id, latency_ms)

            response_parts.append(PartResponse(
                domain=domain,
                domain_label=DOMAIN_DISPLAY.get(domain, domain),
                confidence=route_result["confidence"],
                calibrated_confidence=calibrated,
                confidence_label=label,
                action="answer",
                answer=_sanitize_output(answer_text),
                source=source,
                source_title=result.get("source_title", "") if result else "",
                alternatives=result.get("alternatives", []) if result else [],
                follow_up=result.get("follow_up") if result else None,
            ))

        elif action == "clarify":
            top = route_result["top_domain"]
            second = route_result["second_domain"]
            options = [top, second]
            session["pending_query"] = part_text
            session["pending_domains"] = options
            latency_ms = (time.time() - part_start) * 1000

            db.log_query(request_id, request.session_id, part_text, top,
                         route_result["confidence"], calibrated, "clarify", None, latency_ms)

            response_parts.append(PartResponse(
                domain=top,
                domain_label=DOMAIN_DISPLAY.get(top, top),
                confidence=route_result["confidence"],
                calibrated_confidence=calibrated,
                confidence_label=label,
                action="clarify",
                answer="I found potential matches in multiple areas. Could you clarify which department can help you best?",
                options=options,
                option_labels=[DOMAIN_DISPLAY.get(o, o) for o in options],
            ))

        elif action == "handoff":
            ticket_id = _generate_ticket_id()
            latency_ms = (time.time() - part_start) * 1000
            db.create_ticket(ticket_id, request.session_id, part_text,
                             route_result.get("top_domain", "unknown"))
            db.log_query(request_id, request.session_id, part_text, "unknown",
                         route_result["confidence"], calibrated, "handoff", None, latency_ms)

            response_parts.append(PartResponse(
                domain="unknown",
                domain_label=DOMAIN_DISPLAY.get("unknown", "Unknown"),
                confidence=route_result["confidence"],
                calibrated_confidence=calibrated,
                confidence_label=label,
                action="handoff",
                answer=(
                    f"I wasn't able to find a confident match for your query. "
                    f"A support ticket has been created.\n\n"
                    f"Ticket ID: {ticket_id}\n"
                    f"Contact: {CONTACT_EMAIL} | {CONTACT_PHONE}\n\n"
                    f"A human agent will follow up within 24 hours."
                ),
                ticket_id=ticket_id,
            ))

    return ChatResponse(
        session_id=request.session_id,
        parts=response_parts,
        azure_mode=azure_active,
        request_id=request_id,
    )


@app.post("/feedback")
def submit_feedback(request: FeedbackRequest):
    db.add_feedback(request.session_id, request.faq_id, request.rating)
    return {"status": "ok", "message": "Thank you for your feedback"}


@app.get("/metrics", response_model=MetricsResponse)
def get_metrics():
    metrics = db.get_metrics()
    distribution = db.get_domain_distribution()
    return MetricsResponse(
        answered=metrics.get("answered", 0) or 0,
        clarified=metrics.get("clarified", 0) or 0,
        handed_off=metrics.get("handed_off", 0) or 0,
        sensitive=metrics.get("sensitive", 0) or 0,
        total=metrics.get("total", 0) or 0,
        avg_confidence=round(metrics.get("avg_confidence", 0) or 0, 3),
        avg_latency_ms=round(metrics.get("avg_latency_ms", 0) or 0, 1),
        domain_distribution=distribution,
        azure_enabled=is_azure_enabled(),
    )


@app.get("/history/{session_id}")
def get_history(session_id: str):
    history = db.get_history(session_id)
    return {"session_id": session_id, "history": history}


@app.get("/tickets")
def get_tickets():
    tickets = db.get_tickets()
    return {"tickets": tickets}


@app.post("/admin/reload-kb")
def reload_knowledge_base():
    reload_config()
    get_router(force_reload=True)
    get_retriever(force_reload=True)
    logger.info("Knowledge base reloaded")
    return {"status": "ok", "message": "Knowledge base reloaded successfully"}


@app.get("/admin/stats")
def admin_stats():
    metrics = db.get_metrics()
    distribution = db.get_domain_distribution()
    handoff_queries = db.get_handoff_queries()
    lowest_rated = db.get_lowest_rated_faqs()
    return {
        "metrics": metrics,
        "domain_distribution": distribution,
        "handoff_queries": handoff_queries,
        "lowest_rated_faqs": lowest_rated,
    }


@app.get("/admin/export-csv")
def export_csv():
    logs = db.get_all_logs()
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=[
        "id", "request_id", "session_id", "query", "domain",
        "confidence", "calibrated_confidence", "action", "faq_id",
        "latency_ms", "resolved", "timestamp",
    ])
    writer.writeheader()
    for log in logs:
        writer.writerow(log)
    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=nexusai_logs.csv"},
    )


@app.get("/admin", response_class=HTMLResponse)
def admin_page():
    return FileResponse("static/admin.html")


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")
