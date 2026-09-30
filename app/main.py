import re
import uuid
from collections import defaultdict

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.router import get_router
from app.retrieval import get_retriever

app = FastAPI(title="NexusAI", version="1.0.0")

sessions = {}
metrics = defaultdict(int)

SPLIT_PATTERN = re.compile(r"\band\b|,|\balso\b|\?")
CONTACT_EMAIL = "support@university.edu"
CONTACT_PHONE = "ext. 1234"


class ChatRequest(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    message: str


class PartResponse(BaseModel):
    domain: str
    confidence: float
    action: str
    answer: str = ""
    source: str = ""
    options: list[str] = []


class ChatResponse(BaseModel):
    session_id: str
    parts: list[PartResponse]


class MetricsResponse(BaseModel):
    answered: int
    clarified: int
    handed_off: int


def _split_multi_topic(message):
    parts = SPLIT_PATTERN.split(message)
    cleaned = [p.strip() for p in parts if p.strip() and len(p.strip()) > 3]
    return cleaned if cleaned else [message.strip()]


def _generate_ticket_id():
    return f"TKT-{uuid.uuid4().hex[:8].upper()}"


@app.on_event("startup")
def startup():
    get_router()
    get_retriever()


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    router = get_router()
    retriever = get_retriever()

    session = sessions.setdefault(request.session_id, {
        "pending_query": None,
        "pending_domains": None,
    })

    if session["pending_query"] and request.message.lower() in [d.lower() for d in (session["pending_domains"] or [])]:
        chosen_domain = request.message.lower()
        original_query = session["pending_query"]
        result = retriever.retrieve(original_query, chosen_domain)
        session["pending_query"] = None
        session["pending_domains"] = None

        if result:
            metrics["answered"] += 1
            return ChatResponse(
                session_id=request.session_id,
                parts=[PartResponse(
                    domain=chosen_domain,
                    confidence=1.0,
                    action="answer",
                    answer=result["answer"],
                    source=result["source"],
                )],
            )

    message_parts = _split_multi_topic(request.message)
    response_parts = []

    for part in message_parts:
        route_result = router.route(part)
        action = route_result["action"]

        if action == "answer":
            result = retriever.retrieve(part, route_result["top_domain"])
            metrics["answered"] += 1
            response_parts.append(PartResponse(
                domain=route_result["top_domain"],
                confidence=route_result["confidence"],
                action="answer",
                answer=result["answer"] if result else "No matching FAQ found.",
                source=result["source"] if result else "",
            ))

        elif action == "clarify":
            top = route_result["top_domain"]
            second = route_result["second_domain"]
            options = [top, second]
            session["pending_query"] = part
            session["pending_domains"] = options
            metrics["clarified"] += 1
            response_parts.append(PartResponse(
                domain=top,
                confidence=route_result["confidence"],
                action="clarify",
                answer=f"I found potential matches in multiple areas. Could you clarify which department can help you best?",
                options=options,
            ))

        elif action == "handoff":
            ticket_id = _generate_ticket_id()
            metrics["handed_off"] += 1
            response_parts.append(PartResponse(
                domain="unknown",
                confidence=route_result["confidence"],
                action="handoff",
                answer=f"I wasn't able to find a confident match for your query. A support ticket has been created.\n\nTicket ID: {ticket_id}\nContact: {CONTACT_EMAIL} | {CONTACT_PHONE}\n\nA human agent will follow up within 24 hours.",
            ))

    return ChatResponse(
        session_id=request.session_id,
        parts=response_parts,
    )


@app.get("/metrics", response_model=MetricsResponse)
def get_metrics():
    return MetricsResponse(
        answered=metrics["answered"],
        clarified=metrics["clarified"],
        handed_off=metrics["handed_off"],
    )


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")
