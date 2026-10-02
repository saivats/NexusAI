# Presentation Notes — Microsoft Innovate 2026 (Problem Statement #18)

## Judging Parameter Answers

### 1. Problem Understanding

**Q: What problem are you solving?**

Universities have siloed support systems — separate portals for IT, HR, Finance, Facilities, Academics, and Admissions. Students and staff waste time figuring out which department to contact. We built **NexusAI: One Front Door for Everything** — a single intelligent assistant that understands the topic, routes to the right knowledge base, and answers with source-cited FAQs. It handles multiple topics in one message, asks for clarification when unsure, and escalates when it can't help. Sensitive topics like harassment get a dedicated safe pathway, never a generic response.

### 2. Solution and Innovation

**Q: What's innovative about your approach?**

- **Hybrid routing** that blends three signals (TF-IDF cosine, keyword matching, calibrated SVM classifier) for robust domain detection — works even with typos, Hinglish phrasing, and single-word queries
- **Sensitive topic detection** as a first-class feature — not an afterthought. Queries about harassment, abuse, or mental health get immediate supportive responses with confidential contacts, and the query text is never stored
- **Multi-topic splitting** — one message can contain questions for different departments, each answered separately
- **Confidence calibration** — raw TF-IDF scores are mapped to meaningful 0-100% scale with High/Medium/Low labels, enabling transparent decision-making
- **Offline-first architecture** with optional Azure OpenAI for answer polishing — flipping a flag upgrades the experience without changing the architecture

### 3. Technical Feasibility

**Q: Can this actually work in production?**

- Runs 100% offline with no API keys, no internet, no external services
- Built with production-ready stack: Python 3.11, FastAPI, scikit-learn
- Single `docker compose up` deploys the entire system
- SQLite for logging — zero database setup
- Rate limiting, input validation, output sanitization, structured logging
- Response latency under 100ms (p95) for offline mode
- Knowledge base hot-reload without server restart
- Comprehensive test suite with 150 labelled queries

### 4. Impact and Scalability

**Q: How does this scale and what's the impact?**

- **Adding a new domain** = create a JSON file + reload. No code changes, no retraining
- **150 FAQ entries** across 6 domains currently; scales linearly with TF-IDF/BM25
- **Admin dashboard** shows routing distribution, knowledge gaps, lowest-rated FAQs, and suggests which FAQs to add next
- **Feedback loop** — thumbs up/down on every answer drives continuous improvement
- **Ticket system** captures what the bot can't answer, closing the loop with human support
- **CSV export** for institutional reporting
- For a university with 10,000+ students, this replaces multiple help desks for first-level support

### 5. Presentation and Team Readiness

**Q: How ready are you to present and maintain this?**

- **Demo-stable** — scripted demo mode plays 4 queries automatically, typo-proof
- **Live metrics** — the side panel shows real-time answered/clarified/handoff counts
- **Architecture documented** with Mermaid diagrams
- **Held-out evaluation** with honest numbers — we don't claim accuracy we didn't measure
- **Every team member** can explain their part — the codebase is modular (router, retrieval, normalizer, azure layer, database — each a separate file)
- **README** has 3-command setup, evaluation table, design decisions, and limitations

---

## Slide Structure Suggestion

| Slide | Content | Duration |
|---|---|---|
| 1 | Problem: siloed university support | 30s |
| 2 | Solution: one intelligent front door | 30s |
| 3 | Architecture diagram | 45s |
| 4 | Live demo (4-step scripted) | 2min |
| 5 | Evaluation results table | 30s |
| 6 | Impact + scalability story | 30s |
| 7 | Q&A | remaining |

---

## FAQ for Judges

**Q: Why not use a large language model directly?**
A: We're offline-first by design. An LLM requires internet and API keys, which may not be available during demos or in constrained deployments. Our hybrid TF-IDF+SVM approach achieves 90%+ routing accuracy without any external dependency. When Azure is available, we use it only for answer polishing — never for routing or fact generation.

**Q: How do you handle queries you don't know?**
A: Three-tier decision: answer (high confidence + clear winner), clarify (ambiguous → show buttons), handoff (low confidence → create ticket). Every handoff is logged and shows up in the admin dashboard as a "knowledge gap" — the system tells you what it doesn't know.

**Q: What about privacy?**
A: Sensitive queries (harassment, abuse, mental health) are detected first and routed to a dedicated welfare pathway. The query text is NEVER stored — only the category "sensitive" is logged. Normal queries are logged for analytics but can be exported and purged.

**Q: Can this work for a real university?**
A: The architecture is production-ready. The FAQ content is sample data (labeled as such in the README). To deploy at a real university, replace the JSON files with actual FAQs, tune the thresholds, and optionally connect Azure for polished responses.
