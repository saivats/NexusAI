# NexusAI — One Front Door for Everything

An AI-powered campus assistant that routes queries to the right department, handles multiple topics in one message, asks to clarify when unsure, and escalates when it can't help. Built for **Microsoft Innovate 2026** (Bennett University), Problem Statement #18.

> **Offline-first:** works with zero internet, zero API keys. Optional Azure layer for answer polishing.

## Quick Start

```bash
pip install -r requirements.txt
uvicorn app.main:app --port 8000
```

Open [http://localhost:8000](http://localhost:8000) — the chat UI loads with a demo button.

## Evaluation Results (Held-Out Test Set)

| Metric | Value |
|---|---|
| Routing Accuracy | 96.4% |
| Resolution Rate | 90.9% |
| Macro-F1 | 0.990 |
| Sensitive Recall | 100% |
| Multi-topic Accuracy | 100% |
| Simple Query Accuracy | 100% |
| Handoff Accuracy | 87.5% |
| p50 / p95 Latency | 2.1ms / 3.0ms |

Run `python tests/eval.py` to reproduce on the held-out 40% split of 150 labelled queries.

## Architecture

```
User Query
  ↓
[Query Normalizer] → typo correction, synonyms, stemming
  ↓
[Sensitive Detector] → harassment, abuse, crisis → Welfare pathway
  ↓
[Multi-topic Splitter] → "X and also Y" → [Part 1] [Part 2]
  ↓
[Hybrid Router] → TF-IDF (35%) + Keywords (25%) + Calibrated SVM (40%)
  ↓
[Decision Logic]
  ├─ High confidence + margin → Answer (retrieve FAQ)
  ├─ Ambiguous → Clarify (show domain buttons)
  └─ Low confidence → Handoff (create ticket)
  ↓
[Domain Skill] → TF-IDF + BM25 hybrid retrieval → source-cited answer
  ↓
[Optional Azure] → rephrase with fact-check validation
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed component documentation with Mermaid diagrams.

## Features

### Routing
- **Hybrid scoring** blending 3 signals for robust domain detection
- Handles typos (`pasword`, `wfii`), Hinglish (`kab se class`), single-word queries (`wifi`, `exam`)
- Synonym expansion (wifi → wireless, internet, network, connection)
- Calibrated confidence scores (0-100%) with High/Medium/Low labels

### Multi-topic
- Splits compound queries: "reset password and also check fee deadline"
- Each part gets its own domain badge, confidence score, and source citation
- Supports `and`, `also`, `plus`, `;`, `?`, and numbered lists

### Clarify & Handoff
- When confidence is ambiguous, shows clickable domain buttons
- Session memory: pending query is re-answered when user clicks a domain
- Low-confidence → creates support ticket (TKT-XXXXXXXX) with contact info
- Every handoff appears in admin dashboard as a "knowledge gap"

### Sensitive Topics
- First-class detection: harassment, abuse, bullying, mental health crisis
- Supportive message with confidential contact (email + phone)
- Query text is **never stored** — only the category "sensitive" is logged

### Azure Integration (Optional)
- Flip `USE_AZURE_OPENAI=true` in `.env` to enable
- Rephrases answers in a warm, conversational tone
- Fact-check validation: URLs and numbers must match the source
- 4-second timeout, silent fallback on any error

### Admin Dashboard
- Routing distribution bar chart
- Top unanswered queries (knowledge gaps)
- Lowest-rated FAQs (from feedback)
- Suggested FAQ additions
- CSV export of all query logs
- Live KB reload without server restart

## Domains

| Domain | FAQs | Examples |
|---|---|---|
| IT Support | 15 | WiFi, password, VPN, printer, LMS |
| Human Resources | 15 | Leave, payslip, insurance, resignation |
| Finance & Fees | 15 | Tuition, refund, scholarship, payment plans |
| Facilities | 15 | Hostel, library, parking, gym, shuttle |
| Academics & Exams | 25 | Grades, exams, attendance, courses, transcripts |
| Admissions | 25 | Application, orientation, visa, clubs, counselling |

**Adding a new domain:** Create a JSON file in `data/`, add the domain name to `config.yaml`, hit `/admin/reload-kb`.

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/chat` | POST | Send a message, get routed response |
| `/feedback` | POST | Submit thumbs up/down for a FAQ |
| `/metrics` | GET | Live routing metrics |
| `/health` | GET | Health check with version |
| `/domains` | GET | List all domains with FAQ counts |
| `/history/{session_id}` | GET | Session query history |
| `/tickets` | GET | All support tickets |
| `/admin/reload-kb` | POST | Hot-reload knowledge base |
| `/admin/stats` | GET | Admin analytics |
| `/admin/export-csv` | GET | Download query logs as CSV |
| `/admin` | GET | Admin dashboard UI |

## Commands

```bash
pip install -r requirements.txt    # Install dependencies
uvicorn app.main:app --port 8000   # Start server
uvicorn app.main:app --reload      # Dev mode with auto-reload
python -m pytest tests/test_unit.py -v  # Run 39 unit tests
python tests/eval.py               # Run evaluation (held-out set)
python tests/tune.py               # Auto-tune thresholds (train set)
docker compose up --build          # Docker deployment
```

## Project Structure

```
NexusAI/
├── app/
│   ├── main.py          # FastAPI app, endpoints, session logic
│   ├── router.py         # Hybrid domain router (TF-IDF + SVM)
│   ├── retrieval.py      # TF-IDF + BM25 domain skills
│   ├── normalizer.py     # Typo correction, synonyms, stemming
│   ├── azure_layer.py    # Optional Azure OpenAI integration
│   ├── config.py         # YAML config loader with caching
│   └── database.py       # SQLite logging, tickets, feedback
├── data/
│   ├── it.json           # IT Support FAQs (15)
│   ├── hr.json           # HR FAQs (15)
│   ├── finance.json      # Finance FAQs (15)
│   ├── facilities.json   # Facilities FAQs (15)
│   ├── academics.json    # Academics FAQs (25)
│   └── admissions.json   # Admissions FAQs (25)
├── static/
│   ├── index.html        # Chat UI (dark/light theme, demo mode)
│   └── admin.html        # Admin dashboard
├── tests/
│   ├── queries.json      # 150 labelled test queries
│   ├── eval.py           # Evaluation script (held-out set)
│   ├── tune.py           # Threshold auto-tuner (train set)
│   └── test_unit.py      # 39 pytest unit tests
├── docs/
│   ├── ARCHITECTURE.md   # System design documentation
│   ├── DEMO_SCRIPT.md    # 5-minute demo walkthrough
│   └── PRESENTATION_NOTES.md  # Hackathon judging answers
├── config.yaml           # Thresholds, domains, settings
├── requirements.txt      # Pinned Python dependencies
├── Dockerfile            # Container build
├── docker-compose.yml    # One-command deployment
├── Makefile              # Build automation
└── .env.example          # Azure configuration template
```

## Design Decisions

1. **Why TF-IDF + SVM, not an LLM?** — Offline-first constraint. Our hybrid approach achieves 96.4% routing accuracy without any external dependency. When Azure is available, it only polishes answers — never routes or generates facts.

2. **Why 3 signals instead of 1?** — TF-IDF alone fails on single-word queries and typos. Keywords alone fail on paraphrases. The SVM classifier alone needs more training data. Blending all three covers each weakness.

3. **Why SQLite, not Postgres?** — Zero-setup deployment for a hackathon. WAL mode handles concurrent reads. Swap to Postgres by changing one connection string.

4. **Why not log sensitive queries?** — Safety over analytics. A student reporting harassment should never see their words in a CSV export or admin dashboard.

5. **Why a train/test split for evaluation?** — To avoid overfitting thresholds to the test set. Thresholds are tuned on the 60% train split; accuracy is measured on the 40% held-out split.

## Limitations

- FAQ content is **sample data** for demonstration — real deployment requires actual university FAQs
- No user authentication or role-based access control
- Session memory is in-memory (lost on restart) — for production, use Redis or SQLite sessions
- Single-word queries like "wifi" route correctly but trigger clarify (margin is small)
- Rate limiting is per-session, not per-IP

## License

MIT
