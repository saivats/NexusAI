# NexusAI — One Front Door for Everything

A campus assistant demo that routes student and staff queries to the right department (IT, HR, Finance, Facilities) using TF-IDF cosine similarity. Runs 100% offline with no API keys required.

## Quick Start

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

## Architecture

```
User Query → Multi-topic Splitter → Domain Router (TF-IDF + Keywords)
                                         ↓
                              ┌──────────┼──────────┐
                              ▼          ▼          ▼
                           Answer     Clarify    Handoff
                              │          │          │
                              ▼          │          ▼
                        FAQ Retrieval    │     Ticket + Contact
                              │          │
                              ▼          ▼
                         Source-cited  Domain buttons
                          response    for re-routing
```

## Routing Logic

| Confidence | Action | Behaviour |
|---|---|---|
| ≥ 0.35 (clear winner) | `answer` | Retrieve best FAQ, cite source |
| 0.15–0.35, or top-2 within 0.08 | `clarify` | Show buttons for top 2 domains |
| < 0.15 | `handoff` | Generate ticket ID + contact info |

## API

- **POST /chat** `{session_id, message}` → `{parts: [{domain, confidence, action, answer, source, options}]}`
- **GET /metrics** → `{answered, clarified, handed_off}`

## Evaluation

```bash
python tests/eval.py
```

Prints routing accuracy, resolution rate, handoff rate, and a per-domain confusion matrix.

## Project Structure

```
app/
  main.py          FastAPI application, endpoints, session management
  router.py        TF-IDF domain router with keyword boosting
  retrieval.py     Per-domain FAQ retrieval with source attribution
data/
  it.json          15 IT support FAQs
  hr.json          15 HR FAQs
  finance.json     15 Finance/Fees FAQs
  facilities.json  15 Facilities FAQs
static/
  index.html       Single-page chat UI
tests/
  eval.py          Evaluation script
  queries.json     40 labelled test queries
```

## Tech Stack

- Python 3.11, FastAPI, scikit-learn (TF-IDF), vanilla HTML/CSS/JS
- No database, no external APIs, no API keys
