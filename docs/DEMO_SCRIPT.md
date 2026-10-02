# 🎬 NexusAI Demo Script (5 Minutes)

A scripted 5-minute demo flow for the Microsoft Innovate 2026 hackathon presentation.

> **Tip:** You can use the **"▶ Demo Mode"** button in the UI to auto-play steps 1-4 without typing.

---

## Setup (30 seconds)

```bash
pip install -r requirements.txt
uvicorn app.main:app --port 8000
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

---

## Step 1: IT Query with Informal Phrasing (1 minute)

**Type:** `hey why is wifi not working`

**What to show:**
- Routes to **IT Support** with **High confidence**
- Returns WiFi FAQ (IT-003) with source citation
- Confidence bar shows calibrated percentage + label
- Source chip shows "IT FAQ #IT-003, updated 2026-08"
- Point out: this used to route at 7% confidence — now fixed with synonym coverage

---

## Step 2: Multi-Topic Query (1 minute)

**Type:** `I need help with my password and also when is the fee deadline`

**What to show:**
- Response splits into **Part 1** (IT: password reset) and **Part 2** (Finance: fee deadline)
- Each part has its own domain badge, confidence, and source
- Demonstrate multi-topic splitting with labelled blocks
- Live metrics update in the side panel

---

## Step 3: Ambiguous Query → Clarify (45 seconds)

**Type:** `I need money help for school`

**What to show:**
- System detects ambiguity between Finance and Admissions
- Shows **clarify** action with clickable domain buttons
- Click "Finance & Fees" → re-answers with the original query in the Finance domain
- Session memory remembers the pending query

---

## Step 4: Sensitive Topic (45 seconds)

**Type:** `I am being harassed by a senior student`

**What to show:**
- Routes to **Welfare & Support** — NOT a generic handoff
- Shows supportive message with confidential contact (welfare@university.edu, ext. 1100)
- Query text is NOT logged — only the category "sensitive" is recorded
- Emphasise: safety first, no generic responses for serious issues

---

## Step 5: Out-of-Scope → Handoff (30 seconds)

**Type:** `What is the meaning of life?`

**What to show:**
- Confidence is **Low**, triggers handoff
- Generates a support ticket (TKT-XXXXXXXX) with contact info
- Ticket appears in the admin dashboard

---

## Step 6: Admin Dashboard (30 seconds)

Open [http://localhost:8000/admin](http://localhost:8000/admin)

**What to show:**
- Routing distribution bar chart
- Tickets list with status
- Top unanswered queries (knowledge gaps)
- "Suggested FAQ Additions" section
- CSV export button
- "Reload KB" button — add a FAQ to a JSON file and reload without restarting

---

## Step 7: Evaluation Results (30 seconds)

```bash
python tests/eval.py
```

**What to show:**
- Routing accuracy, resolution rate, macro-F1
- Per-domain F1 breakdown
- Confusion matrix
- p50/p95 latency numbers
- Emphasise: these are held-out test set numbers, not tuned-set numbers

---

## Key Talking Points

1. **One assistant, one front door** — no need for separate bots per department
2. **Hybrid scoring** — TF-IDF + keywords + classifier blended for robust routing
3. **Sensitive topic safety** — dedicated path, never generic, never logged
4. **Measurable** — real metrics on held-out data, not cherry-picked
5. **Offline-first** — works with zero internet, zero API keys
6. **Azure-ready** — flip a flag, get AI-rephrased answers with fact-checking
7. **Scalable** — add a JSON file + reload, new domain online in seconds
