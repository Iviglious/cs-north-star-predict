# Northstar Desk Triage Assistant
### AI Engineer Hackathon — Presentation (`uma` branch)

**Team repo:** [Iviglious/cs-north-star-predict](https://github.com/Iviglious/cs-north-star-predict)  
**Branch:** `uma`  
**Demo:** `cd lessons && python app.py` → http://127.0.0.1:7860

---

## Slide 1 — Title

**Northstar Desk Triage Assistant**  
Decision-support for frontline support agents

- Cambridge Spark AI Engineer Hackathon
- Tracks: **Triage assistant** + **Similar-case retrieval**
- Umapathy Desineni · `dumpathy@gmail.com`

---

## Slide 2 — The problem

Northstar Desk handles billing, access, bugs, integrations, and performance cases.

When a new case arrives, agents must quickly answer:

1. **Who should own this?** (billing, engineering, support…)
2. **How urgent is it?**
3. **Have we seen this before?** What fixed it last time?

Today: memory, Slack, asking a colleague — **slow and inconsistent**, especially for new starters.

> Northstar does **not** want AI that replaces staff. They want a tool that helps humans **decide faster**.

---

## Slide 3 — Our solution

**Paste a case summary → get suggestions in seconds**

| Output | What the agent sees |
|---|---|
| Routing suggestion | Team, category, priority + **confidence %** |
| Risk signals | Escalation probability + keyword flags |
| Similar cases | Top 3 past cases with outcomes (status, resolution, CSAT) |

**Human stays in control** — suggestions, not auto-routing.

---

## Slide 4 — Live demo flow

1. Open Gradio app (`python app.py`)
2. Paste case summary (or click an example)
3. Optional: channel + plan tier
4. Click **Suggest routing**

**Example to demo:**
> *"CSV export is coming out blank for the sales report widget."*

**Expected output:**
- Route: support / data_reporting / Medium
- Flag: `blank`
- Similar case ND-2025-002008 — same issue, escalated to engineering

---

## Slide 5 — Who uses it?

**Primary user:** Frontline support agent (email, webchat, phone, in-app)

**Secondary user:** Team lead — spot-checks routing, uses similar cases in coaching

**30-second pitch:**
> *"Paste the summary, get routing, risk, and three similar cases in seconds. You stay in control — the tool helps you decide with evidence."*

---

## Slide 6 — User story (Sarah)

**Scenario A — Export bug**
- Sarah pastes CSV export issue → tool suggests engineering, flags `blank`
- Similar case shows prior escalation → she adds note referencing case ID
- **Saves 3–5 minutes** of searching

**Scenario B — Failed payment**
- Urgent billing case → 39% escalation risk, flags `failed`, `urgency_language`
- Similar case solved in 3.8h as `fixed` → used as playbook

**Scenario C — Override**
- VAT receipt → billing / Medium at high confidence → Sarah confirms
- Engineering at **42% confidence** → weak hint, she uses judgement

---

## Slide 7 — Data

**Source:** 12 monthly CSV exports (Jan–Dec 2025)

| Step | Result |
|---|---|
| Merge files | ~1,865 rows |
| Dedupe on `case_id` | **1,730 unique cases** |
| Clean teams | `operations` → `support` |
| Derive fields | SLA breach, search text for retrieval |

**Handled:** duplicate IDs, missing CSAT (42%), team rename over time

---

## Slide 8 — Models we use

All **scikit-learn** — lightweight, interpretable, no GPU.

| Component | Method |
|---|---|
| Team / category / priority | **TF-IDF + Logistic Regression** |
| Escalation risk | **TF-IDF + Logistic Regression** (balanced classes) |
| Similar cases | **TF-IDF + Cosine similarity** (top 3) |
| Risk flags | **Keyword rules** (transparent, not black-box) |

**Not used:** random forests, neural nets, LLMs

---

## Slide 9 — Why these models?

| Choice | Why |
|---|---|
| TF-IDF + logistic regression | Fast, confidence scores built-in, easy to explain |
| Cosine similarity retrieval | "What worked before?" — grounded in real cases, no hallucination |
| Keyword flags | Agents can see and challenge rules |
| Gradio UI | Shareable demo in ~1 day |

Trains on **~1,700 cases** in seconds at app startup — no saved model file needed.

---

## Slide 10 — Evaluation results

**5-fold cross-validation** on `case_summary` only (matches app input)

| Task | Score |
|---|---|
| Team routing | **83.2%** ± 2.3% |
| Category routing | **85.5%** ± 1.5% |
| Priority | **60.7%** ± 1.8% |
| Escalation | **87.5%** ROC AUC ± 1.4% |

**Priority is hardest** — overlapping labels (Medium vs High).

Reproducible in `hackathon-notebook.ipynb` Block 6.

---

## Slide 11 — Confidence & trust

We show confidence on every suggestion — agents know when to override.

| Target | Median confidence | Below 50% (weak hint) |
|---|---|---|
| Team | 73% | 14.5% |
| Category | 59% | 34.2% |
| Priority | 57% | 30.8% |

**Decision-support, not automation** — low confidence = use judgement.

---

## Slide 12 — Limitations (honest)

1. Classifiers trained on enriched text; inference uses summary only — **retrain for production**
2. No subcategory prediction (178 classes — too sparse)
3. Similar cases may include **unresolved** outcomes — agent judges relevance
4. Keyword flags are brittle (substring matching)
5. Standalone app — no CMS integration yet
6. Demographics **not used** in routing; fairness audit needed for production

---

## Slide 13 — Project structure

```
cs-north-star-predict/  (branch: uma)
├── requirements.txt
├── README.md
└── lessons/
    ├── app.py                 ← Gradio demo
    ├── data_loader.py         ← data pipeline
    ├── triage_engine.py       ← models
    ├── hackathon-notebook.ipynb
    ├── docs/                  ← this presentation
    └── data/                  ← 12 CSV files
```

---

## Slide 14 — Roadmap (next week)

**Priority 1**
- Retrain on summary-only text (fix train/inference mismatch)
- Accept / Override logging for feedback loop
- Suggested next steps from similar cases ("3/3 resolved with `fixed`")

**Priority 2**
- Time-based evaluation (train Q1–Q3, test Q4)
- Ops dashboard tab (volume trends, escalation spikes)
- CMS sidebar integration

**Month 2+**
- Sentence embeddings for better paraphrase matching
- Closed-loop retraining on agent overrides
- Fairness audit + PII redaction

---

## Slide 15 — Pilot success criteria

| KPI | Target |
|---|---|
| Agent adoption | >70% use on new cases |
| Override rate | <30% |
| Time to route | −20% |
| Escalation surprises | −15% |
| Agent satisfaction | ≥4/5 |

---

## Slide 16 — Q&A / backup

**Run locally**
```bash
pip install -r requirements.txt
cd lessons && python app.py
```

**Notebook**
```bash
jupyter notebook hackathon-notebook.ipynb
```

**Branches in repo**
- `main` — starter data
- `Peter` — structured-field ML pipeline (teammate)
- `uma` — our Gradio triage assistant (this submission)

**Thank you — questions?**
