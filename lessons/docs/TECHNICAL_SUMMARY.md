# Technical Summary — Northstar Desk Triage Assistant

## Problem

Northstar Desk receives a steady mix of billing, access, integration, bug, and performance cases. Frontline agents must quickly decide **where to route** a case, **how urgent** it is, and **whether it may escalate** — often with only a short free-text summary at intake.

This prototype is a **decision-support tool**, not an auto-router. It suggests options and surfaces historical context so a human stays in control.

## Tracks covered

- **Track 1 — Triage assistant:** team, category, priority suggestions + escalation risk
- **Track 3 — Similarity retrieval:** top 3 similar past cases with outcomes

## Data preparation

**Source:** 12 quarterly CSV exports (`Q1-Jan` … `Q4-Dec`), one row per case snapshot.

**Steps (`data_loader.py`):**

1. **Merge** all files (~1,865 rows).
2. **Deduplicate** on `case_id`, keeping the latest `snapshot_at` → **1,730 unique cases**.
3. **Parse types:** timestamps, numeric timings, boolean escalation.
4. **Clean labels:**
   - Normalise `assigned_team` (`operations` → `support` to reflect org rename in later quarters).
   - Drop invalid `sentiment` values (`4`, `Urgent`).
5. **Derive fields:** SLA breach, slow resolution, low CSAT, and a combined `search_text` field (summary + category + subcategory + tags) for retrieval indexing.

**Known data quality issues handled:**

| Issue | Handling |
|---|---|
| Duplicate case IDs across quarterly files | Keep latest snapshot |
| Missing CSAT (~42%) | Shown as "n/a" in similar cases; not used as a hard filter |
| Missing resolution time (~25%, mostly open cases) | Excluded from timing analysis where needed |
| Team naming change over time | Alias map applied |

## Modelling approach

All models are **lightweight and interpretable** — suitable for a hackathon prototype and easy to explain to stakeholders.

### Routing classifiers (team, category, priority)

- **Features:** TF-IDF on text (unigrams + bigrams, English stop words, `min_df=2`).
- **Model:** Multinomial logistic regression (`scikit-learn`).
- **Output:** Top label + **confidence score** (max class probability).

### Escalation risk

- Same TF-IDF features.
- Logistic regression with **`class_weight=balanced`** (escalation rate ~13%).
- **Output:** Probability of escalation + keyword flags (`failed`, `urgent`, `crash`, etc.).

### Similar-case retrieval

- TF-IDF vectors over historical `search_text`.
- **Cosine similarity** against the query.
- Returns top 3 cases with route, status, resolution code, escalation flag, CSAT.

### Why these choices?

| Choice | Rationale |
|---|---|
| TF-IDF + logistic regression | Fast to train, no GPU, easy to debug, confidence scores built-in |
| Keyword risk flags | Transparent rules agents can understand and challenge |
| Similar-case retrieval | Directly answers "what happened last time?" without generative hallucination |
| Gradio UI | Minimal code for a shareable, interactive demo |

## Offline evaluation (5-fold CV)

Evaluated on **1,730 cases** with non-empty summaries, using **`case_summary` only** — the same input the Gradio app receives at inference time. Reproducible in `hackathon-notebook.ipynb` (cells 8–9).

| Task | Metric | Score |
|---|---|---|
| Team routing | Accuracy | **83.2%** (±2.3%) |
| Category routing | Accuracy | **85.5%** (±1.5%) |
| Priority | Accuracy | **60.7%** (±1.8%) |
| Escalation | ROC AUC | **87.5%** (±1.4%) |

**Priority is hardest** — labels overlap heavily (e.g. billing issues span Medium and High).

### Confidence distribution (summary-only)

When models are uncertain, agents should override the suggestion:

| Target | Median confidence | Cases below 50% confidence |
|---|---|---|
| assigned_team | 73% | 14.5% |
| category | 59% | 34.2% |
| priority | 57% | 30.8% |

Category and priority are often ambiguous from summary text alone — the UI shows confidence explicitly so agents know when to use judgement.

## Risks and limitations

1. **Training vs inference text mismatch (important):** In the live app, routing classifiers are still trained on enriched `search_text` (summary + category + subcategory + tags), but inference uses **case summary only**. Notebook evaluation (above) reports honest **summary-only** scores. A production version should retrain classifiers on summary-only text so training matches inference.

2. **Confidence ≠ certainty:** Low-confidence suggestions (40–50%) should be treated as weak hints, not decisions.

3. **No subcategory prediction yet:** 178 subcategories — too sparse for reliable multi-class classification in this prototype.

4. **Historical bias:** Models reflect past routing habits, including mistakes. Similar cases may include unresolved or poorly-rated outcomes.

5. **No live integration:** Standalone Gradio app; no write-back to a case management system.

6. **Fairness not audited:** Demographic fields (`age_band`, `gender`, `region_uk`) exist in the data but are **not used** in routing or risk scoring. A production rollout should include bias monitoring.

7. **Keyword flags are brittle:** Simple substring matching; can miss paraphrases or trigger on benign uses of words like "error".

## Interpretability choices

- Show **confidence percentages** for every routing suggestion.
- Surface **explicit keyword flags** alongside model-based escalation risk.
- Similar cases show **full outcome context** (status, resolution, CSAT) so agents can judge relevance themselves.
- No black-box LLM generation — all outputs trace to historical cases or linear model weights.

## How to reproduce

```bash
source /path/to/.venv/bin/activate
cd lessons
python app.py                    # Gradio prototype
jupyter notebook hackathon-notebook.ipynb   # EDA + evaluation + smoke tests
```

**Dependencies:** `pandas`, `numpy`, `scikit-learn`, `gradio` (see `requirements.txt`).
