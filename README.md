# cs-north-star-predict

AI Hackathon project — **Northstar Desk Triage Assistant** (Cambridge Spark apprenticeship).

Decision-support tool for frontline support agents: paste a case summary, get routing suggestions,
escalation risk flags, and similar past cases.

## Project structure

```
lessons/
├── app.py                  # Gradio UI — run this for the demo
├── data_loader.py          # Load, dedupe, and clean 12 quarterly CSV exports
├── triage_engine.py        # TF-IDF models + similar-case retrieval
├── hackathon-notebook.ipynb # EDA, evaluation, smoke tests (with block-by-block notes)
├── brief.txt               # Hackathon challenge specification
├── docs/                   # Presentation: technical summary, user story, roadmap
└── data/                   # Case exports (Q1-Jan … Q4-Dec)
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd lessons
python app.py
```

Open http://127.0.0.1:7860

## Branches

| Branch | Description |
|---|---|
| `main` | Original starter data |
| `Peter` | Teammate ML pipeline (`model_utils.py`) — structured-field routing |
| `uma` | Triage assistant + Gradio app + docs (this submission) |

## Tracks covered

- **Track 1** — Triage assistant (team, category, priority, escalation risk)
- **Track 3** — Similar-case retrieval ("what worked before")
