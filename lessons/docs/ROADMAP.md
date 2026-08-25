# Roadmap — Northstar Desk Triage Assistant

What we would do next with **one more week**, or if building this internally at Northstar.

---

## Week 1 — Make it production-credible

### Data & modelling

- [ ] **Retrain classifiers on summary-only text** — remove category/subcategory leakage from training features; re-benchmark on held-out cases.
- [ ] **Add subcategory suggestion** — hierarchical approach: predict category first, then subcategory within category (reduces 178-class sparsity).
- [ ] **Holdout evaluation set** — time-based split (train on Q1–Q3, test on Q4) to simulate real deployment.
- [ ] **SLA breach predictor** — flag cases likely to miss SLA before assignment.
- [ ] **Resolution time estimate** — simple regression or quantile model for workload planning.

### UX & workflow

- [ ] **Pre-fill from case ID** — agent enters `ND-2025-00xxxx`, tool loads summary from CMS.
- [ ] **"Accept / Override" buttons** — log agent decisions for feedback loop.
- [ ] **Suggested next steps** — aggregate common `resolution_code` from similar cases ("3 of 3 similar cases were resolved with `fixed` by engineering").
- [ ] **Mobile-friendly layout** — stack inputs/outputs vertically for narrow screens.
- [ ] **Ops dashboard tab (Track 2)** — volume trends, escalation spikes, filters by team/channel/date.

---

## Weeks 2–4 — Internal pilot

### Integration

- [ ] Embed as a **sidebar panel** in the existing case management tool (iframe or API).
- [ ] **Webhook on new case** — auto-run triage, attach suggestions as an internal note.
- [ ] **Batch mode** — score entire open queue overnight for team lead review.

### Governance & trust

- [ ] **Confidence thresholds** — only show suggestions above 60%; otherwise say "insufficient signal".
- [ ] **Fairness audit** — check routing suggestions and escalation scores across `plan_tier`, `region_uk`, tenure (audit only, not used in decisions).
- [ ] **Human-in-the-loop policy** — document when agents must override (e.g. Enterprise customers, security cases).
- [ ] **Explainability panel** — show top TF-IDF terms driving the suggestion.

### Monitoring

- [ ] Track **override rate** (agent changed team/priority vs suggestion).
- [ ] Track **escalation precision/recall** on flagged cases.
- [ ] Alert if **category distribution shifts** (data drift — new product area not in training data).
- [ ] Weekly report for team leads: top misroutes, emerging themes.

---

## Month 2+ — Scale and improve

| Area | Direction |
|---|---|
| **Embeddings** | Replace TF-IDF with sentence embeddings (e.g. `all-MiniLM-L6-v2`) for better paraphrase matching |
| **LLM assist (optional)** | Summarise similar cases into a "recommended playbook" — always grounded in retrieved cases, never free-form |
| **Multi-language** | UK customer base may include non-English summaries |
| **Closed-loop learning** | Retrain monthly on agent overrides and outcomes |
| **Security & privacy** | PII redaction in summaries before indexing; on-prem deployment option |

---

## Priority order (if we only get one week)

1. **Fix training leakage** (summary-only retrain) — highest impact on trust
2. **Accept/Override logging** — enables everything else
3. **Suggested next steps from similar cases** — biggest UX win for agents
4. **Time-based evaluation** — proves the model generalises
5. **Ops dashboard tab** — wins team-lead stakeholder buy-in

---

## Success criteria for a pilot

| KPI | Target |
|---|---|
| Agent adoption | >70% of agents use it on new cases |
| Override rate | <30% (suggestions are useful but not rigid) |
| Time to route | −20% vs baseline |
| Escalation surprise rate | −15% (cases flagged early that did escalate) |
| Agent satisfaction | ≥4/5 in post-pilot survey |
