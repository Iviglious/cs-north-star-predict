"""Decision-support models: routing suggestions, risk flags, similar cases.

Used by app.py (Gradio UI) and hackathon-notebook.ipynb (smoke test + evaluation).

Components:
    - Routing classifiers: team, category, priority (TF-IDF + logistic regression)
    - Escalation risk: balanced logistic regression + keyword flags
    - Similar-case retrieval: TF-IDF + cosine similarity (top 3 matches)

Design choice: lightweight, interpretable models with explicit confidence scores —
appropriate for decision-support where a human always makes the final call.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline

from data_loader import add_derived_fields, load_cases

# Transparent keyword rules — agents can see and challenge these flags
RISK_KEYWORDS = {
    "urgent",
    "failed",
    "crash",
    "outage",
    "duplicate",
    "refund",
    "locked",
    "disconnected",
    "blank",
    "missing",
    "error",
    "down",
}


@dataclass
class TriageSuggestion:
    """Structured output returned to the Gradio UI and notebook smoke test."""

    suggested_team: str
    team_confidence: float
    suggested_category: str
    category_confidence: float
    suggested_priority: str
    priority_confidence: float
    escalation_risk: float
    risk_flags: list[str]
    similar_cases: pd.DataFrame


class TriageEngine:
    """Train on historical cases and suggest routing for a new case summary."""

    def __init__(self, cases: pd.DataFrame | None = None):
        self.cases = add_derived_fields(cases if cases is not None else load_cases())
        self._train()

    def _train(self) -> None:
        """Fit all models at startup (fast on ~1,700 cases — no saved model file needed)."""
        train = self.cases[self.cases["case_summary"].str.len() > 0].copy()

        # Shared vectorizer for similar-case retrieval over enriched search_text
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_features=5000,
        )
        self.text_matrix = self.vectorizer.fit_transform(train["search_text"])
        self.train_cases = train.reset_index(drop=True)

        # Three routing classifiers — each returns top label + confidence
        self.team_model = self._fit_classifier(train["search_text"], train["assigned_team"])
        self.category_model = self._fit_classifier(train["search_text"], train["category"])
        self.priority_model = self._fit_classifier(train["search_text"], train["priority"])

        # Escalation uses class_weight=balanced because only ~13% of cases escalate
        self.escalation_model = Pipeline(
            [
                ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        )
        self.escalation_model.fit(train["search_text"], train["escalated"].astype(int))

    @staticmethod
    def _fit_classifier(text: pd.Series, labels: pd.Series) -> Pipeline:
        """Build a TF-IDF + logistic regression pipeline for a single routing target."""
        model = Pipeline(
            [
                ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)),
                ("clf", LogisticRegression(max_iter=1000)),
            ]
        )
        model.fit(text, labels)
        return model

    @staticmethod
    def _top_prediction(model: Pipeline, text: str) -> tuple[str, float]:
        """Return the highest-probability class and its confidence score."""
        proba = model.predict_proba([text])[0]
        idx = int(np.argmax(proba))
        label = model.classes_[idx]
        return str(label), float(proba[idx])

    def _keyword_flags(self, text: str) -> list[str]:
        """Rule-based risk flags — complements the escalation probability score."""
        lowered = text.lower()
        flags = [word for word in RISK_KEYWORDS if word in lowered]
        if any(token in lowered for token in ("urgent", "asap", "immediately")):
            flags.append("urgency_language")
        return sorted(set(flags))

    def _similar_cases(self, text: str, top_k: int = 3) -> pd.DataFrame:
        """Find historically similar cases so agents can see what worked before (Track 3)."""
        query = self.vectorizer.transform([text])
        scores = cosine_similarity(query, self.text_matrix).flatten()
        ranked = np.argsort(scores)[::-1][:top_k]
        rows = self.train_cases.iloc[ranked].copy()
        rows["similarity"] = scores[ranked]
        cols = [
            "case_id",
            "similarity",
            "case_summary",
            "category",
            "subcategory",
            "assigned_team",
            "priority",
            "status",
            "resolution_code",
            "escalated",
            "resolution_time_hours",
            "csat_score",
        ]
        return rows[cols]

    def suggest(self, case_summary: str, channel: str = "", plan_tier: str = "") -> TriageSuggestion:
        """Main entry point: combine routing, risk, and retrieval for one new case."""
        text = " ".join(part for part in [case_summary, channel, plan_tier] if part).strip()
        if not text:
            raise ValueError("Please enter a case summary.")

        team, team_conf = self._top_prediction(self.team_model, text)
        category, category_conf = self._top_prediction(self.category_model, text)
        priority, priority_conf = self._top_prediction(self.priority_model, text)
        escalation_risk = float(self.escalation_model.predict_proba([text])[0][1])
        flags = self._keyword_flags(case_summary)
        if escalation_risk >= 0.35:
            flags.append("historical_escalation_pattern")

        return TriageSuggestion(
            suggested_team=team,
            team_confidence=team_conf,
            suggested_category=category,
            category_confidence=category_conf,
            suggested_priority=priority,
            priority_confidence=priority_conf,
            escalation_risk=escalation_risk,
            risk_flags=sorted(set(flags)),
            similar_cases=self._similar_cases(text),
        )


def format_suggestion(result: TriageSuggestion) -> str:
    """Render routing and risk output as markdown for the Gradio UI."""
    lines = [
        "## Routing suggestion",
        f"- **Team:** {result.suggested_team} ({result.team_confidence:.0%} confidence)",
        f"- **Category:** {result.suggested_category} ({result.category_confidence:.0%} confidence)",
        f"- **Priority:** {result.suggested_priority} ({result.priority_confidence:.0%} confidence)",
        "",
        "## Risk signals",
        f"- **Escalation risk:** {result.escalation_risk:.0%}",
    ]
    if result.risk_flags:
        lines.append("- **Flags:** " + ", ".join(result.risk_flags))
    else:
        lines.append("- **Flags:** none detected")
    return "\n".join(lines)


def format_similar_cases(df: pd.DataFrame) -> str:
    """Render similar past cases as markdown for the Gradio UI."""
    if df.empty:
        return "No similar cases found."

    blocks = []
    for row in df.itertuples(index=False):
        resolution = row.resolution_code if pd.notna(row.resolution_code) and row.resolution_code else "n/a"
        csat = f"{row.csat_score:.0f}/5" if pd.notna(row.csat_score) else "n/a"
        blocks.append(
            "\n".join(
                [
                    f"### {row.case_id} (similarity {row.similarity:.0%})",
                    f"**Summary:** {row.case_summary}",
                    f"**Route:** {row.assigned_team} · {row.category}/{row.subcategory} · {row.priority}",
                    f"**Outcome:** {row.status} · {resolution} · escalated={row.escalated} · CSAT {csat}",
                ]
            )
        )
    return "\n\n".join(blocks)
