"""Northstar Desk triage assistant — Gradio decision-support prototype."""

from __future__ import annotations

import argparse

import gradio as gr

from triage_engine import TriageEngine, format_similar_cases, format_suggestion

engine = TriageEngine()

CHANNELS = ["", "email", "webchat", "phone", "in_app"]
PLAN_TIERS = ["", "Free", "Standard", "Pro", "Enterprise"]


def triage_case(case_summary: str, channel: str, plan_tier: str) -> tuple[str, str]:
    result = engine.suggest(case_summary, channel=channel, plan_tier=plan_tier)
    return format_suggestion(result), format_similar_cases(result.similar_cases)


demo = gr.Blocks(title="Northstar Desk Triage Assistant")
with demo:
    gr.Markdown(
        """
        # Northstar Desk Triage Assistant
        Paste a new case summary and get routing suggestions, escalation risk flags,
        and the most similar resolved cases from the last 12 months.
        """
    )
    with gr.Row():
        with gr.Column():
            case_summary = gr.Textbox(
                label="Case summary",
                placeholder="Customer says their Polaris Board export is blank and they need it for month-end.",
                lines=4,
            )
            channel = gr.Dropdown(label="Channel (optional)", choices=CHANNELS, value="")
            plan_tier = gr.Dropdown(label="Plan tier (optional)", choices=PLAN_TIERS, value="")
            submit = gr.Button("Suggest routing", variant="primary")
        with gr.Column():
            routing_output = gr.Markdown(label="Routing & risk")
            similar_output = gr.Markdown(label="Similar past cases")

    submit.click(
        triage_case,
        inputs=[case_summary, channel, plan_tier],
        outputs=[routing_output, similar_output],
    )

    gr.Examples(
        examples=[
            [
                "Urgent: credit card failed for Orbit ID renewal and account is about to lock.",
                "email",
                "Pro",
            ],
            [
                "CSV export is coming out blank for the sales report widget.",
                "webchat",
                "Enterprise",
            ],
            [
                "Need a VAT receipt for last month's subscription payment.",
                "email",
                "Standard",
            ],
        ],
        inputs=[case_summary, channel, plan_tier],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Northstar Desk Triage Assistant")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a temporary public gradio.live URL (72 hours)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Local host to bind")
    parser.add_argument("--port", type=int, default=7860, help="Local port")
    args = parser.parse_args()

    demo.launch(server_name=args.host, server_port=args.port, share=args.share)
